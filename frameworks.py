"""Tracing hooks for frameworks other than PyTorch.

Two kinds of frameworks are supported:

* Array frameworks (JAX, TensorFlow): every eager op is recorded and mapped
  onto the same ATen-style event format the PyTorch tracer produces, so
  cudagen translates them with the same kernel templates. Cells run with
  `jax.disable_jit()` / `tf.config.run_functions_eagerly(True)` so compiled
  functions are traced op by op instead of disappearing into XLA/Grappler.
* Kernel frameworks (Numba CUDA, Triton, CuPy): the user already writes or
  JIT-compiles real GPU kernels, so we capture what was actually compiled
  (PTX for Numba/Triton, CUDA C++ source for CuPy) along with launch configs.

Hooks are installed lazily, once a framework has been imported, and only
record while a cell is executing (``ACTIVE`` is set by the engine).
"""
import contextlib
import inspect
import re
import sys

ACTIVE = None          # the engine's Recorder for the running cell, or None
_installed = set()

TRACKED_MODULES = ("jax", "tensorflow", "keras", "numba", "triton", "cupy", "flax", "optax")


def install_hooks():
    """Install hooks for every supported framework that is already imported."""
    for name, installer in (("jax", _install_jax), ("tensorflow", _install_tf),
                            ("numba", _install_numba), ("triton", _install_triton),
                            ("cupy", _install_cupy)):
        if name in sys.modules and name not in _installed:
            try:
                installer()
                _installed.add(name)
            except Exception as e:  # never break the user's cell over tracing
                print(f"[cuda-jupyter] could not hook {name}: {e}", file=sys.__stderr__)


@contextlib.contextmanager
def eager_mode():
    """Run jit-compiled JAX / TF code op by op so every op is visible."""
    with contextlib.ExitStack() as stack:
        if "jax" in sys.modules:
            stack.enter_context(sys.modules["jax"].disable_jit())
        if "tensorflow" in sys.modules:
            tf = sys.modules["tensorflow"]
            prev = tf.config.functions_run_eagerly()
            tf.config.run_functions_eagerly(True)
            stack.callback(tf.config.run_functions_eagerly, prev)
        yield


def _dtype(dt):
    s = str(getattr(dt, "name", dt))
    return {"float": "float32", "double": "float64", "half": "float16"}.get(s, s)


def _contig(shape):
    st, acc = [], 1
    for s in reversed(shape):
        st.append(acc)
        acc *= max(s, 1)
    return st[::-1]


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def view(t, shape=None, stride=None, off=None):
    return dict(t, shape=list(shape if shape is not None else t["shape"]),
                stride=list(stride if stride is not None else t["stride"]),
                off=t["off"] if off is None else off)


def broadcast_view(t, out_shape, bdims):
    """View of t broadcast to out_shape; bdims[i] is the out dim of t's dim i."""
    stride = [0] * len(out_shape)
    for i, d in enumerate(bdims):
        if t["shape"][i] == out_shape[d]:
            stride[d] = t["stride"][i]
    return view(t, out_shape, stride)


def gemm_event(A, B, C, lhs_contract, rhs_contract, lhs_batch=(), rhs_batch=()):
    """Express a general contraction as one strided batched GEMM, if it is one."""
    la, lb = len(A["shape"]), len(B["shape"])
    if len(lhs_contract) != 1 or len(rhs_contract) != 1 or len(lhs_batch) > 1 or len(rhs_batch) > 1:
        return None
    lfree = [d for d in range(la) if d not in lhs_contract and d not in lhs_batch]
    rfree = [d for d in range(lb) if d not in rhs_contract and d not in rhs_batch]
    if len(lfree) > 1 or len(rfree) > 1:
        return None
    ka, kb = lhs_contract[0], rhs_contract[0]
    M = A["shape"][lfree[0]] if lfree else 1
    N = B["shape"][rfree[0]] if rfree else 1
    K = A["shape"][ka]
    batch = A["shape"][lhs_batch[0]] if lhs_batch else 1
    sa = [A["stride"][lhs_batch[0]] if lhs_batch else 0, A["stride"][lfree[0]] if lfree else 0, A["stride"][ka]]
    sb = [B["stride"][rhs_batch[0]] if rhs_batch else 0, B["stride"][kb], B["stride"][rfree[0]] if rfree else 0]
    cs = _contig([batch, M, N])
    sc = [cs[0] if lhs_batch else 0, cs[1] if lfree else 0, cs[2] if rfree else 0]
    return "_gemm", {"A": A, "B": B, "M": M, "N": N, "K": K, "batch": batch, "sa": sa, "sb": sb, "sc": sc}


# ============================================================================ JAX
_JAX_EW = {
    "add": "add", "sub": "sub", "mul": "mul", "div": "div", "max": "maximum", "min": "minimum",
    "pow": "pow", "neg": "neg", "exp": "exp", "exp2": "exp2", "log": "log", "log1p": "log1p",
    "expm1": "expm1", "tanh": "tanh", "logistic": "sigmoid", "sqrt": "sqrt", "rsqrt": "rsqrt",
    "abs": "abs", "sin": "sin", "cos": "cos", "tan": "tan", "erf": "erf", "sign": "sign",
    "floor": "floor", "ceil": "ceil", "round": "round", "eq": "eq", "ne": "ne", "lt": "lt",
    "le": "le", "gt": "gt", "ge": "ge", "and": "logical_and", "or": "logical_or", "not": "logical_not",
    "rem": "fmod", "atan2": "atan2", "square": "square", "erf_inv": "erfinv", "xor": "bitwise_xor",
    "shift_right_logical": "shift_right", "shift_left": "shift_left",
}
_JAX_SILENT = {"stage", "stop_gradient", "random_seed", "random_wrap", "random_unwrap", "device_put",
               "copy_p", "pjit", "optimization_barrier", "reduce_precision"}
# primitives that only wrap other primitives: record what runs inside instead
_JAX_CALLS = {"jit", "pjit", "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr",
              "closed_call", "remat", "checkpoint", "custom_lin", "core_call"}


def _jax_key(a):
    try:
        return ("jax", a.unsafe_buffer_pointer())
    except Exception:
        return ("jax-id", id(a))


def jax_desc(rec, a, role):
    import numpy as np
    shape = list(a.shape)
    dt = _dtype(a.dtype)
    nbytes = _numel(shape) * np.dtype(a.dtype).itemsize if dt != "key<fry>" else 0
    sid = rec.ext_storage(_jax_key(a), nbytes, dt, role, a)
    return {"kind": "tensor", "sid": sid, "shape": shape, "stride": _contig(shape), "off": 0, "dtype": dt}


def _jax_map(prim, ins, params, outs):
    """Map a JAX primitive onto (aten_name, args) or None for 'untranslated'."""
    out = outs[0] if outs else None
    if prim in _JAX_EW:
        name = _JAX_EW[prim]
        if name == "square":
            return "pow", {"self": ins[0], "exponent": 2}
        if len(ins) == 1:
            return name, {"self": ins[0]}
        if name == "pow":
            return "pow", {"self": ins[0], "exponent": ins[1]}
        return name, {"self": ins[0], "other": ins[1]}
    if prim == "integer_pow":
        return "pow", {"self": ins[0], "exponent": params["y"]}
    if prim == "select_n" and len(ins) == 3:
        return "where", {"condition": ins[0], "self": ins[2], "other": ins[1]}
    if prim in ("convert_element_type", "bitcast_convert_type"):
        return "_to_copy", {"self": ins[0]}
    if prim == "broadcast_in_dim":
        return "clone", {"self": broadcast_view(ins[0], out["shape"], params["broadcast_dimensions"])}
    if prim in ("reshape", "squeeze", "expand_dims"):
        if out["sid"] == ins[0]["sid"]:
            return "view", {"self": ins[0]}
        return "clone", {"self": view(ins[0], out["shape"], _contig(out["shape"]))}
    if prim == "transpose":
        perm = params["permutation"]
        x = ins[0]
        return "clone", {"self": view(x, [x["shape"][p] for p in perm], [x["stride"][p] for p in perm])}
    if prim == "slice":
        x = ins[0]
        starts, limits = params["start_indices"], params["limit_indices"]
        steps = params.get("strides") or [1] * len(starts)
        off = x["off"] + sum(s * st for s, st in zip(starts, x["stride"]))
        shape = [(lim - s + k - 1) // k for s, lim, k in zip(starts, limits, steps)]
        return "clone", {"self": view(x, shape, [st * k for st, k in zip(x["stride"], steps)], off)}
    if prim == "dot_general":
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        return gemm_event(ins[0], ins[1], out, tuple(lc), tuple(rc), tuple(lb), tuple(rb))
    if prim in ("reduce_sum", "reduce_max", "reduce_min", "reduce_prod"):
        name = {"reduce_sum": "sum", "reduce_max": "amax", "reduce_min": "amin", "reduce_prod": "prod"}[prim]
        return name, {"self": ins[0], "dim": list(params["axes"])}
    if prim in ("argmax", "argmin"):
        return prim, {"self": ins[0], "dim": params["axes"][0]}
    if prim == "iota" and len(out["shape"]) == 1:
        return "arange", {"start": 0, "step": 1}
    if prim == "concatenate":
        return "cat", {"tensors": list(ins), "dim": params["dimension"]}
    if prim == "conv_general_dilated":
        dn = params["dimension_numbers"]
        ident = (0, 1, 2, 3)
        pads = params["padding"]
        if (tuple(dn.lhs_spec), tuple(dn.rhs_spec), tuple(dn.out_spec)) == (ident, ident, ident) \
                and all(d == 1 for d in params["lhs_dilation"]) and all(lo == hi for lo, hi in pads):
            return "convolution", {"input": ins[0], "weight": ins[1], "bias": None,
                                   "stride": list(params["window_strides"]),
                                   "padding": [lo for lo, _ in pads],
                                   "dilation": list(params["rhs_dilation"]), "transposed": False,
                                   "output_padding": [0, 0], "groups": params["feature_group_count"]}
    return None


def _install_jax():
    from jax._src import core

    orig_bind = core.Primitive.bind
    stack = []  # child-event lists of the concrete binds currently executing

    def bind(self, *args, **params):
        rec = ACTIVE
        if rec is None or rec.paused or any(isinstance(a, core.Tracer) for a in args):
            return orig_bind(self, *args, **params)
        stack.append([])
        try:
            out = orig_bind(self, *args, **params)
        finally:
            children = stack.pop()
        prim = self.name
        if prim in _JAX_CALLS or (prim not in _JAX_EW and children and _jax_map_safe(prim) is None):
            events = children           # a wrapper: keep what ran inside it
        elif prim in _JAX_SILENT:
            events = []
        else:
            events = [_jax_event(rec, prim, args, params, out)]
        if stack:
            stack[-1].extend(events)
        else:
            for ev in events:
                rec.add_event(ev)
        return out

    core.Primitive.bind = bind


def _jax_map_safe(prim):
    return prim if prim in _JAX_EW else None


def _jax_event(rec, prim, args, params, out):
    import jax
    ins = [jax_desc(rec, a, "in") if isinstance(a, jax.Array) else a for a in args]
    outs_raw = out if isinstance(out, (list, tuple)) else [out]
    outs = [jax_desc(rec, o, "out") for o in outs_raw if isinstance(o, jax.Array)]
    try:
        mapped = _jax_map(prim, ins, params, outs)
    except Exception:
        mapped = None
    name, bound = mapped if mapped else (prim, {f"x{i}": v for i, v in enumerate(ins)})
    shown = {k: v for k, v in params.items()
             if k in ("axes", "dimension_numbers", "shape", "broadcast_dimensions", "new_dtype", "y")}
    return {"fw": "jax", "orig": prim, "name": name if mapped else f"jax_{prim}", "overload": "",
            "args": bound, "outs": outs, "params": {k: str(v) for k, v in shown.items()}}


# ============================================================================ TensorFlow
_TF_EW = {
    "AddV2": "add", "Add": "add", "BiasAdd": "add", "Sub": "sub", "Mul": "mul", "RealDiv": "div",
    "Div": "div", "Maximum": "maximum", "Minimum": "minimum", "Pow": "pow", "Neg": "neg", "Exp": "exp",
    "Log": "log", "Log1p": "log1p", "Tanh": "tanh", "Sigmoid": "sigmoid", "Relu": "relu", "Elu": "elu",
    "Sqrt": "sqrt", "Rsqrt": "rsqrt", "Abs": "abs", "Sin": "sin", "Cos": "cos", "Erf": "erf",
    "Floor": "floor", "Ceil": "ceil", "Round": "round", "Sign": "sign", "Reciprocal": "reciprocal",
    "Inv": "reciprocal", "Greater": "gt", "GreaterEqual": "ge", "Less": "lt", "LessEqual": "le",
    "Equal": "eq", "NotEqual": "ne", "LogicalAnd": "logical_and", "LogicalOr": "logical_or",
    "LogicalNot": "logical_not", "Softplus": "softplus", "Square": "square", "FloorMod": "remainder",
    "Erfinv": "erfinv",
}
_TF_SILENT = {"Identity", "IdentityN", "VarHandleOp", "Shape", "ShapeN", "Rank", "Size",
              "BroadcastGradientArgs", "DestroyResourceOp", "VarIsInitializedOp", "NoOp", "Const",
              "Placeholder", "StopGradient", "PreventGradient"}


def _install_tf():
    import tensorflow as tf
    from tensorflow.python.framework import op_callbacks

    def callback(op_type, inputs, attrs, outputs, op_name=None, graph=None):
        rec = ACTIVE
        if isinstance(op_type, bytes):
            op_type = op_type.decode()
        if rec is None or rec.paused or graph is not None or op_type in _TF_SILENT:
            return None
        try:
            rec.add_event(_tf_event(rec, tf, op_type, inputs, attrs, outputs))
        except Exception as e:
            rec.add_event({"fw": "tf", "orig": op_type, "name": f"tf_{op_type}", "overload": "",
                           "args": {"error": str(e)[:80]}, "outs": []})
        return None

    op_callbacks.add_op_callback(callback)


def tf_desc(rec, tf, t, role, shape=None):
    if t.dtype == tf.resource:
        key = ("tf-res", id(t))
        sid = rec.ext_lookup(key)
        shp = shape if shape is not None else rec.tf_res_shape.get(key, [])
        if sid is None:
            sid = rec.ext_storage(key, 0, "float32", role, t)
        else:
            rec.grow_storage(sid, _numel(shp) * 4)
        if shape is not None:
            rec.tf_res_shape[key] = list(shape)
        dt = rec.storage_meta[sid]["dtype"]
        return {"kind": "tensor", "sid": sid, "shape": list(shp), "stride": _contig(shp), "off": 0, "dtype": dt}
    shp = [int(s) for s in t.shape]
    dt = _dtype(t.dtype)
    sid = rec.ext_storage(("tf", id(t)), _numel(shp) * t.dtype.size, dt, role, t)
    return {"kind": "tensor", "sid": sid, "shape": shp, "stride": _contig(shp), "off": 0, "dtype": dt}


def _tf_value(t):
    try:
        v = t.numpy()
        return v.tolist()
    except Exception:
        return None


def _tf_event(rec, tf, op, inputs, attrs, outputs):
    a = dict(zip(attrs[::2], attrs[1::2])) if attrs else {}
    outs_raw = list(outputs or [])
    # resource-variable ops write into the variable's buffer
    if op in ("AssignVariableOp", "AssignAddVariableOp", "AssignSubVariableOp"):
        val = tf_desc(rec, tf, inputs[1], "in")
        res = tf_desc(rec, tf, inputs[0], "in", shape=val["shape"])
        rec.storage_meta[res["sid"]]["dtype"] = val["dtype"]
        res["dtype"] = val["dtype"]
        name = {"AssignVariableOp": "copy_", "AssignAddVariableOp": "add_", "AssignSubVariableOp": "sub_"}[op]
        args = {"self": res, "src": val} if name == "copy_" else {"self": res, "other": val}
        return {"fw": "tf", "orig": op, "name": name, "overload": "", "args": args, "outs": [res]}
    if op == "ReadVariableOp":
        res = tf_desc(rec, tf, inputs[0], "in")
        if outs_raw:  # the read value aliases the variable buffer
            rec.alias_storage(("tf", id(outs_raw[0])), res["sid"], outs_raw[0])
        return {"fw": "tf", "orig": op, "name": "view", "overload": "", "args": {"self": res}, "outs": [res]}

    ins = [tf_desc(rec, tf, t, "in") for t in inputs]
    outs = [tf_desc(rec, tf, t, "out") for t in outs_raw]
    out = outs[0] if outs else None
    mapped = None
    if op in _TF_EW and _TF_EW[op]:
        name = _TF_EW[op]
        if name == "square":
            mapped = ("pow", {"self": ins[0], "exponent": 2})
        elif name == "softplus":
            mapped = ("softplus", {"self": ins[0], "beta": 1, "threshold": 20})
        elif name == "elu":
            mapped = ("elu", {"self": ins[0], "alpha": 1.0, "scale": 1.0, "input_scale": 1.0})
        elif len(ins) == 1:
            mapped = (name, {"self": ins[0]})
        elif name == "pow":
            mapped = ("pow", {"self": ins[0], "exponent": ins[1]})
        else:
            mapped = (name, {"self": ins[0], "other": ins[1]})
    elif op == "Relu6":
        mapped = ("hardtanh", {"self": ins[0], "min_val": 0.0, "max_val": 6.0})
    elif op == "LeakyRelu":
        mapped = ("leaky_relu", {"self": ins[0], "negative_slope": float(a.get("alpha", 0.2))})
    elif op == "ReluGrad":
        mapped = ("threshold_backward", {"grad_output": ins[0], "self": ins[1], "threshold": 0.0})
    elif op == "SigmoidGrad":
        mapped = ("sigmoid_backward", {"grad_output": ins[1], "output": ins[0]})
    elif op == "TanhGrad":
        mapped = ("tanh_backward", {"grad_output": ins[1], "output": ins[0]})
    elif op in ("Select", "SelectV2"):
        mapped = ("where", {"condition": ins[0], "self": ins[1], "other": ins[2]})
    elif op == "BiasAddGrad":
        mapped = ("sum", {"self": ins[0], "dim": list(range(len(ins[0]["shape"]) - 1))})
    elif op == "Cast":
        mapped = ("_to_copy", {"self": ins[0]})
    elif op in ("Reshape", "ExpandDims", "Squeeze"):
        rec.alias_storage(("tf", id(outs_raw[0])), ins[0]["sid"], outs_raw[0])
        out = dict(out, sid=ins[0]["sid"])
        mapped = ("view", {"self": ins[0]})
        outs = [out]
    elif op == "Transpose":
        perm = _tf_value(inputs[1])
        x = ins[0]
        mapped = ("clone", {"self": view(x, [x["shape"][p] for p in perm], [x["stride"][p] for p in perm])})
    elif op == "MatMul":
        A, B = ins[0], ins[1]
        if a.get("transpose_a"):
            A = view(A, A["shape"][::-1], A["stride"][::-1])
        if a.get("transpose_b"):
            B = view(B, B["shape"][::-1], B["stride"][::-1])
        mapped = gemm_event(A, B, out, (1,), (0,))
    elif op in ("BatchMatMulV2", "BatchMatMulV3", "BatchMatMul") and len(ins[0]["shape"]) == 3:
        A, B = ins[0], ins[1]
        if a.get("adj_x"):
            A = view(A, [A["shape"][0], A["shape"][2], A["shape"][1]], [A["stride"][0], A["stride"][2], A["stride"][1]])
        if a.get("adj_y"):
            B = view(B, [B["shape"][0], B["shape"][2], B["shape"][1]], [B["stride"][0], B["stride"][2], B["stride"][1]])
        mapped = gemm_event(A, B, out, (2,), (1,), (0,), (0,))
    elif op in ("Sum", "Mean", "Max", "Min", "Prod"):
        axes = _tf_value(inputs[1])
        axes = [axes] if isinstance(axes, int) else list(axes or [])
        name = {"Sum": "sum", "Mean": "mean", "Max": "amax", "Min": "amin", "Prod": "prod"}[op]
        mapped = (name, {"self": ins[0], "dim": axes})
    elif op in ("ArgMax", "ArgMin"):
        mapped = (op.lower(), {"self": ins[0], "dim": _tf_value(inputs[1])})
    elif op in ("Softmax", "LogSoftmax"):
        mapped = ("_softmax" if op == "Softmax" else "_log_softmax", {"self": ins[0], "dim": -1})
    elif op == "ConcatV2":
        mapped = ("cat", {"tensors": ins[:-1], "dim": _tf_value(inputs[-1])})
    elif op == "Fill":
        mapped = ("full", {"fill_value": _tf_value(inputs[1])})
    elif op in ("ZerosLike", "OnesLike"):
        mapped = ("zeros_like" if op == "ZerosLike" else "ones_like", {"self": ins[0]})
    elif op == "RandomStandardNormal":
        mapped = ("randn", {})
    elif op == "RandomUniform":
        mapped = ("rand", {})
    elif op == "Conv2D" and a.get("data_format", b"NHWC") in (b"NCHW", "NCHW"):
        st = list(a.get("strides", [1, 1, 1, 1]))[2:]
        mapped = ("convolution", {"input": ins[0], "weight": view(ins[1], [ins[1]["shape"][i] for i in (3, 2, 0, 1)],
                                                                  [ins[1]["stride"][i] for i in (3, 2, 0, 1)]),
                                  "bias": None, "stride": st, "padding": [0, 0], "dilation": [1, 1],
                                  "transposed": False, "output_padding": [0, 0], "groups": 1})
    if mapped is None:
        return {"fw": "tf", "orig": op, "name": f"tf_{op}", "overload": "",
                "args": {f"x{i}": v for i, v in enumerate(ins)}, "outs": outs}
    return {"fw": "tf", "orig": op, "name": mapped[0], "overload": "", "args": mapped[1], "outs": outs}


# ============================================================================ kernel frameworks
_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")
_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"')


def ptx_line_map(ptx, filename):
    """Map PTX lines to cell lines using the .file/.loc debug directives."""
    lines = ptx.split("\n")
    # .file directives may come after the .loc lines that use them (Triton).
    files = {m.group(1): m.group(2) for m in map(_FILE.match, lines) if m}
    cur, out = 0, []
    for ln in lines:
        m = _LOC.match(ln)
        if m:
            cur = int(m.group(2)) if files.get(m.group(1), "").endswith(filename) else 0
        out.append(cur)
    return out


def _py_source(fn):
    try:
        return inspect.getsource(fn)
    except Exception:
        return ""


def _comment_block(text):
    return "\n".join("// " + ln if ln else "//" for ln in text.rstrip().split("\n"))


def _install_numba():
    import numba.cuda.dispatcher as nd

    orig_call = nd.CUDADispatcher.call

    def call(self, args, griddim, blockdim, stream, sharedmem):
        res = orig_call(self, args, griddim, blockdim, stream, sharedmem)
        rec = ACTIVE
        if rec is not None and not rec.paused:
            line, _ = rec._site()
            key = ("numba", id(self), str(griddim), str(blockdim))
            if key not in rec.file_keys:
                rec.file_keys.add(key)
                try:
                    asm = self.inspect_asm()
                    sig, ptx = list(asm.items())[-1] if isinstance(asm, dict) else ("", asm)
                except Exception as e:
                    sig, ptx = "", f"// could not get PTX: {e}"
                name = self.py_func.__name__
                head = (f"// Numba CUDA kernel `{name}` compiled to PTX by NVVM\n"
                        f"// signature: {sig}\n// launch: {name}[{_dim(griddim)}, {_dim(blockdim)}]"
                        f"  (grid, block)   from L{line}\n//\n" + _comment_block(_py_source(self.py_func)) + "\n\n")
                rec.add_file(f"{name}.ptx", "ptx", head + ptx, launch_line=line)
        return res

    nd.CUDADispatcher.call = call


def _dim(d):
    if isinstance(d, (tuple, list)):
        d = [x for x in d if x is not None]
        return "(" + ", ".join(map(str, d)) + ")" if len(d) > 1 else str(d[0] if d else 1)
    return str(d)


def _install_triton():
    import triton.runtime.jit as tj

    orig_run = tj.JITFunction.run

    def run(self, *args, grid, warmup, **kwargs):
        kernel = orig_run(self, *args, grid=grid, warmup=warmup, **kwargs)
        rec = ACTIVE
        if rec is not None and not rec.paused and not warmup and kernel is not None:
            line, _ = rec._site()
            name = getattr(kernel, "name", self.fn.__name__)
            key = ("triton", name, id(kernel))
            if key not in rec.file_keys:
                rec.file_keys.add(key)
                asm = getattr(kernel, "asm", {}) or {}
                ptx = asm.get("ptx", "// PTX not available")
                if isinstance(ptx, bytes):
                    ptx = ptx.decode()
                g = grid if not callable(grid) else "grid(meta)"
                meta = getattr(kernel, "metadata", None)
                warps = getattr(meta, "num_warps", "?")
                head = (f"// Triton kernel `{name}` lowered Triton IR -> TTGIR -> LLVM -> PTX\n"
                        f"// launch: {self.fn.__name__}[{g}]  num_warps={warps}  from L{line}\n//\n"
                        + _comment_block(_py_source(self.fn)) + "\n\n")
                rec.add_file(f"{name}.ptx", "ptx", head + ptx, launch_line=line)
        return kernel

    tj.JITFunction.run = run


def _install_cupy():
    import cupy.cuda.compiler as cc

    orig = cc._compile_module_with_cache

    def compile_module(source, *args, **kwargs):
        rec = ACTIVE
        if rec is not None and not rec.paused and isinstance(source, str):
            line, _ = rec._site()
            m = re.search(r'__global__\s+void\s+(\w+)', source)
            name = m.group(1) if m else "kernel"
            name = name if name.startswith("cupy_") else "cupy_" + name
            if ("cupy", source) not in rec.file_keys:
                rec.file_keys.add(("cupy", source))
                head = (f"// CUDA C++ that CuPy generated and compiled with NVRTC for L{line}\n"
                        "// (CuPy caches compiled kernels, so each one appears the first time it is built)\n\n")
                rec.add_file(f"{name}.cu", "cpp", head + source, launch_line=line)
        return orig(source, *args, **kwargs)

    cc._compile_module_with_cache = compile_module
