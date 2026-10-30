"""Translate a recorded ATen op trace into a readable CUDA C++ translation unit.

Generation is template based: each ATen op we know maps to a small hand-written
kernel (elementwise, tiled GEMM, reductions, softmax, layer norm, conv, ...),
specialised with the dtypes, shapes and strides observed while the cell ran.
Ops without a template are kept as commented ATen calls so the trace stays
complete. Every emitted line remembers the Python line that caused it, which
the UI uses to cross-highlight the two panes.
"""
import math
from collections import OrderedDict

THREADS = 256
MAX_DIMS = 6
MAX_EMITTED_OPS = 600

CTYPE = {
    "float32": "float", "float64": "double", "float16": "__half", "bfloat16": "__nv_bfloat16",
    "int64": "int64_t", "int32": "int32_t", "int16": "int16_t", "int8": "int8_t",
    "uint8": "uint8_t", "uint16": "uint16_t", "uint32": "uint32_t", "uint64": "uint64_t", "bool": "bool",
}
FLOAT_TYPES = {"float32", "float64", "float16", "bfloat16"}

VIEW_OPS = {
    "view", "_unsafe_view", "reshape", "_reshape_alias", "t", "transpose", "permute", "expand",
    "unsqueeze", "squeeze", "slice", "select", "as_strided", "unflatten", "flatten", "narrow",
    "unfold", "split", "split_with_sizes", "chunk", "unbind", "diagonal", "view_as", "expand_as",
    "movedim", "numpy_T", "mT", "swapaxes", "view_as_real", "view_as_complex",
}
SILENT_OPS = {"detach", "alias", "is_same_size", "_has_compatible_shallow_copy_type", "sym_size",
              "sym_stride", "sym_numel", "sym_storage_offset", "set_", "resize_", "_assert_async"}
ALLOC_OPS = {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided"}


# --------------------------------------------------------------------------- helpers
def ctype(dt):
    return CTYPE.get(dt, "float")


def acctype(dt):
    if dt in ("float16", "bfloat16"):
        return "float"
    return ctype(dt)


def is_tensor(v):
    return isinstance(v, dict) and v.get("kind") == "tensor"


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def contiguous_strides(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= max(s, 1)
    return out[::-1]


def is_contig(t):
    return all(sz == 1 or st == ex for sz, st, ex in
               zip(t["shape"], t["stride"], contiguous_strides(t["shape"])))


def cdiv(a, b):
    return (a + b - 1) // b


def next_pow2(n, lo=32, hi=1024):
    p = lo
    while p < n and p < hi:
        p *= 2
    return p


def bcast_strides(t, out_shape):
    nd = len(out_shape)
    shift = nd - len(t["shape"])
    res = []
    for d in range(nd):
        k = d - shift
        if k < 0 or t["shape"][k] != out_shape[d]:
            res.append(0)
        else:
            res.append(t["stride"][k])
    return res


def strided_lit(sizes, strides):
    return "Strided{%d, {%s}, {%s}}" % (len(sizes), ", ".join(map(str, sizes)),
                                        ", ".join(map(str, strides)))


def lit(v, ct):
    if isinstance(v, bool):
        return "true" if v else "false"
    if ct in ("float", "double", "__half", "__nv_bfloat16") or isinstance(v, float):
        v = float(v)
        if math.isinf(v):
            return "INFINITY" if v > 0 else "-INFINITY"
        if math.isnan(v):
            return "NAN"
        s = repr(v)
        if "e" not in s and "." not in s:
            s += ".0"
        return s if ct == "double" else s + "f"
    v = int(v)
    return str(v) + ("LL" if abs(v) > 2**31 - 1 else "")


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def as_list(v, n):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        return list(v) * n if len(v) == 1 else list(v)
    return [v] * n


def norm_dim(d, ndim):
    return d + ndim if d < 0 else d


# --------------------------------------------------------------------------- elementwise
class EwCtx:
    """Collects the tensor operands and scalar parameters of one elementwise op."""

    SYMS = "abcdefgh"
    RENAME = {"self": "base", "min": "lo", "max": "hi", "end": "end_", "min_val": "lo", "max_val": "hi"}

    def __init__(self):
        self.tensors = []
        self.scalars = []

    def t(self, v, name="other"):
        if is_tensor(v):
            self.tensors.append(v)
            return self.SYMS[len(self.tensors) - 1]
        return self.s(v, name)

    def s(self, v, name):
        base = self.RENAME.get(name, name)
        name, k = base, 2
        taken = {n for n, _ in self.scalars} | set(self.SYMS) | {"i", "n", "out"}
        while name in taken:
            name, k = f"{base}{k}", k + 1
        self.scalars.append((name, v))
        return name


def _axpy(sign):
    def f(c, b):
        x, y = c.t(b["self"], "self"), c.t(b["other"])
        alpha = b.get("alpha", 1)
        if alpha in (None, 1):
            return f"{x} {sign} {y}"
        return f"{x} {sign} {c.s(alpha, 'alpha')} * {y}"
    return f


def _rsub(c, b):
    x, y = c.t(b["self"], "self"), c.t(b["other"])
    alpha = b.get("alpha", 1)
    return f"{y} - {x}" if alpha in (None, 1) else f"{y} - {c.s(alpha, 'alpha')} * {x}"


def _u(fmt):
    return lambda c, b: fmt.format(a=c.t(b["self"], "self"))


def _bin(fmt, left="self", right="other"):
    def f(c, b):
        a = c.t(b[left], left)
        bb = c.t(b[right], right)
        return fmt.format(a=a, b=bb)
    return f


def _div(c, b):
    a, d = c.t(b["self"], "self"), c.t(b["other"])
    mode = b.get("rounding_mode")
    if mode == "trunc":
        return f"trunc((double){a} / {d})"
    if mode == "floor":
        return f"floor((double){a} / {d})"
    return f"{a} / {d}"


def _pow(c, b):
    if not is_tensor(b.get("self")):
        base = c.s(b["self"], "base")
        return f"pow({base}, {c.t(b['exponent'], 'exponent')})"
    a = c.t(b["self"], "self")
    e = b["exponent"]
    if not is_tensor(e):
        if e == 2:
            return f"{a} * {a}"
        if e == 0.5:
            return f"sqrt({a})"
        if e == -1:
            return f"1 / {a}"
    return f"pow({a}, {c.t(e, 'exponent')})"


def _clamp(lo_key, hi_key):
    def f(c, b):
        e = c.t(b["self"], "self")
        if b.get(lo_key) is not None:
            lo = c.t(b[lo_key], "min")
            e = f"({e} < {lo} ? {lo} : {e})"
        if b.get(hi_key) is not None:
            hi = c.t(b[hi_key], "max")
            e = f"({e} > {hi} ? {hi} : {e})"
        return e
    return f


def _gelu(c, b):
    a = c.t(b["self"], "self")
    if b.get("approximate") == "tanh":
        return f"0.5f * {a} * (1 + tanh(0.7978845608f * ({a} + 0.044715f * {a} * {a} * {a})))"
    return f"0.5f * {a} * (1 + erf({a} * 0.7071067812f))"


def _gelu_bwd(c, b):
    g, x = c.t(b["grad_output"], "grad_output"), c.t(b["self"], "self")
    return (f"{g} * (0.5f * (1 + erf({x} * 0.7071067812f)) + "
            f"{x} * 0.3989422804f * exp(-0.5f * {x} * {x}))")


def _mse_bwd(c, b):
    g, x, y = c.t(b["grad_output"]), c.t(b["self"]), c.t(b["target"])
    e = f"2 * ({x} - {y}) * {g}"
    if b.get("reduction") == 1:
        e += f" / {numel(b['self']['shape'])}.0f"
    return e


def _const(key=None, value=None):
    def f(c, b):
        return c.s(b[key], "value") if key else value
    return f


def _arange(c, b):
    start = b.get("start", 0) or 0
    step = b.get("step", 1) or 1
    s = c.s(start, "start") if start != 0 else None
    st = c.s(step, "step") if step != 1 else None
    e = f"i * {st}" if st else "i"
    return f"{s} + {e}" if s else e


EW = {
    "add": _axpy("+"), "sub": _axpy("-"), "rsub": _rsub,
    "mul": _bin("{a} * {b}"), "div": _div,
    "remainder": _bin("{a} - floor((double){a} / {b}) * {b}"), "fmod": _bin("fmod({a}, {b})"),
    "maximum": _bin("({a} > {b} ? {a} : {b})"), "minimum": _bin("({a} < {b} ? {a} : {b})"),
    "pow": _pow, "atan2": _bin("atan2({a}, {b})"),
    "eq": _bin("{a} == {b}"), "ne": _bin("{a} != {b}"), "lt": _bin("{a} < {b}"),
    "le": _bin("{a} <= {b}"), "gt": _bin("{a} > {b}"), "ge": _bin("{a} >= {b}"),
    "logical_and": _bin("{a} && {b}"), "logical_or": _bin("{a} || {b}"),
    "logical_not": _u("!{a}"), "erfinv": _u("erfinv({a})"),
    "bitwise_xor": _bin("{a} ^ {b}"), "bitwise_and": _bin("{a} & {b}"), "bitwise_or": _bin("{a} | {b}"),
    "shift_right": _bin("{a} >> {b}"), "shift_left": _bin("{a} << {b}"),
    "relu": _u("{a} > 0 ? {a} : 0"), "sigmoid": _u("1 / (1 + exp(-{a}))"),
    "tanh": _u("tanh({a})"), "exp": _u("exp({a})"), "exp2": _u("exp2({a})"),
    "log": _u("log({a})"), "log2": _u("log2({a})"), "log10": _u("log10({a})"),
    "log1p": _u("log1p({a})"), "expm1": _u("expm1({a})"), "sqrt": _u("sqrt({a})"),
    "rsqrt": _u("1 / sqrt({a})"), "neg": _u("-{a}"), "abs": _u("fabs({a})"),
    "sin": _u("sin({a})"), "cos": _u("cos({a})"), "tan": _u("tan({a})"), "erf": _u("erf({a})"),
    "floor": _u("floor({a})"), "ceil": _u("ceil({a})"), "round": _u("rint({a})"),
    "trunc": _u("trunc({a})"), "sign": _u("({a} > 0) - ({a} < 0)"), "sgn": _u("({a} > 0) - ({a} < 0)"),
    "reciprocal": _u("1 / {a}"), "silu": _u("{a} / (1 + exp(-{a}))"),
    "mish": _u("{a} * tanh(log1p(exp({a})))"), "gelu": _gelu,
    "softplus": lambda c, b: (lambda a, beta, th: f"{a} * {beta} > {th} ? {a} : log1p(exp({a} * {beta})) / {beta}")(
        c.t(b["self"], "self"), c.s(b["beta"], "beta"), c.s(b["threshold"], "threshold")),
    "leaky_relu": lambda c, b: (lambda a, s: f"{a} > 0 ? {a} : {a} * {s}")(
        c.t(b["self"], "self"), c.s(b["negative_slope"], "negative_slope")),
    "elu": lambda c, b: (lambda a, al, sc, isc: f"{a} > 0 ? {sc} * {a} : {sc} * {al} * (exp({a} * {isc}) - 1)")(
        c.t(b["self"], "self"), c.s(b["alpha"], "alpha"), c.s(b["scale"], "scale"),
        c.s(b["input_scale"], "input_scale")),
    "threshold": lambda c, b: (lambda a, th, v: f"{a} > {th} ? {a} : {v}")(
        c.t(b["self"], "self"), c.s(b["threshold"], "threshold"), c.s(b["value"], "value")),
    "clamp": _clamp("min", "max"), "clamp_min": _clamp("min", None), "clamp_max": _clamp(None, "max"),
    "hardtanh": _clamp("min_val", "max_val"),
    "where": lambda c, b: (lambda m, x, y: f"{m} ? {x} : {y}")(
        c.t(b["condition"], "condition"), c.t(b["self"], "self"), c.t(b["other"])),
    "masked_fill": lambda c, b: (lambda x, m, v: f"{m} ? {v} : {x}")(
        c.t(b["self"], "self"), c.t(b["mask"], "mask"), c.t(b["value"], "value")),
    "addcmul": lambda c, b: (lambda x, t1, t2, v: f"{x} + {v} * {t1} * {t2}")(
        c.t(b["self"], "self"), c.t(b["tensor1"]), c.t(b["tensor2"]), c.s(b["value"], "value")),
    "addcdiv": lambda c, b: (lambda x, t1, t2, v: f"{x} + {v} * {t1} / {t2}")(
        c.t(b["self"], "self"), c.t(b["tensor1"]), c.t(b["tensor2"]), c.s(b["value"], "value")),
    "lerp": lambda c, b: (lambda x, e, w: f"{x} + {w} * ({e} - {x})")(
        c.t(b["self"], "self"), c.t(b["end"], "end"), c.t(b["weight"], "weight")),
    # autograd backward formulas
    "threshold_backward": lambda c, b: (lambda g, x, th: f"{x} <= {th} ? 0 : {g}")(
        c.t(b["grad_output"]), c.t(b["self"]), c.s(b["threshold"], "threshold")),
    "sigmoid_backward": _bin("{a} * {b} * (1 - {b})", "grad_output", "output"),
    "tanh_backward": _bin("{a} * (1 - {b} * {b})", "grad_output", "output"),
    "gelu_backward": _gelu_bwd,
    "silu_backward": _bin("{a} * (1 / (1 + exp(-{b}))) * (1 + {b} * (1 - 1 / (1 + exp(-{b}))))",
                          "grad_output", "self"),
    "leaky_relu_backward": lambda c, b: (lambda g, x, s: f"{x} > 0 ? {g} : {g} * {s}")(
        c.t(b["grad_output"]), c.t(b["self"]), c.s(b["negative_slope"], "negative_slope")),
    "hardtanh_backward": lambda c, b: (lambda g, x, lo, hi: f"({x} <= {lo} || {x} >= {hi}) ? 0 : {g}")(
        c.t(b["grad_output"]), c.t(b["self"]), c.s(b["min_val"], "min"), c.s(b["max_val"], "max")),
    "mse_loss_backward": _mse_bwd,
    # producers / copies
    "fill": lambda c, b: c.t(b["value"], "value"),
    "zero": _const(value="0"), "zeros": _const(value="0"), "zeros_like": _const(value="0"),
    "ones": _const(value="1"), "ones_like": _const(value="1"), "new_ones": _const(value="1"),
    "new_zeros": _const(value="0"),
    "full": _const("fill_value"), "full_like": _const("fill_value"), "new_full": _const("fill_value"),
    "scalar_tensor": _const("s"), "arange": _arange,
    "copy": lambda c, b: c.t(b["src"], "src"),
    "_to_copy": _u("{a}"), "clone": _u("{a}"), "_copy_from": _u("{a}"),
}
MEMSET_ZERO = {"zero", "zeros", "zeros_like", "new_zeros"}
COPY_OPS = {"copy", "_to_copy", "clone", "_copy_from"}


# --------------------------------------------------------------------------- reductions
REDUCE = {
    # kind: (init, accumulate element y into x, merge partials x and y, finalize)
    "sum": ("0", "{x} + {y}", "{x} + {y}", "{acc}"),
    "mean": ("0", "{x} + {y}", "{x} + {y}", "{acc} / {n}"),
    "prod": ("1", "{x} * {y}", "{x} * {y}", "{acc}"),
    "amax": ("LOWEST", "({y} > {x} ? {y} : {x})", "({y} > {x} ? {y} : {x})", "{acc}"),
    "amin": ("HIGHEST", "({y} < {x} ? {y} : {x})", "({y} < {x} ? {y} : {x})", "{acc}"),
    "norm2": ("0", "{x} + {y} * {y}", "{x} + {y}", "sqrt({acc})"),
}


def _init_lit(init, dt):
    if init == "LOWEST":
        return "-INFINITY" if dt in FLOAT_TYPES else "INT64_MIN"
    if init == "HIGHEST":
        return "INFINITY" if dt in FLOAT_TYPES else "INT64_MAX"
    return init


PRELUDE_STRIDED = """\
// Tensor geometry passed by value to kernels that handle non-contiguous / broadcast operands.
#define MAX_DIMS 6
struct Strided {
    int ndim;
    int64_t sizes[MAX_DIMS];
    int64_t strides[MAX_DIMS];  // in elements; 0 = broadcast along that dim
};

// Map a linear (row-major) element index to a storage offset.
__device__ __forceinline__ int64_t offset_of(int64_t linear, const Strided& s) {
    int64_t off = 0;
    for (int d = s.ndim - 1; d >= 0; --d) {
        off += (linear % s.sizes[d]) * s.strides[d];
        linear /= s.sizes[d];
    }
    return off;
}"""

PRELUDE_BLOCK_REDUCE = """\
// Block-wide reductions: warp shuffle first, then one value per warp through shared memory.
template <typename T>
__device__ T block_reduce_sum(T v) {
    __shared__ T warp_buf[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) warp_buf[warp] = v;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < n_warps) ? warp_buf[threadIdx.x] : T(0);
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) warp_buf[0] = v;
    __syncthreads();
    return warp_buf[0];
}

template <typename T>
__device__ T block_reduce_max(T v) {
    __shared__ T warp_buf[32];
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffff, v, o));
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) warp_buf[warp] = v;
    __syncthreads();
    int n_warps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < n_warps) ? warp_buf[threadIdx.x] : T(-INFINITY);
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) warp_buf[0] = v;
    __syncthreads();
    return warp_buf[0];
}"""


# --------------------------------------------------------------------------- generator
class Gen:
    def __init__(self, events, storage_meta, names, modules, source, cell, seed):
        self.events = events
        self.meta = storage_meta
        self.pynames = names
        self.modules = modules
        self.src_lines = source.splitlines()
        self.cell = cell
        self.seed = seed
        self.kernels = OrderedDict()
        self.kernel_keys = {}
        self.body = []
        self.bufnames = {}
        self.used_names = set()
        self.n_tmp = 0
        self.n_in = 0
        self.h2d = set()
        self.prelude = set()
        self.philox = 0
        self.n_item = 0
        self.n_lit = 0
        self.cur_line = 0
        self.stats = {"ops": 0, "kernels": 0, "launches": 0, "views": 0, "fallback": 0, "memcpy": 0}
        self.trace = []

    # -- output helpers --------------------------------------------------------
    def emit(self, text=""):
        for ln in text.split("\n"):
            self.body.append(("    " + ln if ln else "", self.cur_line))

    def launch(self, kname, grid, block, args, comment=None):
        self.stats["launches"] += 1
        head = f"{kname}<<<{grid}, {block}>>>("
        one = head + ", ".join(args) + ");"
        if comment:
            self.emit(f"// {comment}")
        if len(one) <= 100:
            self.emit(one)
        else:
            self.emit(head)
            for k, a in enumerate(args):
                self.emit("        " + a + ("," if k < len(args) - 1 else ");"))

    def kernel(self, key, base, build):
        if key in self.kernel_keys:
            return self.kernel_keys[key]
        name, k = base, 2
        while name in self.kernels:
            name = base.replace("_kernel", f"_v{k}_kernel") if base.endswith("_kernel") else f"{base}{k}"
            k += 1
        self.kernel_keys[key] = name
        self.kernels[name] = (build(name), self.cur_line)
        self.stats["kernels"] += 1
        return name

    # -- buffers -----------------------------------------------------------------
    def buf(self, sid, hint="buf"):
        if sid in self.bufnames:
            return self.bufnames[sid]
        nm = self.pynames.get(sid)
        if not nm:
            if self.meta[sid]["first"] == "in" and sid not in self.h2d:
                nm = f"in{self.n_in}"
                self.n_in += 1
            else:
                nm = f"t{self.n_tmp}_{hint.strip('_')}"
                self.n_tmp += 1
        base, k = nm, 2
        while nm in self.used_names:
            nm, k = f"{base}_{k}", k + 1
        self.used_names.add(nm)
        self.bufnames[sid] = nm
        return nm

    def ptr(self, t, hint="buf"):
        if t["sid"] is None:
            return "nullptr"
        name = self.buf(t["sid"], hint)
        bdt = self.meta[t["sid"]]["dtype"]
        expr = name
        if bdt != t["dtype"]:
            expr = f"reinterpret_cast<{ctype(t['dtype'])}*>({name})"
        return f"({expr} + {t['off']})" if t["off"] else expr

    def desc(self, v):
        if is_tensor(v):
            nm = self.buf(v["sid"]) if v["sid"] is not None else "?"
            return f"{nm}[{', '.join(map(str, v['shape']))}]"
        if isinstance(v, list):
            if v and all(is_tensor(x) for x in v):
                return "[" + ", ".join(self.desc(x) for x in v) + "]"
            return "[" + ", ".join(map(str, v)) + "]"
        if isinstance(v, dict):
            return str(v.get("v"))
        return repr(v)

    def op_comment(self, ev, count):
        b, outs = ev["args"], ev["outs"]
        for o in outs:
            self.buf(o["sid"], ev["name"]) if o["sid"] is not None else None
        parts = [self.desc(v) for k, v in b.items()
                 if v is not None and not (isinstance(v, dict) and v.get("kind") == "other")
                 and not (ev["name"] == "_gemm" and k not in ("A", "B"))]
        s = f"{op_label(ev)}({', '.join(parts)})"
        if ev.get("params"):
            s += " " + " ".join(f"{k}={v}" for k, v in ev["params"].items())
        if len(s) > 96:
            s = s[:93] + "...)"
        if outs:
            s += " -> " + ", ".join(self.desc(o) for o in outs)
        if count > 1:
            s += f"   [x{count}]"
        self.emit("// " + s)

    # -- elementwise ----------------------------------------------------------------
    def ew(self, label, expr, c, out):
        ins = c.tensors
        oshape = out["shape"]
        n = numel(oshape)
        if n == 0:
            self.emit("// empty tensor: nothing to launch")
            return
        if len(oshape) > MAX_DIMS or any(len(t["shape"]) > MAX_DIMS for t in ins):
            self.emit(f"// > {MAX_DIMS} dims: collapse dimensions before launching (omitted)")
            return
        flat = is_contig(out) and all(t["shape"] == oshape and is_contig(t) for t in ins)
        out_ct = ctype(out["dtype"])
        if out["dtype"] == "bool" and ins:
            compute = acctype(ins[0]["dtype"])
        elif out["dtype"] == "bool":
            compute = "float"
        else:
            compute = acctype(out["dtype"])
        sc = [(nm, compute if compute != "bool" else "float", v) for nm, v in c.scalars]
        key = ("ew", label, expr, tuple(t["dtype"] for t in ins), out["dtype"], flat,
               tuple(t for _, t, _ in sc))

        def build(name):
            params = [f"const {ctype(t['dtype'])}* in{k}" for k, t in enumerate(ins)]
            params.append(f"{out_ct}* out")
            params += [f"{t} {nm}" for nm, t, _ in sc]
            if not flat:
                params += [f"Strided d{k}" for k in range(len(ins))] + ["Strided dout"]
            params.append("int64_t n")
            lines = [f"// out = {expr}",
                     f"__global__ void {name}({', '.join(params)}) {{",
                     "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                     "    if (i >= n) return;"]
            for k, t in enumerate(ins):
                idx = "i" if flat else f"offset_of(i, d{k})"
                lines.append(f"    {acctype(t['dtype'])} {EwCtx.SYMS[k]} = in{k}[{idx}];")
            oidx = "i" if flat else "offset_of(i, dout)"
            lines.append(f"    out[{oidx}] = static_cast<{out_ct}>({expr});")
            lines.append("}")
            return lines

        if not flat:
            self.prelude.add("strided")
        kname = self.kernel(key, f"{label}_kernel" if flat else f"{label}_strided_kernel", build)
        args = [self.ptr(t) for t in ins] + [self.ptr(out, label)] + [lit(v, t) for _, t, v in sc]
        if not flat:
            args += [strided_lit(oshape, bcast_strides(t, oshape)) for t in ins]
            args.append(strided_lit(oshape, out["stride"]))
        args.append(str(n))
        self.launch(kname, cdiv(n, THREADS), THREADS, args)

    def contiguous(self, t, hint="contig"):
        """Return t itself, or a contiguous copy of it (emitting the copy kernel)."""
        if not is_tensor(t) or is_contig(t):
            return t
        self.meta.append({"nbytes": numel(t["shape"]) * self.itemsize(t["dtype"]),
                          "dtype": t["dtype"], "first": "out"})
        sid = len(self.meta) - 1
        dst = {"kind": "tensor", "sid": sid, "shape": t["shape"], "dtype": t["dtype"],
               "stride": contiguous_strides(t["shape"]), "off": 0}
        self.buf(sid, hint)
        self.emit("// kernel below needs contiguous rows: gather the strided input first")
        c = EwCtx()
        self.ew("copy", c.t(t, "src"), c, dst)
        return dst

    def memcpy(self, dst, src):
        nbytes = numel(src["shape"]) * self.itemsize(src["dtype"])
        self.stats["memcpy"] += 1
        self.emit(f"CUDA_CHECK(cudaMemcpyAsync({self.ptr(dst)}, {self.ptr(src)}, {nbytes}, "
                  f"cudaMemcpyDeviceToDevice));")

    @staticmethod
    def itemsize(dt):
        return {"float64": 8, "int64": 8, "float32": 4, "int32": 4, "float16": 2, "bfloat16": 2,
                "int16": 2}.get(dt, 1)

    # -- dispatch ---------------------------------------------------------------------
    def handle(self, ev, count):
        name = ev["name"]
        if name in SILENT_OPS:
            return
        self.stats["ops"] += 1
        if name.startswith("_foreach_"):
            self.op_comment(ev, count)
            self.foreach(ev)
            return
        base = name[:-1] if name.endswith("_") and not name.endswith("__") else name
        outs = ev["outs"] or ([ev["args"]["self"]] if is_tensor(ev["args"].get("self")) else [])
        self.trace.append({
            "line": ev["line"], "op": op_label(ev, overload=True),
            "ins": [self._shape(v) for v in ev["args"].values() if is_tensor(v)],
            "outs": [self._shape(o) for o in ev["outs"]],
            "count": count, "phase": "backward" if ev["node"] else "forward",
            "kind": "view" if base in VIEW_OPS else "op",
        })
        if base in VIEW_OPS:
            self.stats["views"] += 1
            src = ev["args"].get("self")
            o = outs[0] if outs else None
            if is_tensor(src) and o is not None:
                self.emit(f"// view  {op_label(ev)}: {self.buf(src['sid'])} -> [{', '.join(map(str, o['shape']))}]"
                          f" strides [{', '.join(map(str, o['stride']))}]  (no kernel, metadata only)")
            return
        self.op_comment(ev, count)
        self.dispatch(base, ev["args"], outs, ev)

    @staticmethod
    def _shape(t):
        return f"{t['dtype']}[{', '.join(map(str, t['shape']))}]"

    def dispatch(self, base, b, outs, ev):
        if base in ("lift_fresh", "lift_fresh_copy"):
            return self.op_lift_fresh(b, outs)
        if base in ALLOC_OPS:
            self.emit("// allocation only (cudaMalloc above), no kernel")
            return
        if base in MEMSET_ZERO and outs and is_contig(outs[0]):
            o = outs[0]
            self.stats["memcpy"] += 1
            self.emit(f"CUDA_CHECK(cudaMemsetAsync({self.ptr(o, base)}, 0, "
                      f"{numel(o['shape']) * self.itemsize(o['dtype'])}));")
            return
        if base in COPY_OPS and outs:
            src = b.get("src") if base == "copy" else b.get("self")
            o = outs[0]
            if is_tensor(src) and src["dtype"] == o["dtype"] and src["shape"] == o["shape"] \
                    and is_contig(src) and is_contig(o):
                if src["sid"] == o["sid"] and src["off"] == o["off"]:
                    self.emit("// same buffer: no-op")
                else:
                    self.memcpy(o, src)
                return
        if base in EW and outs:
            c = EwCtx()
            try:
                expr = EW[base](c, b)
            except (KeyError, TypeError):
                return self.fallback(base, b)
            label = "cast" if base == "_to_copy" else ("copy" if base in COPY_OPS else base)
            return self.ew(label, expr, c, outs[0])
        h = getattr(self, f"op_{base}", None)
        if h is not None:
            try:
                return h(b, outs)
            except (KeyError, TypeError, IndexError, ValueError):
                pass
        self.fallback(base, b)

    def op__gemm(self, b, outs):
        self.gemm(b["A"], b["B"], outs[0], b.get("bias"), 1, 0, "matmul", M=b["M"], N=b["N"], K=b["K"],
                  sa=b["sa"], sb=b["sb"], sc=b["sc"], batch=b["batch"])

    def fallback(self, base, b):
        self.stats["fallback"] += 1
        args = ", ".join(k for k, v in b.items() if v is not None)
        for fw in ("jax", "tf"):
            if base.startswith(fw + "_"):
                self.emit(f"// no CUDA template for {fw}::{base[len(fw) + 1:]} yet (XLA / TF would emit its own kernel)")
                return
        self.emit(f"// no CUDA template for aten::{base} yet; a real build would call into ATen:")
        self.emit(f"//   at::{base}({args});")

    def foreach(self, ev):
        b = ev["args"]
        base = ev["name"][len("_foreach_"):]
        inplace = base.endswith("_")
        base = base.rstrip("_")
        items = b.get("self") or []
        n = len(items)
        lists = {k for k, v in b.items() if isinstance(v, list) and len(v) == n}
        self.emit(f"// multi-tensor apply over {n} tensors (one launch per tensor here; "
                  f"PyTorch fuses these)")
        for i in range(n):
            bi = {k: (v[i] if k in lists else v) for k, v in b.items()}
            for alias in ("scalar", "scalars"):
                if alias in bi:
                    bi["other"] = bi.pop(alias)
            out = bi["self"] if inplace or i >= len(ev["outs"]) else ev["outs"][i]
            if base in EW:
                c = EwCtx()
                try:
                    expr = EW[base](c, bi)
                except (KeyError, TypeError):
                    self.fallback(base, bi)
                    continue
                self.ew(base, expr, c, out)
            else:
                self.fallback(base, bi)

    # -- data movement --------------------------------------------------------------
    def op_lift_fresh(self, b, outs):
        t = b["self"]
        self.h2d.add(t["sid"])
        nm = self.buf(t["sid"], "lit")
        n = numel(t["shape"])
        ct = ctype(t["dtype"])
        k = self.n_lit
        self.n_lit += 1
        vals = t.get("values")
        if vals is not None:
            body = ", ".join(lit(v, ct) for v in vals) if vals else "0"
            self.emit(f"static const {ct} h_lit{k}[{max(n, 1)}] = {{{body}}};  // torch.tensor(...) literal")
        else:
            self.emit(f"static {ct} h_lit{k}[{n}] = {{}};  // {n} values supplied from Python (elided)")
        self.stats["memcpy"] += 1
        self.emit(f"CUDA_CHECK(cudaMemcpy({nm}, h_lit{k}, sizeof(h_lit{k}), cudaMemcpyHostToDevice));")

    def op__local_scalar_dense(self, b, outs):
        t = b["self"]
        ct = ctype(t["dtype"])
        k = self.n_item
        self.n_item += 1
        self.stats["memcpy"] += 1
        self.emit(f"{ct} h_item{k};  // tensor.item(): synchronous device -> host copy")
        self.emit(f"CUDA_CHECK(cudaMemcpy(&h_item{k}, {self.ptr(t)}, sizeof({ct}), cudaMemcpyDeviceToHost));")

    op_is_nonzero = op__local_scalar_dense

    def op_cat(self, b, outs):
        out = outs[0]
        tensors = [t for t in b["tensors"] if is_tensor(t) and numel(t["shape"]) > 0]
        dim = norm_dim(b.get("dim", 0) or 0, len(out["shape"]))
        pos = 0
        for t in tensors:
            view = dict(out, shape=t["shape"], off=out["off"] + pos * out["stride"][dim])
            c = EwCtx()
            expr = c.t(t, "src")
            self.ew("copy", expr, c, view)
            pos += t["shape"][dim]

    def op_stack(self, b, outs):
        out = outs[0]
        dim = norm_dim(b.get("dim", 0) or 0, len(out["shape"]))
        for k, t in enumerate(b["tensors"]):
            stride = out["stride"][:dim] + out["stride"][dim + 1:]
            view = dict(out, shape=t["shape"], stride=stride, off=out["off"] + k * out["stride"][dim])
            c = EwCtx()
            self.ew("copy", c.t(t, "src"), c, view)

    # -- random -----------------------------------------------------------------------
    def rng_kernel(self, label, out, expr, scalars, src=None):
        self.prelude.add("curand")
        n = numel(out["shape"])
        if n == 0:
            return
        ct = ctype(out["dtype"])
        acc = acctype(out["dtype"])
        key = ("rng", label, expr, out["dtype"], src["dtype"] if src else None)

        def build(name):
            params = ([f"const {ctype(src['dtype'])}* in0"] if src else []) + [f"{ct}* out"]
            params += [f"{acc} {nm}" for nm, _ in scalars]
            params += ["int64_t n", "unsigned long long seed", "unsigned long long offset"]
            return [
                f"// out = {expr}   (Philox4x32-10 counter-based RNG, one subsequence per element)",
                f"__global__ void {name}({', '.join(params)}) {{",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    curandStatePhilox4_32_10_t rng;",
                "    curand_init(seed, i, offset, &rng);",
                *(["    " + f"{acctype(src['dtype'])} a = in0[i];"] if src else []),
                f"    out[i] = static_cast<{ct}>({expr});",
                "}",
            ]

        kname = self.kernel(key, f"{label}_kernel", build)
        args = ([self.ptr(src)] if src else []) + [self.ptr(out, label)]
        args += [lit(v, acc) for _, v in scalars] + [str(n), "kSeed", f"{self.philox}ULL"]
        self.philox += 4
        if not is_contig(out):
            self.emit("// note: output is non-contiguous; generated as if contiguous")
        self.launch(kname, cdiv(n, THREADS), THREADS, args)

    def op_randn(self, b, outs):
        self.rng_kernel("randn", outs[0], "curand_normal(&rng)", [])

    op_randn_like = op_randn

    def op_rand(self, b, outs):
        self.rng_kernel("rand", outs[0], "curand_uniform(&rng)", [])

    op_rand_like = op_rand

    def op_normal(self, b, outs):
        if not is_tensor(b.get("self")):
            return self.fallback("normal", b)
        self.rng_kernel("normal", outs[0], "mean + std * curand_normal(&rng)",
                        [("mean", b.get("mean", 0.0)), ("std", b.get("std", 1.0))])

    def op_uniform(self, b, outs):
        lo, hi = b.get("from", 0.0), b.get("to", 1.0)
        self.rng_kernel("uniform", outs[0], "lo + (hi - lo) * curand_uniform(&rng)", [("lo", lo), ("hi", hi)])

    def op_randint(self, b, outs):
        lo = b.get("low", 0) or 0
        self.rng_kernel("randint", outs[0], "lo + (int64_t)(curand_uniform(&rng) * (hi - lo)) % (int64_t)(hi - lo)",
                        [("lo", lo), ("hi", b["high"])])

    def op_bernoulli(self, b, outs):
        p = b.get("p")
        if not isinstance(p, (int, float)):
            return self.fallback("bernoulli", b)
        self.rng_kernel("bernoulli", outs[0], "curand_uniform(&rng) < p ? 1 : 0", [("p", p)])

    def op_native_dropout(self, b, outs):
        x = b["input"]
        p = b["p"]
        out, mask = outs[0], outs[1]
        if not b.get("train", True) or p == 0:
            return self.memcpy(out, x)
        self.prelude.add("curand")
        n = numel(x["shape"])
        ct, acc = ctype(x["dtype"]), acctype(x["dtype"])

        def build(name):
            return [
                "// inverted dropout: keep with prob (1 - p) and rescale by 1 / (1 - p)",
                f"__global__ void {name}(const {ct}* x, {ct}* out, bool* mask, {acc} p, int64_t n,",
                "                        unsigned long long seed, unsigned long long offset) {",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    curandStatePhilox4_32_10_t rng;",
                "    curand_init(seed, i, offset, &rng);",
                "    bool keep = curand_uniform(&rng) > p;",
                "    mask[i] = keep;",
                f"    out[i] = keep ? static_cast<{ct}>(x[i] / (1 - p)) : static_cast<{ct}>(0);",
                "}",
            ]

        kname = self.kernel(("dropout", x["dtype"]), "dropout_kernel", build)
        self.launch(kname, cdiv(n, THREADS), THREADS,
                    [self.ptr(x), self.ptr(out, "dropout"), self.ptr(mask, "mask"), lit(p, acc), str(n),
                     "kSeed", f"{self.philox}ULL"])
        self.philox += 4

    # -- matmul --------------------------------------------------------------------
    def gemm_kernel(self, dt):
        ct, acc = ctype(dt), acctype(dt)

        def build(name):
            return [
                "// C[b] = alpha * A[b] @ B[b] + beta * bias[b]     (row-major, arbitrary strides)",
                "// 16x16 output tile per block; each K-slice of A and B is staged through shared memory.",
                "#define TILE 16",
                f"__global__ void {name}(const {ct}* A, const {ct}* B, const {ct}* bias, {ct}* C,",
                "                                  int M, int N, int K,",
                "                                  int64_t sAb, int64_t sAm, int64_t sAk,",
                "                                  int64_t sBb, int64_t sBk, int64_t sBn,",
                "                                  int64_t sbb, int64_t sbm, int64_t sbn,",
                "                                  int64_t sCb, int64_t sCm, int64_t sCn,",
                f"                                  {acc} alpha, {acc} beta) {{",
                f"    __shared__ {acc} As[TILE][TILE];",
                f"    __shared__ {acc} Bs[TILE][TILE];",
                "    int b = blockIdx.z;",
                "    int row = blockIdx.y * TILE + threadIdx.y;",
                "    int col = blockIdx.x * TILE + threadIdx.x;",
                "    A += b * sAb;  B += b * sBb;  C += b * sCb;",
                f"    {acc} acc = 0;",
                "    for (int t = 0; t < K; t += TILE) {",
                "        int ka = t + threadIdx.x, kb = t + threadIdx.y;",
                "        As[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? A[row * sAm + ka * sAk] : 0;",
                "        Bs[threadIdx.y][threadIdx.x] = (kb < K && col < N) ? B[kb * sBk + col * sBn] : 0;",
                "        __syncthreads();",
                "        #pragma unroll",
                "        for (int k = 0; k < TILE; ++k)",
                "            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];",
                "        __syncthreads();",
                "    }",
                "    if (row < M && col < N) {",
                f"        {acc} v = alpha * acc;",
                "        if (bias) v += beta * bias[b * sbb + row * sbm + col * sbn];",
                f"        C[row * sCm + col * sCn] = static_cast<{ct}>(v);",
                "    }",
                "}",
            ]

        return self.kernel(("gemm", dt), "gemm_tiled_kernel", build)

    def gemm(self, A, B, C, bias, alpha, beta, label, M=None, N=None, K=None, sa=None, sb=None, sc=None,
             batch=1):
        if sa is None:
            batched = len(A["shape"]) == 3
            batch = A["shape"][0] if batched else 1
            M, K = A["shape"][-2:]
            N = B["shape"][-1]
            sa = A["stride"] if batched else [0] + A["stride"]
            sb = B["stride"] if batched else [0] + B["stride"]
            sc = C["stride"] if batched else [0] + C["stride"]
        if bias is not None and is_tensor(bias):
            cs = ([batch] if len(C["shape"]) == 3 else []) + [M, N]
            sbias = bcast_strides(bias, cs)
            if len(sbias) == 2:
                sbias = [0] + sbias
        else:
            bias, sbias = None, [0, 0, 0]
        kname = self.gemm_kernel(C["dtype"])
        acc = acctype(C["dtype"])
        if C["dtype"] == "float32" and batch == 1 and label in ("mm", "addmm"):
            ta = "CUBLAS_OP_N" if sa[2] == 1 else "CUBLAS_OP_T"
            tb = "CUBLAS_OP_N" if sb[2] == 1 else "CUBLAS_OP_T"
            lda = sa[1] if sa[2] == 1 else sa[2]
            ldb = sb[1] if sb[2] == 1 else sb[2]
            self.emit(f"// cuBLAS (column-major, so compute C^T = B^T A^T): cublasSgemm(handle, {tb}, {ta}, "
                      f"{N}, {M}, {K}, &alpha, B, {ldb}, A, {lda}, &beta, C, {N});")
        args = [self.ptr(A), self.ptr(B), self.ptr(bias) if bias else "nullptr", self.ptr(C, label),
                f"{M}, {N}, {K}",
                ", ".join(map(str, sa)), ", ".join(map(str, sb)), ", ".join(map(str, sbias)),
                ", ".join(map(str, sc)),
                lit(alpha, acc), lit(beta if bias else 0, acc)]
        grid = f"dim3({cdiv(N, 16)}, {cdiv(M, 16)}, {batch})"
        self.launch(kname, grid, "dim3(16, 16)", args)

    def op_mm(self, b, outs):
        self.gemm(b["self"], b["mat2"], outs[0], None, 1, 0, "mm")

    def op_addmm(self, b, outs):
        self.gemm(b["mat1"], b["mat2"], outs[0], b["self"], b.get("alpha", 1), b.get("beta", 1), "addmm")

    def op_bmm(self, b, outs):
        self.gemm(b["self"], b["mat2"], outs[0], None, 1, 0, "bmm")

    def op_baddbmm(self, b, outs):
        self.gemm(b["batch1"], b["batch2"], outs[0], b["self"], b.get("alpha", 1), b.get("beta", 1), "baddbmm")

    def op_mv(self, b, outs):
        A, x, y = b["self"], b["vec"], outs[0]
        M, K = A["shape"]
        self.gemm(A, x, y, None, 1, 0, "mv", M=M, N=1, K=K,
                  sa=[0] + A["stride"], sb=[0, x["stride"][0], 0], sc=[0, y["stride"][0], 0])

    def op_dot(self, b, outs):
        x, y, o = b["self"], b["tensor"], outs[0]
        K = x["shape"][0]
        self.gemm(x, y, o, None, 1, 0, "dot", M=1, N=1, K=K,
                  sa=[0, 0, x["stride"][0]], sb=[0, y["stride"][0], 0], sc=[0, 0, 0])

    # -- reductions ---------------------------------------------------------------
    def reduce(self, kind, x, dims, out):
        nd = len(x["shape"])
        if dims is None or dims == []:
            dims = list(range(nd))
        elif isinstance(dims, int):
            dims = [dims]
        dims = sorted({norm_dim(d, nd) for d in dims}) if nd else []
        keep = [d for d in range(nd) if d not in dims]
        ks, kst = [x["shape"][d] for d in keep], [x["stride"][d] for d in keep]
        rs, rst = [x["shape"][d] for d in dims], [x["stride"][d] for d in dims]
        n_out, n_red = numel(ks), numel(rs)
        init, comb, merge, fin = REDUCE[kind]
        dt = x["dtype"]
        ct, acc = ctype(dt), acctype(dt)
        oct_ = ctype(out["dtype"])
        init = _init_lit(init, dt)
        self.prelude.add("strided")
        contiguous_all = n_out == 1 and is_contig(x)

        if n_out == 1 and n_red >= 2048:
            def build(name):
                v = "in[r]" if contiguous_all else "in[offset_of(r, red)]"
                return [
                    f"// single-block tree reduction ({kind}); production code would use a",
                    "// two-pass multi-block reduction such as cub::DeviceReduce.",
                    f"__global__ void {name}(const {ct}* in, {oct_}* out, Strided red, int64_t n) {{",
                    f"    __shared__ {acc} smem[1024];",
                    f"    {acc} acc = {init};",
                    "    for (int64_t r = threadIdx.x; r < n; r += blockDim.x) {",
                    f"        {acc} v = {v};",
                    f"        acc = {comb.format(x='acc', y='v')};",
                    "    }",
                    "    smem[threadIdx.x] = acc;",
                    "    __syncthreads();",
                    "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {",
                    "        if (threadIdx.x < s)",
                    f"            smem[threadIdx.x] = {merge.format(x='smem[threadIdx.x]', y='smem[threadIdx.x + s]')};",
                    "        __syncthreads();",
                    "    }",
                    f"    if (threadIdx.x == 0) out[0] = static_cast<{oct_}>({fin.format(acc='smem[0]', n='(' + acc + ')n')});",
                    "}",
                ]
            kname = self.kernel(("reduce_all", kind, dt, out["dtype"], contiguous_all),
                                f"{kind}_reduce_all_kernel", build)
            self.launch(kname, 1, 1024, [self.ptr(x), self.ptr(out, kind), strided_lit(rs, rst), str(n_red)])
            return

        def build(name):
            return [
                f"// one thread per output element; loops over the {kind}-reduced dims",
                f"__global__ void {name}(const {ct}* in, {oct_}* out, Strided keep, Strided red,",
                "                                int64_t n_out, int64_t n_red) {",
                "    int64_t o = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (o >= n_out) return;",
                f"    const {ct}* base = in + offset_of(o, keep);",
                f"    {acc} acc = {init};",
                "    for (int64_t r = 0; r < n_red; ++r) {",
                f"        {acc} v = base[offset_of(r, red)];",
                f"        acc = {comb.format(x='acc', y='v')};",
                "    }",
                f"    out[o] = static_cast<{oct_}>({fin.format(acc='acc', n='(' + acc + ')n_red')});",
                "}",
            ]
        kname = self.kernel(("reduce", kind, dt, out["dtype"]), f"{kind}_reduce_kernel", build)
        self.launch(kname, cdiv(n_out, THREADS), THREADS,
                    [self.ptr(x), self.ptr(out, kind), strided_lit(ks, kst), strided_lit(rs, rst),
                     str(n_out), str(n_red)])

    def op_sum(self, b, outs):
        self.reduce("sum", b["self"], b.get("dim"), outs[0])

    def op_mean(self, b, outs):
        self.reduce("mean", b["self"], b.get("dim"), outs[0])

    def op_prod(self, b, outs):
        self.reduce("prod", b["self"], b.get("dim"), outs[0])

    def op_amax(self, b, outs):
        self.reduce("amax", b["self"], b.get("dim"), outs[0])

    def op_amin(self, b, outs):
        self.reduce("amin", b["self"], b.get("dim"), outs[0])

    def op_linalg_vector_norm(self, b, outs):
        if b.get("ord", 2) != 2:
            raise ValueError
        self.reduce("norm2", b["self"], b.get("dim"), outs[0])

    def op_max(self, b, outs):
        if b.get("dim") is None:
            return self.reduce("amax", b["self"], None, outs[0])
        self.arg_reduce(">", b["self"], b["dim"], outs[0], outs[1])

    def op_min(self, b, outs):
        if b.get("dim") is None:
            return self.reduce("amin", b["self"], None, outs[0])
        self.arg_reduce("<", b["self"], b["dim"], outs[0], outs[1])

    def op_argmax(self, b, outs):
        self.arg_reduce(">", b["self"], b.get("dim"), None, outs[0])

    def op_argmin(self, b, outs):
        self.arg_reduce("<", b["self"], b.get("dim"), None, outs[0])

    def arg_reduce(self, cmp, x, dim, vals, idx):
        nd = len(x["shape"])
        if dim is None:
            ks, kst, rs = [], [], [numel(x["shape"])]
            if not is_contig(x):
                raise ValueError
            rst = [1]
        else:
            d = norm_dim(dim, nd)
            ks = [x["shape"][k] for k in range(nd) if k != d]
            kst = [x["stride"][k] for k in range(nd) if k != d]
            rs, rst = [x["shape"][d]], [x["stride"][d]]
        dt = x["dtype"]
        ct, acc = ctype(dt), acctype(dt)
        self.prelude.add("strided")
        label = "argmax" if cmp == ">" else "argmin"

        def build(name):
            return [
                f"// {label} along one dim (+ the value itself when `vals` is non-null)",
                f"__global__ void {name}(const {ct}* in, {ct}* vals, int64_t* idx, Strided keep, Strided red,",
                "                              int64_t n_out, int64_t n_red) {",
                "    int64_t o = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (o >= n_out) return;",
                f"    const {ct}* base = in + offset_of(o, keep);",
                f"    {acc} best = base[0];",
                "    int64_t best_i = 0;",
                "    for (int64_t r = 1; r < n_red; ++r) {",
                f"        {acc} v = base[offset_of(r, red)];",
                f"        if (v {cmp} best) {{ best = v; best_i = r; }}",
                "    }",
                "    if (vals) vals[o] = best;",
                "    idx[o] = best_i;",
                "}",
            ]
        kname = self.kernel(("argreduce", cmp, dt), f"{label}_kernel", build)
        n_out = numel(ks)
        self.launch(kname, cdiv(n_out, THREADS), THREADS,
                    [self.ptr(x), self.ptr(vals, "max") if vals else "nullptr", self.ptr(idx, label),
                     strided_lit(ks, kst), strided_lit(rs, rst), str(n_out), str(rs[0])])

    # -- softmax / normalisation ------------------------------------------------------
    def _rows(self, x, dim):
        nd = len(x["shape"])
        d = norm_dim(dim, nd) if nd else 0
        shape = x["shape"] or [1]
        outer, D, inner = numel(shape[:d]), shape[d], numel(shape[d + 1:])
        return outer, D, inner

    def softmax_like(self, label, x, dim, out, grad=None):
        self.prelude.add("block_reduce")
        outer, D, inner = self._rows(x, dim)
        rows = outer * inner
        dt = x["dtype"]
        ct, acc = ctype(dt), acctype(dt)
        x = self.contiguous(x)
        grad = self.contiguous(grad)
        bodies = {
            "softmax": [
                f"    {acc} m = -INFINITY;",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) m = max(m, (" + acc + ")xr[j * inner]);",
                "    m = block_reduce_max(m);",
                f"    {acc} s = 0;",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) s += exp(xr[j * inner] - m);",
                "    s = block_reduce_sum(s);",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x)",
                "        yr[j * inner] = exp(xr[j * inner] - m) / s;",
            ],
            "log_softmax": [
                f"    {acc} m = -INFINITY;",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) m = max(m, (" + acc + ")xr[j * inner]);",
                "    m = block_reduce_max(m);",
                f"    {acc} s = 0;",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) s += exp(xr[j * inner] - m);",
                "    s = block_reduce_sum(s);",
                f"    {acc} lse = m + log(s);",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x)",
                "        yr[j * inner] = xr[j * inner] - lse;",
            ],
            # x = forward output y, g = incoming gradient
            "softmax_backward": [
                f"    {acc} dot = 0;",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) dot += gr[j * inner] * xr[j * inner];",
                "    dot = block_reduce_sum(dot);",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x)",
                "        yr[j * inner] = xr[j * inner] * (gr[j * inner] - dot);",
            ],
            "log_softmax_backward": [
                f"    {acc} gsum = 0;",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x) gsum += gr[j * inner];",
                "    gsum = block_reduce_sum(gsum);",
                "    for (int64_t j = threadIdx.x; j < D; j += blockDim.x)",
                "        yr[j * inner] = gr[j * inner] - exp(xr[j * inner]) * gsum;",
            ],
        }
        g_param = f"const {ct}* g, " if grad is not None else ""

        def build(name):
            return [
                f"// {label}: one block per row; row r covers elements base + j * inner, j < D",
                f"__global__ void {name}({g_param}const {ct}* x, {ct}* y, int64_t D, int64_t inner) {{",
                "    int64_t r = blockIdx.x;",
                "    int64_t base = (r / inner) * D * inner + (r % inner);",
                f"    const {ct}* xr = x + base;",
                f"    {ct}* yr = y + base;",
                *([f"    const {ct}* gr = g + base;"] if grad is not None else []),
                *bodies[label],
                "}",
            ]
        kname = self.kernel((label, dt), f"{label}_kernel", build)
        threads = next_pow2(D)
        args = ([self.ptr(grad)] if grad is not None else []) + [self.ptr(x), self.ptr(out, label), str(D), str(inner)]
        self.launch(kname, rows, threads, args, comment=f"{rows} rows x {D} cols, {threads} threads per row")

    def op__softmax(self, b, outs):
        self.softmax_like("softmax", b["self"], b["dim"], outs[0])

    def op__log_softmax(self, b, outs):
        self.softmax_like("log_softmax", b["self"], b["dim"], outs[0])

    def op__softmax_backward_data(self, b, outs):
        self.softmax_like("softmax_backward", b["output"], b["dim"], outs[0], grad=b["grad_output"])

    def op__log_softmax_backward_data(self, b, outs):
        self.softmax_like("log_softmax_backward", b["output"], b["dim"], outs[0], grad=b["grad_output"])

    def op_native_layer_norm(self, b, outs):
        self.prelude.add("block_reduce")
        x, w, bias, eps = self.contiguous(b["input"]), b.get("weight"), b.get("bias"), b["eps"]
        N = numel(b["normalized_shape"])
        rows = numel(x["shape"]) // max(N, 1)
        dt = x["dtype"]
        ct, acc = ctype(dt), acctype(dt)

        def build(name):
            return [
                "// LayerNorm over the last N elements of each row; also stores mean and 1/std for backward",
                f"__global__ void {name}(const {ct}* x, const {ct}* w, const {ct}* b, {ct}* y,",
                f"                                  {acc}* mean_out, {acc}* rstd_out, int64_t N, {acc} eps) {{",
                f"    const {ct}* xr = x + blockIdx.x * N;",
                f"    {ct}* yr = y + blockIdx.x * N;",
                f"    {acc} s = 0;",
                "    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) s += xr[j];",
                f"    {acc} mean = block_reduce_sum(s) / N;",
                f"    {acc} v = 0;",
                "    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) v += (xr[j] - mean) * (xr[j] - mean);",
                f"    {acc} rstd = 1 / sqrt(block_reduce_sum(v) / N + eps);",
                "    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) {",
                f"        {acc} h = (xr[j] - mean) * rstd;",
                f"        yr[j] = static_cast<{ct}>((w ? w[j] : 1) * h + (b ? b[j] : 0));",
                "    }",
                "    if (threadIdx.x == 0) { mean_out[blockIdx.x] = mean; rstd_out[blockIdx.x] = rstd; }",
                "}",
            ]
        kname = self.kernel(("layer_norm", dt), "layer_norm_kernel", build)
        threads = next_pow2(N)
        self.launch(kname, rows, threads,
                    [self.ptr(x), self.ptr(w) if is_tensor(w) else "nullptr",
                     self.ptr(bias) if is_tensor(bias) else "nullptr",
                     self.ptr(outs[0], "layer_norm"), self.ptr(outs[1], "ln_mean"), self.ptr(outs[2], "ln_rstd"),
                     str(N), lit(eps, acc)])

    # -- losses ------------------------------------------------------------------------
    def op_nll_loss_forward(self, b, outs):
        self.prelude.add("block_reduce")
        x, tgt, w = self.contiguous(b["self"]), self.contiguous(b["target"]), b.get("weight")
        red, ign = b["reduction"], b["ignore_index"]
        N, C = (x["shape"] if len(x["shape"]) == 2 else [1, x["shape"][0]])
        ct, acc, tt = ctype(x["dtype"]), acctype(x["dtype"]), ctype(tgt["dtype"])

        def build(name):
            return [
                "// negative log-likelihood; reduction: 0 = none, 1 = mean, 2 = sum",
                f"__global__ void {name}(const {ct}* logp, const {tt}* target, const {ct}* weight,",
                f"                                        {ct}* out, {ct}* total_weight, int64_t N, int64_t C,",
                "                                        int64_t ignore_index, int reduction) {",
                f"    {acc} loss = 0, wsum = 0;",
                "    for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {",
                "        int64_t t = target[i];",
                "        if (t == ignore_index) { if (reduction == 0) out[i] = 0; continue; }",
                f"        {acc} w = weight ? ({acc})weight[t] : 1;",
                f"        {acc} l = -w * logp[i * C + t];",
                "        if (reduction == 0) out[i] = l;",
                "        loss += l;",
                "        wsum += w;",
                "    }",
                "    loss = block_reduce_sum(loss);",
                "    wsum = block_reduce_sum(wsum);",
                "    if (threadIdx.x == 0 && reduction != 0) {",
                "        out[0] = reduction == 1 ? loss / wsum : loss;",
                "        *total_weight = wsum;",
                "    }",
                "}",
            ]
        kname = self.kernel(("nll_fwd", x["dtype"], tgt["dtype"]), "nll_loss_forward_kernel", build)
        self.launch(kname, 1, next_pow2(N, 32, 256),
                    [self.ptr(x), self.ptr(tgt), self.ptr(w) if is_tensor(w) else "nullptr",
                     self.ptr(outs[0], "nll_loss"), self.ptr(outs[1], "total_weight"),
                     str(N), str(C), str(ign), str(red)])

    def op_nll_loss_backward(self, b, outs):
        g, x, tgt, w = b["grad_output"], b["self"], b["target"], b.get("weight")
        red, ign, tw = b["reduction"], b["ignore_index"], b["total_weight"]
        gi = outs[0]
        N, C = (x["shape"] if len(x["shape"]) == 2 else [1, x["shape"][0]])
        ct, tt = ctype(x["dtype"]), ctype(tgt["dtype"])

        def build(name):
            return [
                "// d(loss)/d(logp): zero everywhere except -w * grad at each sample's target class",
                f"__global__ void {name}(const {ct}* grad_out, const {tt}* target, const {ct}* weight,",
                f"                                         const {ct}* total_weight, {ct}* grad_in,",
                "                                         int64_t N, int64_t C, int64_t ignore_index, int reduction) {",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= N) return;",
                "    int64_t t = target[i];",
                "    if (t == ignore_index) return;",
                f"    {ct} g = reduction == 0 ? grad_out[i] : grad_out[0];",
                "    if (reduction == 1) g /= *total_weight;",
                "    grad_in[i * C + t] = -(weight ? weight[t] : 1) * g;",
                "}",
            ]
        kname = self.kernel(("nll_bwd", x["dtype"], tgt["dtype"]), "nll_loss_backward_kernel", build)
        self.emit(f"CUDA_CHECK(cudaMemsetAsync({self.ptr(gi, 'nll_grad')}, 0, "
                  f"{numel(gi['shape']) * self.itemsize(gi['dtype'])}));")
        self.launch(kname, cdiv(N, THREADS), THREADS,
                    [self.ptr(g), self.ptr(tgt), self.ptr(w) if is_tensor(w) else "nullptr", self.ptr(tw),
                     self.ptr(gi), str(N), str(C), str(ign), str(red)])

    def op_mse_loss(self, b, outs):
        x, y, red = b["self"], b["target"], b.get("reduction", 1)
        if red != 0:
            x, y = self.contiguous(x), self.contiguous(y)
        if red == 0:
            c = EwCtx()
            e = f"({c.t(x)} - {c.t(y)}) * ({EwCtx.SYMS[0]} - {EwCtx.SYMS[1]})"
            return self.ew("squared_error", e, c, outs[0])
        self.prelude.add("block_reduce")
        n = numel(x["shape"])
        ct, acc = ctype(x["dtype"]), acctype(x["dtype"])

        def build(name):
            return [
                "// sum((x - y)^2), optionally divided by n (single block)",
                f"__global__ void {name}(const {ct}* x, const {ct}* y, {ct}* out, int64_t n, int mean) {{",
                f"    {acc} s = 0;",
                "    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {",
                f"        {acc} d = x[i] - y[i];",
                "        s += d * d;",
                "    }",
                "    s = block_reduce_sum(s);",
                "    if (threadIdx.x == 0) out[0] = mean ? s / n : s;",
                "}",
            ]
        kname = self.kernel(("mse", x["dtype"]), "mse_loss_kernel", build)
        self.launch(kname, 1, 1024, [self.ptr(x), self.ptr(y), self.ptr(outs[0], "mse"), str(n),
                                     "1" if red == 1 else "0"])

    # -- conv / pooling / embedding -------------------------------------------------
    def op_convolution(self, b, outs):
        x, w, bias, y = self.contiguous(b["input"]), self.contiguous(b["weight"]), b.get("bias"), outs[0]
        if b.get("transposed") or len(x["shape"]) not in (3, 4):
            raise ValueError
        conv1d = len(x["shape"]) == 3
        xs = x["shape"] if not conv1d else [x["shape"][0], x["shape"][1], 1, x["shape"][2]]
        ws = w["shape"] if not conv1d else [w["shape"][0], w["shape"][1], 1, w["shape"][2]]
        ys = y["shape"] if not conv1d else [y["shape"][0], y["shape"][1], 1, y["shape"][2]]
        nsp = 1 if conv1d else 2
        st = as_list(b["stride"], nsp)
        pd = as_list(b["padding"], nsp)
        dl = as_list(b["dilation"], nsp)
        if conv1d:
            st, pd, dl = [1] + st, [0] + pd, [1] + dl
        groups = b.get("groups", 1)
        dt = x["dtype"]
        ct, acc = ctype(dt), acctype(dt)

        def build(name):
            return [
                "// direct convolution, NCHW, one thread per output element.",
                "// (cuDNN equivalent: cudnnConvolutionForward with an implicit-GEMM / Winograd algo)",
                f"__global__ void {name}(const {ct}* x, const {ct}* w, const {ct}* bias, {ct}* y,",
                "                              int N, int C, int H, int W, int K, int R, int S, int P, int Q,",
                "                              int sh, int sw, int ph, int pw, int dh, int dw, int groups) {",
                "    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (idx >= (int64_t)N * K * P * Q) return;",
                "    int q = idx % Q, p = (idx / Q) % P, k = (idx / ((int64_t)P * Q)) % K;",
                "    int n = idx / ((int64_t)K * P * Q);",
                "    int cpg = C / groups, g = k / (K / groups);",
                f"    {acc} acc = bias ? ({acc})bias[k] : 0;",
                "    for (int c = 0; c < cpg; ++c)",
                "        for (int r = 0; r < R; ++r) {",
                "            int h = p * sh - ph + r * dh;",
                "            if (h < 0 || h >= H) continue;",
                "            for (int s = 0; s < S; ++s) {",
                "                int ww = q * sw - pw + s * dw;",
                "                if (ww < 0 || ww >= W) continue;",
                "                acc += x[(((int64_t)n * C + g * cpg + c) * H + h) * W + ww] *",
                "                       w[(((int64_t)k * cpg + c) * R + r) * S + s];",
                "            }",
                "        }",
                f"    y[idx] = static_cast<{ct}>(acc);",
                "}",
            ]
        kname = self.kernel(("conv2d", dt), "conv2d_kernel", build)
        total = ys[0] * ys[1] * ys[2] * ys[3]
        self.launch(kname, cdiv(total, THREADS), THREADS,
                    [self.ptr(x), self.ptr(w), self.ptr(bias) if is_tensor(bias) else "nullptr",
                     self.ptr(y, "conv"),
                     ", ".join(map(str, [xs[0], xs[1], xs[2], xs[3]])),
                     ", ".join(map(str, [ws[0], ws[2], ws[3], ys[2], ys[3]])),
                     ", ".join(map(str, [st[0], st[1], pd[0], pd[1], dl[0], dl[1]])), str(groups)])

    def _pool_geom(self, x, y):
        if len(x["shape"]) == 4:
            NC = x["shape"][0] * x["shape"][1]
        else:
            NC = x["shape"][0]
        return NC, x["shape"][-2], x["shape"][-1], y["shape"][-2], y["shape"][-1]

    def op_max_pool2d_with_indices(self, b, outs):
        x, y, idx = self.contiguous(b["self"]), outs[0], outs[1]
        ks = as_list(b["kernel_size"], 2)
        st = as_list(b.get("stride"), 2) or ks
        pd = as_list(b.get("padding"), 2) or [0, 0]
        dl = as_list(b.get("dilation"), 2) or [1, 1]
        NC, H, W, P, Q = self._pool_geom(x, y)
        ct = ctype(x["dtype"])

        def build(name):
            return [
                "// 2-D max pooling; also records the argmax position for the backward pass",
                f"__global__ void {name}(const {ct}* x, {ct}* y, int64_t* idx, int NC, int H, int W, int P, int Q,",
                "                                 int kh, int kw, int sh, int sw, int ph, int pw, int dh, int dw) {",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= (int64_t)NC * P * Q) return;",
                "    int q = i % Q, p = (i / Q) % P, nc = i / ((int64_t)P * Q);",
                f"    const {ct}* xp = x + (int64_t)nc * H * W;",
                f"    {ct} best = -INFINITY;",
                "    int64_t best_i = -1;",
                "    for (int r = 0; r < kh; ++r)",
                "        for (int s = 0; s < kw; ++s) {",
                "            int h = p * sh - ph + r * dh, w = q * sw - pw + s * dw;",
                "            if (h < 0 || h >= H || w < 0 || w >= W) continue;",
                f"            {ct} v = xp[h * W + w];",
                "            if (v > best || isnan(v)) { best = v; best_i = h * W + w; }",
                "        }",
                "    y[i] = best;",
                "    idx[i] = best_i;",
                "}",
            ]
        kname = self.kernel(("maxpool", x["dtype"]), "max_pool2d_kernel", build)
        self.launch(kname, cdiv(NC * P * Q, THREADS), THREADS,
                    [self.ptr(x), self.ptr(y, "maxpool"), self.ptr(idx, "pool_idx"),
                     ", ".join(map(str, [NC, H, W, P, Q])),
                     ", ".join(map(str, ks + st + pd + dl))])

    def op_avg_pool2d(self, b, outs):
        x, y = self.contiguous(b["self"]), outs[0]
        ks = as_list(b["kernel_size"], 2)
        st = as_list(b.get("stride"), 2) or ks
        pd = as_list(b.get("padding"), 2) or [0, 0]
        NC, H, W, P, Q = self._pool_geom(x, y)
        cip = b.get("count_include_pad", True)
        ct, acc = ctype(x["dtype"]), acctype(x["dtype"])

        def build(name):
            return [
                "// 2-D average pooling",
                f"__global__ void {name}(const {ct}* x, {ct}* y, int NC, int H, int W, int P, int Q,",
                "                                 int kh, int kw, int sh, int sw, int ph, int pw, bool count_pad) {",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= (int64_t)NC * P * Q) return;",
                "    int q = i % Q, p = (i / Q) % P, nc = i / ((int64_t)P * Q);",
                f"    {acc} s = 0;",
                "    int cnt = 0;",
                "    for (int r = 0; r < kh; ++r)",
                "        for (int c = 0; c < kw; ++c) {",
                "            int h = p * sh - ph + r, w = q * sw - pw + c;",
                "            if (h < 0 || h >= H || w < 0 || w >= W) continue;",
                "            s += x[((int64_t)nc * H + h) * W + w];",
                "            ++cnt;",
                "        }",
                "    y[i] = s / (count_pad ? kh * kw : cnt);",
                "}",
            ]
        kname = self.kernel(("avgpool", x["dtype"]), "avg_pool2d_kernel", build)
        self.launch(kname, cdiv(NC * P * Q, THREADS), THREADS,
                    [self.ptr(x), self.ptr(y, "avgpool"), ", ".join(map(str, [NC, H, W, P, Q])),
                     ", ".join(map(str, ks + st + pd)), "true" if cip else "false"])

    def op__adaptive_avg_pool2d(self, b, outs):
        x, y = self.contiguous(b["self"]), outs[0]
        NC, H, W, P, Q = self._pool_geom(x, y)
        ct, acc = ctype(x["dtype"]), acctype(x["dtype"])

        def build(name):
            return [
                "// adaptive average pooling: output cell (p, q) averages rows [h0, h1) x cols [w0, w1)",
                f"__global__ void {name}(const {ct}* x, {ct}* y, int NC, int H, int W, int P, int Q) {{",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= (int64_t)NC * P * Q) return;",
                "    int q = i % Q, p = (i / Q) % P, nc = i / ((int64_t)P * Q);",
                "    int h0 = (p * H) / P, h1 = ((p + 1) * H + P - 1) / P;",
                "    int w0 = (q * W) / Q, w1 = ((q + 1) * W + Q - 1) / Q;",
                f"    {acc} s = 0;",
                "    for (int h = h0; h < h1; ++h)",
                "        for (int w = w0; w < w1; ++w) s += x[((int64_t)nc * H + h) * W + w];",
                "    y[i] = s / ((h1 - h0) * (w1 - w0));",
                "}",
            ]
        kname = self.kernel(("adaptive_avgpool", x["dtype"]), "adaptive_avg_pool2d_kernel", build)
        self.launch(kname, cdiv(NC * P * Q, THREADS), THREADS,
                    [self.ptr(x), self.ptr(y, "adaptive_pool"), ", ".join(map(str, [NC, H, W, P, Q]))])

    def op_embedding(self, b, outs):
        w, ix, y = self.contiguous(b["weight"]), self.contiguous(b["indices"]), outs[0]
        D = w["shape"][1]
        n = numel(y["shape"])
        ct, it = ctype(w["dtype"]), ctype(ix["dtype"])

        def build(name):
            return [
                "// embedding lookup: row gather out[i, :] = weight[indices[i], :]",
                f"__global__ void {name}(const {ct}* weight, const {it}* indices, {ct}* out, int64_t D, int64_t n) {{",
                "    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;",
                "    if (i >= n) return;",
                "    out[i] = weight[indices[i / D] * D + i % D];",
                "}",
            ]
        kname = self.kernel(("embedding", w["dtype"], ix["dtype"]), "embedding_kernel", build)
        self.launch(kname, cdiv(n, THREADS), THREADS,
                    [self.ptr(w), self.ptr(ix), self.ptr(y, "embedding"), str(D), str(n)])

    # -- assembly -----------------------------------------------------------------------
    def run(self):
        ops = [e for e in self.events]
        segments, alias = segment(ops, self.pynames, self.meta)
        if alias:
            for seg, _ in segments:
                for ev in seg:
                    apply_alias(ev["args"], alias)
                    apply_alias(ev["outs"], alias)
        emitted = 0
        cur_header = None
        clipped = False
        for seg, count in segments:
            for ev in seg:
                if emitted >= MAX_EMITTED_OPS:
                    clipped = True
                    break
                self.cur_line = ev["line"]
                mod = ev["module"]
                header = (ev["line"], mod[0] if mod else None, ev["node"])
                if header != cur_header and ev["name"] not in SILENT_OPS:
                    cur_header = header
                    self.emit("")
                    src = self.src_lines[ev["line"] - 1].strip() if 0 < ev["line"] <= len(self.src_lines) else ""
                    if len(src) > 60:
                        src = src[:57] + "..."
                    h = f"// ---- L{ev['line']}: {src}"
                    if mod:
                        h += f"   [{self.modules.get(mod[0], '?')} : {mod[1]}]"
                    if ev["node"]:
                        h += f"   <backward: {ev['node']}>"
                    if count > 1:
                        h += f"   (repeated x{count})"
                    self.emit(h)
                self.handle(ev, count)
                emitted += 1
            if clipped:
                break
        if clipped:
            self.cur_line = 0
            self.emit("")
            self.emit(f"// ... output clipped after {MAX_EMITTED_OPS} ops")
        return clipped


def segment(ops, pynames, meta):
    """Split the trace into per-line segments and collapse repeated ones (loops).

    A new segment starts when the source line changes or when the same call
    site dispatches the same op again (the next loop iteration). Two segments
    are equal when they run the same ops on the same named tensors; temporaries
    are compared by where in the segment they were produced.
    """
    segs = []
    cur = None
    for ev in ops:
        site = (ev["line"], ev["lasti"], ev["name"])
        if cur is None or ev["line"] != cur["line"] or site in cur["sites"]:
            cur = {"line": ev["line"], "sites": set(), "ops": []}
            segs.append(cur)
        cur["sites"].add(site)
        cur["ops"].append(ev)

    def token(v, local):
        if is_tensor(v):
            sid = v["sid"]
            if sid in local:
                return ("L", local[sid], tuple(v["shape"]))
            if sid in pynames or (sid is not None and meta[sid]["first"] == "in"):
                return ("N", sid, tuple(v["shape"]), v["off"])
            return ("X", tuple(v["shape"]), v["dtype"])
        if isinstance(v, list):
            return tuple(token(x, local) for x in v)
        return None

    order, counts, alias = [], {}, {}
    for s in segs:
        local, sig = {}, []
        for k, ev in enumerate(s["ops"]):
            ins = tuple(token(v, local) for v in ev["args"].values())
            outs = tuple(("N", o["sid"]) if o["sid"] in pynames else ("T", tuple(o["shape"])) for o in ev["outs"])
            sig.append((ev["name"], ev["overload"], ev["node"], ev["module"], ins, outs))
            for o in ev["outs"]:
                local.setdefault(o["sid"], k)
        key = (s["line"], tuple(sig))
        if key in counts:
            counts[key][1] += 1
            # A dropped repeat reuses the buffers of the first occurrence, the
            # way a loop body reuses its allocations every iteration.
            for ev, first in zip(s["ops"], counts[key][0]):
                for o, fo in zip(ev["outs"], first["outs"]):
                    if o["sid"] is not None and fo["sid"] is not None and o["sid"] != fo["sid"]:
                        alias[o["sid"]] = fo["sid"]
        else:
            counts[key] = [s["ops"], 1]
            order.append(key)
    return [(counts[k][0], counts[k][1]) for k in order], alias


def apply_alias(v, alias):
    if is_tensor(v):
        sid = v["sid"]
        seen = 0
        while sid in alias and seen < 64:
            sid, seen = alias[sid], seen + 1
        if sid != v["sid"]:
            v["sid"] = sid
    elif isinstance(v, list):
        for x in v:
            apply_alias(x, alias)
    elif isinstance(v, dict) and "kind" not in v:
        for x in v.values():
            apply_alias(x, alias)


def op_label(ev, overload=False):
    fw = ev.get("fw", "aten")
    name = ev.get("orig") or ev["name"]
    label = f"{fw}::{name}"
    if overload and ev.get("overload") not in (None, "", "default"):
        label += "." + ev["overload"]
    return label


FW_NAMES = {"aten": "PyTorch", "jax": "JAX", "tf": "TensorFlow"}


def generate(events, storage_meta, names, modules, source, cell, seed=0, truncated=False, arch="sm_80",
             files=(), filename=""):
    files_out = []
    stats = {"ops": 0, "kernels": 0, "launches": 0, "views": 0, "fallback": 0, "memcpy": 0}
    ops = []
    if events:
        main = generate_main(events, storage_meta, names, modules, source, cell, seed, truncated, arch)
        files_out.append(main["file"])
        stats.update(main["stats"])
        ops = main["ops"]
    for f in files:
        if f["lang"] == "ptx":
            import frameworks
            lm = frameworks.ptx_line_map(f["code"], filename)
            n_head = f["code"][:f["code"].find("\n\n")].count("\n") + 1  # comment header -> launch line
            lm = [x or (f["launch_line"] if i < n_head else 0) for i, x in enumerate(lm)]
        else:
            lm = [f["launch_line"]] * (f["code"].count("\n") + 1)
        files_out.append({"name": f["name"], "lang": f["lang"], "code": f["code"], "line_map": lm})
        stats["kernels"] += 1
    stats["files"] = len(files_out)
    return {"files": files_out, "ops": ops, "stats": stats}


def generate_main(events, storage_meta, names, modules, source, cell, seed=0, truncated=False, arch="sm_80"):
    g = Gen(events, storage_meta, names, modules, source, cell, seed)
    clipped = g.run()

    out = []  # (text, py_line)

    def add(text="", line=0):
        for ln in text.split("\n"):
            out.append((ln, line))

    used = list(g.bufnames.items())
    seen_names = set()
    inputs, temps = [], []
    for sid, nm in used:
        if nm in seen_names:
            continue
        seen_names.add(nm)
        m = storage_meta[sid]
        if m["first"] == "in" and sid not in g.h2d:
            inputs.append((sid, nm))
        else:
            temps.append((sid, nm))

    st = g.stats
    add("// " + "=" * 74)
    add(f"//  cell [{cell}]  ->  CUDA C++   (auto-generated from the PyTorch ATen trace)")
    fws = sorted({ev.get("fw", "aten") for ev in events}, key=list(FW_NAMES).index)
    add(f"//  traced from: {', '.join(FW_NAMES[f] for f in fws)}")
    if "jax" in fws or "tf" in fws:
        add("//  (JAX / TF run op by op while tracing; under jit, XLA would fuse many of these kernels)")
    add(f"//  {st['ops']} ops  |  {st['kernels']} kernels  |  {st['launches']} launches  |  "
        f"{st['views']} views  |  {st['memcpy']} memcpy/memset  |  {st['fallback']} untranslated")
    add(f"//  build:  nvcc -O3 -arch={arch} -o cell{cell} cell{cell}.cu")
    if truncated:
        add("//  note: trace truncated (op limit reached)")
    add("// " + "=" * 74)
    add("#include <cuda_runtime.h>")
    if "curand" in g.prelude:
        add("#include <curand_kernel.h>")
    if any(storage_meta[s]["dtype"] in ("float16", "bfloat16") for s, _ in used):
        add("#include <cuda_fp16.h>")
        add("#include <cuda_bf16.h>")
    add("#include <cstdint>")
    add("#include <cstdio>")
    add("#include <cmath>")
    add("")
    add("#define CUDA_CHECK(call)                                                        \\")
    add("    do {                                                                        \\")
    add("        cudaError_t err_ = (call);                                              \\")
    add("        if (err_ != cudaSuccess) {                                              \\")
    add("            fprintf(stderr, \"CUDA error %s at %s:%d\\n\", cudaGetErrorString(err_), \\")
    add("                    __FILE__, __LINE__);                                        \\")
    add("            exit(1);                                                            \\")
    add("        }                                                                       \\")
    add("    } while (0)")
    if "curand" in g.prelude:
        add("")
        add(f"static const unsigned long long kSeed = {seed}ULL;  // torch.initial_seed()")
    if "strided" in g.prelude:
        add("")
        add(PRELUDE_STRIDED)
    if "block_reduce" in g.prelude:
        add("")
        add(PRELUDE_BLOCK_REDUCE)

    if g.kernels:
        add("")
        add("// " + "-" * 30 + " kernels " + "-" * 35)
        for name, (lines, py) in g.kernels.items():
            add("")
            for ln in lines:
                add(ln, py)

    add("")
    add("// " + "-" * 30 + " host " + "-" * 38)
    params = [f"{ctype(storage_meta[s]['dtype'])}* {nm}" for s, nm in inputs]
    add(f"// inputs are device buffers produced by earlier cells")
    add(f"void run_cell_{cell}({', '.join(params)}) {{")
    if temps:
        add("    // ---- device buffers ----")
        width = max(len(f"{ctype(storage_meta[s]['dtype'])}* {nm} = nullptr;") for s, nm in temps)
        for s, nm in temps:
            m = storage_meta[s]
            decl = f"{ctype(m['dtype'])}* {nm} = nullptr;".ljust(width)
            add(f"    {decl}  CUDA_CHECK(cudaMalloc(&{nm}, {max(m['nbytes'], 1)}));  // {human_bytes(m['nbytes'])}")
    for text, py in g.body:
        add(text, py)
    add("")
    add("    CUDA_CHECK(cudaGetLastError());")
    add("    CUDA_CHECK(cudaDeviceSynchronize());")
    named = [nm for s, nm in temps if s in names]
    unnamed = [nm for s, nm in temps if s not in names]
    if named:
        add(f"    // still referenced from Python, kept alive: {', '.join(named)}")
    for nm in unnamed:
        add(f"    CUDA_CHECK(cudaFree({nm}));")
    add("}")
    add("")
    add("int main() {")
    for s, nm in inputs:
        m = storage_meta[s]
        add(f"    {ctype(m['dtype'])}* {nm} = nullptr;  CUDA_CHECK(cudaMalloc(&{nm}, {max(m['nbytes'], 1)}));")
    add(f"    run_cell_{cell}({', '.join(nm for _, nm in inputs)});")
    for _, nm in inputs:
        add(f"    cudaFree({nm});")
    add("    return 0;")
    add("}")

    return {
        "file": {"name": f"cell{cell}.cu", "lang": "cpp", "code": "\n".join(t for t, _ in out),
                 "line_map": [ln for _, ln in out]},
        "ops": g.trace,
        "stats": dict(st, clipped=clipped, truncated=truncated),
    }
