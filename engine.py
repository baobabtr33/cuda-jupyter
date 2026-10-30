"""Notebook execution engine.

Runs cells in a persistent namespace (like an IPython kernel) while a
TorchDispatchMode records every ATen op the cell dispatches; frameworks.py
adds JAX / TensorFlow ops and Numba / Triton / CuPy kernels to the same
recording. The trace is handed to cudagen, which turns it into the CUDA code
shown next to the cell.
"""
import ast
import contextlib
import importlib
import io
import linecache
import os
import sys
import threading
import time
import traceback

# Let PyTorch, JAX and TensorFlow share one GPU instead of each grabbing it all.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

import cudagen
import frameworks

MAX_EVENTS = 20000
KEEPALIVE_BYTES = 512 * 1024 * 1024
_C_KEYWORDS = {
    "auto", "bool", "break", "case", "char", "class", "const", "continue", "default", "delete",
    "do", "double", "else", "enum", "extern", "float", "for", "goto", "if", "int", "long", "main",
    "namespace", "new", "operator", "private", "public", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "template", "this", "typedef", "union", "unsigned",
    "void", "volatile", "while", "dim3", "half", "Strided", "offset_of",
}


def _dtype_name(dt):
    return str(dt).replace("torch.", "")


def _cuda_arch():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return f"sm_{major}{minor}"
    return "sm_80"


def c_ident(name):
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    if not out or out[0].isdigit() or out in _C_KEYWORDS:
        out = "v_" + out
    return out


class Recorder(TorchDispatchMode):
    """Records ATen ops (with schema-bound arguments) dispatched while active."""

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.events = []
        self.paused = 0
        self.truncated = False
        self.storage_ids = {}
        self.storage_meta = []
        self.module_stack = []
        self._keep = []
        self._kept_bytes = 0
        self._last_site = (0, -1)
        self.files = []          # compiled kernels captured from Numba / Triton / CuPy
        self.file_keys = set()
        self.tf_res_shape = {}

    # -- other frameworks ----------------------------------------------------------
    def ext_lookup(self, key):
        return self.storage_ids.get(key)

    def ext_storage(self, key, nbytes, dtype, role, obj=None):
        sid = self.storage_ids.get(key)
        if sid is None:
            sid = len(self.storage_meta)
            self.storage_ids[key] = sid
            self.storage_meta.append({"nbytes": nbytes, "dtype": dtype, "first": role})
            if obj is not None and self._kept_bytes < KEEPALIVE_BYTES:
                self._keep.append(obj)
                self._kept_bytes += nbytes
        return sid

    def grow_storage(self, sid, nbytes):
        self.storage_meta[sid]["nbytes"] = max(self.storage_meta[sid]["nbytes"], nbytes)

    def alias_storage(self, key, sid, obj=None):
        self.storage_ids[key] = sid
        if obj is not None:
            self._keep.append(obj)

    def add_event(self, ev):
        if len(self.events) >= MAX_EVENTS:
            self.truncated = True
            return
        line, lasti = self._site()
        ev.setdefault("line", line)
        ev.setdefault("lasti", lasti)
        ev.setdefault("module", None)
        ev.setdefault("node", None)
        self.events.append(ev)

    def add_file(self, name, lang, code, launch_line=0, launch=None):
        self.files.append({"name": name, "lang": lang, "code": code, "launch_line": launch_line,
                           "launch": launch})

    # -- tensors -------------------------------------------------------------
    def _storage_key(self, t):
        try:
            st = t.untyped_storage()
        except Exception:
            return None, None
        ptr = st.data_ptr()
        return (ptr if ptr else ("empty", id(t))), st

    def lookup(self, t):
        key, _ = self._storage_key(t)
        return self.storage_ids.get(key)

    def tensor(self, t, role):
        key, st = self._storage_key(t)
        desc = {
            "kind": "tensor",
            "shape": list(t.shape),
            "stride": list(t.stride()) if t.layout == torch.strided else [],
            "off": t.storage_offset() if t.layout == torch.strided else 0,
            "dtype": _dtype_name(t.dtype),
            "sid": None,
        }
        if key is None:
            return desc
        sid = self.storage_ids.get(key)
        if sid is None:
            sid = len(self.storage_meta)
            self.storage_ids[key] = sid
            self.storage_meta.append({"nbytes": st.nbytes(), "dtype": desc["dtype"], "first": role})
            # Keep storages alive so their address cannot be recycled for a
            # different tensor mid-cell (which would merge two buffers).
            if self._kept_bytes < KEEPALIVE_BYTES:
                self._keep.append(st)
                self._kept_bytes += st.nbytes()
        desc["sid"] = sid
        return desc

    def serialize(self, v, role="in"):
        if isinstance(v, torch.Tensor):
            return self.tensor(v, role)
        if isinstance(v, (list, tuple)):
            return [self.serialize(x, role) for x in v]
        if isinstance(v, (bool, int, float)) or v is None:
            return v
        if isinstance(v, torch.dtype):
            return {"kind": "dtype", "v": _dtype_name(v)}
        if isinstance(v, (torch.SymInt, torch.SymFloat)):
            return int(v) if isinstance(v, torch.SymInt) else float(v)
        if isinstance(v, str):
            return v
        return {"kind": "other", "v": str(v)}

    # -- source attribution ----------------------------------------------------
    def _site(self):
        f = sys._getframe(2)
        while f is not None:
            if f.f_code.co_filename == self.filename:
                self._last_site = (f.f_lineno, f.f_lasti)
                return self._last_site
            f = f.f_back
        # Autograd may run backward nodes off the Python stack.
        return self._last_site[0], -1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.paused or func.namespace != "aten":
            return func(*args, **kwargs)
        if len(self.events) >= MAX_EVENTS:
            self.truncated = True
            return func(*args, **kwargs)

        bound = {}
        for i, a in enumerate(func._schema.arguments):
            if i < len(args):
                v = args[i]
            elif a.name in kwargs:
                v = kwargs[a.name]
            else:
                v = a.default_value
            bound[a.name] = self.serialize(v)

        out = func(*args, **kwargs)

        name = func._schema.name.split("::", 1)[1]
        if name in ("lift_fresh", "lift_fresh_copy") and isinstance(args[0], torch.Tensor):
            t = args[0]
            if t.numel() <= 64:
                bound["self"]["values"] = t.detach().reshape(-1).tolist()
        outs = []
        for o in (out if isinstance(out, (list, tuple)) else [out]):
            if isinstance(o, torch.Tensor):
                outs.append(self.tensor(o, "out"))
            elif isinstance(o, (list, tuple)):
                outs.extend(self.tensor(x, "out") for x in o if isinstance(x, torch.Tensor))
        node = torch._C._current_autograd_node()
        line, lasti = self._site()
        self.events.append({
            "name": name,
            "overload": func._overloadname,
            "args": bound,
            "outs": outs,
            "line": line,
            "lasti": lasti,
            "module": self.module_stack[-1] if self.module_stack else None,
            "node": node.name().replace("torch::autograd::", "") if node is not None else None,
        })
        return out


@contextlib.contextmanager
def _module_hooks(rec):
    def pre(mod, inputs):
        rec.module_stack.append((id(mod), type(mod).__name__))

    def post(mod, inputs, output):
        if rec.module_stack:
            rec.module_stack.pop()

    h1 = nn.modules.module.register_module_forward_pre_hook(pre)
    h2 = nn.modules.module.register_module_forward_hook(post, always_call=True)
    try:
        yield
    finally:
        h1.remove()
        h2.remove()


@contextlib.contextmanager
def _quiet_repr(rec):
    """Tensor printing runs dozens of ATen ops; keep them out of the trace."""
    import torch._tensor_str as ts

    orig = ts._str

    def wrapped(*a, **k):
        rec.paused += 1
        try:
            return orig(*a, **k)
        finally:
            rec.paused -= 1

    ts._str = wrapped
    try:
        yield
    finally:
        ts._str = orig


class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.ns = {"__name__": "__main__", "__builtins__": __builtins__}
        self.count = 0

    def _python_names(self, rec):
        names, modules = {}, {}

        def name_sid(sid, name):
            if sid is not None and sid not in names:
                names[sid] = c_ident(name)

        def visit(name, t, depth=0):
            if isinstance(t, torch.Tensor):
                name_sid(rec.lookup(t), name)
                if t.is_leaf and t.grad is not None:
                    visit(name + "_grad", t.grad)
                return
            mod = type(t).__module__ or ""
            if mod.startswith("jax"):
                if hasattr(t, "unsafe_buffer_pointer"):
                    name_sid(rec.ext_lookup(frameworks._jax_key(t)), name)
                return
            if mod.startswith("tensorflow") or mod.startswith("keras"):
                handle = getattr(t, "handle", None)
                if handle is not None:
                    name_sid(rec.ext_lookup(("tf-res", id(handle))), name)
                else:
                    name_sid(rec.ext_lookup(("tf", id(t))), name)
                return
            if depth < 3 and isinstance(t, dict) and len(t) <= 256:
                for k, v in t.items():
                    visit(f"{name}_{k}", v, depth + 1)
            elif depth < 3 and isinstance(t, (list, tuple)) and len(t) <= 64:
                for i, v in enumerate(t):
                    visit(f"{name}{i}", v, depth + 1)

        for var, val in list(self.ns.items()):
            if var.startswith("_"):
                continue
            try:
                if isinstance(val, torch.Tensor):
                    visit(var, val)
                elif isinstance(val, nn.Module):
                    for mn, m in val.named_modules(prefix=var):
                        modules.setdefault(id(m), mn)
                    for pn, p in val.named_parameters(prefix=var):
                        visit(pn, p)
                    for bn, b in val.named_buffers(prefix=var):
                        visit(bn, b)
                elif isinstance(val, torch.optim.Optimizer):
                    for k, (_, state) in enumerate(val.state.items()):
                        for sk, sv in state.items():
                            visit(f"{var}_s{k}_{sk}", sv)
                elif hasattr(val, "trainable_variables") and not isinstance(val, type):
                    for v in val.trainable_variables:  # Keras models / layers
                        visit(f"{var}_{getattr(v, 'path', v.name)}", v)
                else:
                    visit(var, val)
            except Exception:
                continue
        return names, modules

    def run(self, source):
        with self.lock:
            self.count += 1
            n = self.count
            filename = f"<cell-{n}>"
            linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
            stdout, stderr = io.StringIO(), io.StringIO()
            rec = Recorder(filename)
            result, error = None, None
            t_start = time.perf_counter()
            try:
                tree = ast.parse(source, filename)
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                code = compile(tree, filename, "exec")
                self._preimport(tree)
                frameworks.install_hooks()
                frameworks.ACTIVE = rec
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                        _module_hooks(rec), _quiet_repr(rec), frameworks.eager_mode(), rec:
                    exec(code, self.ns)
                    if last is not None:
                        val = eval(compile(last, filename, "eval"), self.ns)
                        if val is not None:
                            self.ns["_"] = val
                            rec.paused += 1
                            try:
                                result = repr(val)
                            finally:
                                rec.paused -= 1
            except BaseException as e:  # noqa: BLE001 - report everything like Jupyter
                error = self._format_error(e, filename)
            finally:
                frameworks.ACTIVE = None

            t_ran = time.perf_counter()
            cuda = None
            if rec.events or rec.files:
                names, modules = self._python_names(rec)
                try:
                    cuda = cudagen.generate(
                        rec.events, rec.storage_meta, names, modules,
                        source=source, cell=n, seed=torch.initial_seed(), truncated=rec.truncated,
                        arch=_cuda_arch(), files=rec.files, filename=filename,
                    )
                except Exception:
                    code = "// CUDA generation failed:\n// " + traceback.format_exc().replace("\n", "\n// ")
                    cuda = {"files": [{"name": f"cell{n}.cu", "lang": "cpp", "code": code,
                                       "line_map": []}], "ops": [], "stats": {}}
            return {
                "execution_count": n,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "result": result,
                "error": error,
                "cuda": cuda,
                "timing": {"run": round(t_ran - t_start, 3), "translate": round(time.perf_counter() - t_ran, 3)},
            }

    @staticmethod
    def _preimport(tree):
        """Import traced frameworks before the cell runs so their hooks see its ops."""
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            for name in names:
                if name.split(".")[0] in frameworks.TRACKED_MODULES:
                    try:
                        importlib.import_module(name)
                    except Exception:
                        pass  # not a module (e.g. `from jax import numpy`) or not installed

    @staticmethod
    def _format_error(e, filename):
        tb = e.__traceback__
        while tb is not None and tb.tb_frame.f_code.co_filename != filename:
            tb = tb.tb_next
        if isinstance(e, SyntaxError):
            lines = traceback.format_exception_only(type(e), e)
        else:
            lines = traceback.format_exception(type(e), e, tb)
        return {"ename": type(e).__name__, "evalue": str(e), "traceback": lines}
