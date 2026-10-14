"""Cold fusion of contiguous independent-world stages in Newton iterations.

The masked maxiter schedule and line search are retained. S04 transpose,
Hessian assembly, and S07 remain synchronization boundaries. No shared solver
module or sealed instance is modified.
"""
import ast
import copy
import hashlib
import inspect
import textwrap

import warp as wp
from .compose import compose, _load


def fused_solver_class(package, base_class=None):
    """Return a cold solver subclass for a native-dimension candidate package.

    package.solver must expose FwdConstraintSolver. Optional base_class allows
    another cold subclass to supply unrelated stage launchers.
    """
    original = package.solver.FwdConstraintSolver
    base = original if base_class is None else base_class
    method = original._loop_body
    namespace = dict(method.__globals__)
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    fn = tree.body[0]
    inline = {"_s08b", "_gauss", "_count", "_s06_launch", "_s08c"}
    kernels, groups, pending, body = [], [], [], []

    def descriptor(call):
        if not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != "self":
            return None
        name = call.func.attr
        if name == "_glue":
            if len(call.args) != 2 or call.keywords:
                raise ValueError("iteration glue must retain one-thread-per-world dimension")
            return eval(ast.unparse(call.args[0]), namespace), copy.deepcopy(call.args[1].elts)
        if name not in inline:
            return None
        helper = getattr(original, name)
        helper_fn = ast.parse(textwrap.dedent(inspect.getsource(helper))).body[0]
        parameters = [a.arg for a in helper_fn.args.args[1:]]
        if len(parameters) != len(call.args) or call.keywords:
            raise ValueError((name, "helper argument shape changed"))
        substitutions = dict(zip(parameters, call.args))

        class Replace(ast.NodeTransformer):
            def visit_Name(self, node):
                return copy.deepcopy(substitutions[node.id]) if node.id in substitutions else node

        launches = [n for n in ast.walk(helper_fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "wp.launch"]
        if not launches:
            glue = [n for n in ast.walk(helper_fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "self._glue"]
            if len(glue) == 1:
                return descriptor(Replace().visit(copy.deepcopy(glue[0])))
        if len(launches) != 1:
            raise ValueError((name, "single original launch required"))
        launch = Replace().visit(copy.deepcopy(launches[0]))
        if ast.unparse(launch.args[1]) != "ws.nworld":
            raise ValueError((name, "global producer boundary cannot fuse"))
        args = []
        kw = {k.arg: k.value for k in launch.keywords}
        for key in ("inputs", "outputs"):
            if key in kw:
                if not isinstance(kw[key], ast.List):
                    raise ValueError("explicit stage argument list required")
                args.extend(kw[key].elts)
        return eval(ast.unparse(launch.args[0]), helper.__globals__), args

    def flush():
        if not pending:
            return
        stages, bindings = [], {}
        for index, (kernel, expressions) in enumerate(pending):
            params = list(inspect.signature(kernel.func).parameters)
            if len(params) != len(expressions):
                raise ValueError((kernel.key, "original signature mismatch"))
            mapping = {name: "s%d_%s" % (index, name) for name in params}
            stages.append((kernel, mapping))
            bindings.update({mapping[name]: expression for name, expression in zip(params, expressions)})
        identity = hashlib.sha256("|".join(inspect.getsource(k.func) for k, _ in pending).encode()).hexdigest()[:16]
        kernel = compose("newton_iteration_%d_%s" % (len(kernels), identity), stages)
        index = len(kernels)
        kernels.append(kernel)
        groups.append([k.key for k, _ in pending])
        args = ast.List(elts=[bindings[name] for name in kernel.fusion_argument_names], ctx=ast.Load())
        statement = ast.parse("wp.launch(_iteration_kernels[0], ws.nworld, inputs=[], device=ws.device, stream=ws.stream)").body[0]
        statement.value.args[0].slice = ast.Constant(index)
        statement.value.keywords[0].value = args
        body.append(statement)
        pending.clear()

    for statement in fn.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            desc = descriptor(statement.value)
            if desc is not None:
                pending.append(desc)
                continue
        flush()
        body.append(statement)
    flush()
    fn.body = body
    fn.decorator_list = []
    ast.fix_missing_locations(fn)
    namespace["_iteration_kernels"] = tuple(kernels)
    namespace["wp"] = wp
    source = ast.unparse(fn) + "\n"
    package_identity = hashlib.sha256(original.__module__.encode()).hexdigest()[:16]
    fused_method = _load("newton_iteration_launcher_" + package_identity, source, namespace)._loop_body
    return type("FusedIterationSolver", (base,), {"__module__": original.__module__, "_loop_body": fused_method,
        "fusion_iteration_groups": tuple(tuple(g) for g in groups),
        "fusion_iteration_kernels": tuple(kernels),
        "fusion_iteration_source": source,
        "fusion_iteration_original": original})
