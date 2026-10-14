"""Dimension-specialized Newton diagnostic candidates from retained sources.

No existing fixed implementation or certificate is modified. S07's original
CUDA source, PTX and compile guards are reused byte-for-byte: its launch ABI
already takes runtime nv/nM. Candidate creation is not production acceptance.
"""
from __future__ import annotations
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import sys
import types


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def candidate_sources(nv: int, nm: int):
    if type(nv) is not int or type(nm) is not int or nv < 1 or nm < nv:
        raise ValueError('positive nv and valid packed mass size required')
    gpu = (GPU_ROOT / '.')
    sources = {
        'kernels': (GPU_ROOT / 'host/runtime/newton/kernels.py'),
        's05': (GPU_ROOT / 'reference/newton/newton_hessian/kernel.py'),
        's06': (GPU_ROOT / 'reference/newton/newton_direction/kernel.py'),
        'workspace': (GPU_ROOT / 'host/runtime/newton/workspace.py'),
        'solver': (GPU_ROOT / 'host/runtime/newton/solver.py'),
    }
    outputs, records = {}, {}
    for name, path in sources.items():
        raw = path.read_bytes()
        text = raw.decode('utf-8-sig')
        for constant, value in [('NV', nv), ('NM', nm)]:
            text = re.sub(r'^' + constant + r' = \d+$', constant + ' = ' + str(value), text, flags=re.M)
        if name == 'solver':
            text = text.replace('from libero.gpu.host.runtime.newton import kernels as K',
                                'from . import kernels as K')
            text = text.replace('from libero.gpu.reference.newton.newton_hessian import kernel as s05',
                                'from . import s05')
            text = text.replace('from libero.gpu.reference.newton.newton_direction import kernel as s06',
                                'from . import s06')
            text = text.replace('from libero.gpu.host.runtime.newton.workspace import',
                                'from .workspace import')
        product = text.encode('utf-8')
        outputs[name] = product
        records[name] = dict(source_path=str(path), source_sha256=_sha(raw),
                             generated_sha256=_sha(product))
    bridge = (GPU_ROOT / 'host/runtime/newton/s07_bridge.py')
    linesearch = (GPU_ROOT / 'reference/newton/newton_linesearch/kernel.cu')
    records['unchanged_s07'] = dict(bridge_sha256=_sha(bridge.read_bytes()),
                                   kernel_sha256=_sha(linesearch.read_bytes()),
                                   runtime_dimension_abi=True)
    identity = _sha(json.dumps(dict(nv=nv, nm=nm, files=records), sort_keys=True).encode())
    return outputs, dict(candidate_id=identity, nv=nv, nM=nm,
                         acceptance='diagnostic-pending', files=records)


def load_candidate(nv: int, nm: int, build_root: Path):
    """Materialize and import a source-bound private candidate package."""
    sources, manifest = candidate_sources(nv, nm)
    digest = manifest['candidate_id']
    destination = Path(build_root).resolve() / ('nv%d_nm%d_%s' % (nv, nm, digest[:16]))
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in sources.items():
        path = destination / (name + '.py')
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(('candidate source cache drift', str(path)))
        else:
            with path.open('xb') as file:
                file.write(raw)
    encoded = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    record_path = destination / 'candidate.json'
    if record_path.exists():
        if record_path.read_bytes() != encoded:
            raise ValueError('candidate manifest drift')
    else:
        with record_path.open('xb') as file:
            file.write(encoded)
    package_name = '_libero_newton_candidate_' + digest
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(destination)]
        sys.modules[package_name] = package
        for name in ('kernels', 's05', 's06', 'workspace', 'solver'):
            fullname = package_name + '.' + name
            spec = importlib.util.spec_from_file_location(fullname, destination / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[fullname] = module
            setattr(package, name, module)
            spec.loader.exec_module(module)
    return sys.modules[package_name], manifest


def prepare_candidate(model, *, rows: int, contacts: int, nworld: int,
                      build_root: Path, device='cuda:0'):
    """Prepare native-model topology and scalar inputs for diagnostic math.

    Caller must retain actual model/source identities and obtain independent
    acceptance before registering this candidate for production use.
    """
    import numpy as np
    import mujoco
    if int(model.opt.solver) != int(mujoco.mjtSolver.mjSOL_NEWTON):
        raise ValueError('candidate requires Newton solver')
    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        raise ValueError('candidate requires elliptic cone')
    jacobian = int(model.opt.jacobian)
    dense = int(mujoco.mjtJacobian.mjJAC_DENSE)
    auto = int(mujoco.mjtJacobian.mjJAC_AUTO)
    if jacobian != dense and not (jacobian == auto and model.nv < 60):
        raise ValueError('sparse Jacobian not yet qualified')
    if model.neq or model.ntendon or model.nplugin or int(model.opt.noslip_iterations):
        raise ValueError('equality/tendon/plugin/noslip candidate not yet implemented')
    if int(model.opt.enableflags):
        raise ValueError('optional enabled solver features require separate qualification')
    warm_disabled = int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    if int(model.opt.disableflags) & ~warm_disabled:
        raise ValueError('other disableflags require separate qualification')
    package, manifest = load_candidate(int(model.nv), int(model.nM), build_root)
    workspace = package.workspace.NewtonFwdWorkspace(
        nworld, rows, contacts, device=device, maxiter=int(model.opt.iterations),
        warmstart_enabled=not bool(int(model.opt.disableflags) & warm_disabled))
    workspace.bind_topology(np.asarray(model.dof_Madr, np.int32),
                            np.asarray(model.dof_parentid, np.int32),
                            np.asarray(model.dof_simplenum, np.int32),
                            np.asarray([model.opt.tolerance, model.stat.meaninertia], np.float64))
    solver = package.solver.FwdConstraintSolver(workspace)
    manifest = dict(manifest, runtime=dict(rows=rows, contacts=contacts, nworld=nworld,
                                         iterations=int(model.opt.iterations),
                                         tolerance=float(model.opt.tolerance),
                                         meaninertia=float(model.stat.meaninertia)))
    return workspace, solver, manifest

class _AdmittedModel:
    def __init__(self, admission):
        self.admission = admission

    def _validate(self):
        if not self.admission.execution_allowed:
            raise ValueError(('Newton candidate not admitted', self.admission.reasons))


def runtime_adapter(workspace, solver, *, model_spec, registry,
                    registered_candidate_id, source_manifest, mode='diagnostic'):
    """Canonical-public adapter gated by actual model/source qualification.

    Production requires a matching independent registry acceptance. The old
    fixed adapter continues to require its original FixedModelRules directly.
    A dimension-specialized solver cannot be substituted with the fixed one.
    """
    from libero.gpu.host.runtime.newton.runtime_adapter import _NewtonRuntimeAdapter
    dimensions = model_spec.dimensions
    if (int(dimensions.nv), int(dimensions.nM)) != (
            int(source_manifest['nv']), int(source_manifest['nM'])):
        raise ValueError('model dimensions differ from generated candidate')
    module_prefix = '_libero_newton_candidate_' + source_manifest['candidate_id']
    if (type(workspace).__module__ != module_prefix + '.workspace' or
            type(solver).__module__ != module_prefix + '.solver' or solver.ws is not workspace):
        raise ValueError('workspace/solver not from the source-bound specialization')
    if (workspace.qacc.shape != (workspace.nworld, dimensions.nv) or
            workspace.qM.shape != (workspace.nworld, dimensions.nM)):
        raise ValueError('workspace shape differs from native model layout')
    admission = registry.admit(registered_candidate_id, model_spec, mode=mode,
                               required_scope='component-runtime' if mode == 'production' else None)
    adapter = _NewtonRuntimeAdapter(workspace, rules=_AdmittedModel(admission), solver=solver)
    adapter.model_spec = model_spec
    adapter.admission = admission
    adapter.source_manifest = source_manifest
    return adapter



