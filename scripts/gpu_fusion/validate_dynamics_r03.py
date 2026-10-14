"""Small real GPU D phase comparisons against the admitted batch-two entry."""
from libero.gpu.host.paths import GPU_ROOT, REPO_ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import mujoco
import warp as wp
from libero.gpu.host.runtime.core.namespace import require_canonical_gpu_namespace
require_canonical_gpu_namespace()
from libero.gpu.model.factory import BuiltTaskModel
from libero.gpu.host.runtime.step.shape_factory import _native_intake_package, default_integration_state, actual_decoder_record
from libero.gpu.host.assembly.compose.factory_r02 import prepare_fused_physics
from libero.gpu.host.assembly.pipeline.dynamics import prepare_dynamics, runtime_methods, prepare_post, FusedRuntimePostAdapter, plan_catalog
from libero.gpu.host.assembly.pipeline.dynamics import _shifted_external_alias

MODEL_ROOT = Path('/mnt/d/GPT/LIBERO-GPU/alltasks-execution/20261009-r01/bulk-evidence/g09-jar-only-repeated-recovery-r05')


def arrays(owner):
    found, seen = {}, set()
    def walk(value, path, depth):
        if isinstance(value, wp.array):
            found[path] = value
            return
        if id(value) in seen or depth > 6: return
        seen.add(id(value))
        if isinstance(value, dict): items = value.items()
        elif isinstance(value, (tuple, list)): items = enumerate(value)
        elif hasattr(value, '__dict__') and not isinstance(value, type):
            items = vars(value).items()
        else: return
        for name, child in items:
            if str(name) in ('registry', 'spec', '_d_fusion_plans', '_d_fusion_contexts', 'metadata', 'robot'): continue
            if isinstance(child, (wp.array, dict, tuple, list)) or (
                hasattr(child, '__dict__') and not isinstance(child, type) and (
                type(child).__module__.startswith(('libero.', 'warp.')) or
                type(child).__name__ == 'SimpleNamespace')):
                walk(child, path + '.' + str(name), depth + 1)
    walk(owner, 'root', 0)
    return found


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True)
    p.add_argument('--cache'); p.add_argument('--alias-only', action='store_true')
    p.add_argument('--position-only', action='store_true'); args = p.parse_args()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / 'work' / 'tmp').mkdir(parents=True); os.chdir(out / 'work')
    wp.config.kernel_cache_dir = args.cache or str(out / 'warp-cache')
    os.environ['P06_STEP_COLD_COPY'] = 'cn'
    metadata = json.loads((MODEL_ROOT / 'actual-native-metadata.json').read_text())
    model = mujoco.MjModel.from_binary_path(str(MODEL_ROOT / 'actual-model.mjb'))
    built = BuiltTaskModel(model, '', None, metadata)
    built.metadata['decoder_record'] = actual_decoder_record(built)
    spec, package = _native_intake_package(built)
    bundle = prepare_fused_physics(built, state=default_integration_state(model),
        model_spec=spec, constraint_package=package, mode='diagnostic', with_controller=True)
    bundle.controller.bootstrap_from_current_physics(bundle.runtime.intent)
    d = prepare_dynamics(bundle.dynamics)
    post = prepare_post(bundle.post)
    sw = bundle.runtime.forward.s.workspace
    calls = {
        'launch_position': {}, 'launch_velocity': {},
        'launch_bias': dict(reference_done=sw.reference_ready, reference_error=sw.status,
            reference_written=sw.ref_epoch, reference_expected=bundle.runtime.forward.expected_forward),
        'launch_force': dict(ctrl=bundle.runtime.forward.ctrl, ctrl_valid=bundle.runtime.forward.ctrl_valid),
        'gather_controller_inputs': dict(mask=bundle.runtime.intent),
    }
    methods = runtime_methods()
    refs = arrays(d) | {'first_error': d.first_error}
    refs.update({'post_' + k: v for k, v in arrays(post).items()})
    cases = []
    try:
        if not args.position_only:
            shifted = d.position.body.diag.body_route.reshape((-1,))[1:1+d.capacity]
            shifted.assign(np.ones(d.capacity, np.int32))
            assert _shifted_external_alias(d, (shifted,))
            wp.synchronize_stream(d.stream)
            before = {k: a.numpy() for k, a in refs.items()}
            d.gather_controller_inputs(shifted)
            wp.synchronize_stream(d.stream)
            baseline = {k: a.numpy() for k, a in refs.items()}
            for k, a in refs.items(): a.assign(before[k])
            methods['gather_controller_inputs'](d, shifted)
            wp.synchronize_stream(d.stream)
            failed = [k for k, a in refs.items() if a.numpy().tobytes() != baseline[k].tobytes()]
            assert not failed, ('shifted_mask', failed)
            cases.append(dict(name='shifted_mask_fallback', compared=len(refs), passed=True, max_abs=0.0))
            print(json.dumps(cases[-1]), flush=True)
        for blocked in (() if args.alias_only else (False, True)):
            if blocked: d.first_error.assign(np.full(d.capacity, 208, np.int32))
            for name in (('launch_position',) if args.position_only else (*calls, 'post')):
                wp.synchronize_stream(d.stream)
                before = {k: a.numpy() for k, a in refs.items()}
                if name == 'post': post.enqueue()
                else: getattr(d, name)(**calls[name])
                wp.synchronize_stream(d.stream)
                baseline = {k: a.numpy() for k, a in refs.items()}
                for k, a in refs.items(): a.assign(before[k])
                if name == 'post': FusedRuntimePostAdapter.enqueue(post)
                else: methods[name](d, **calls[name])
                wp.synchronize_stream(d.stream)
                failed = [k for k, a in refs.items() if a.numpy().tobytes() != baseline[k].tobytes()]
                assert not failed, (name, blocked, failed)
                case = dict(name=name, blocked=blocked, compared=len(refs), max_abs=0.0, passed=True)
                cases.append(case); print(json.dumps(case), flush=True)
        report = dict(passed=True, cases=cases, plans=plan_catalog(d), post_plans=plan_catalog(post),
            source_sha256=hashlib.sha256((GPU_ROOT / 'host/assembly/pipeline/dynamics.py').read_bytes()).hexdigest())
    except Exception as e:
        report = dict(passed=False, cases=cases, error=repr(e))
        raise
    finally:
        (out / 'REPORT.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__': main()
