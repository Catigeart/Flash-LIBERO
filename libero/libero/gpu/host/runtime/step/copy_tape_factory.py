from pathlib import Path
import hashlib
import numpy as np

def fixed_profile_factory(**config):
    """Actual fixed-model reset scope, explicitly separate from task raw models."""
    from libero.gpu.host.runtime.dynamics.model_pack import ModelPack
    from scripts.gpu_phase06_dynamics.run_author import FROZEN_SPEC,FROZEN_MANIFEST
    from .prepare_copy_tape_opt import prepare_fixed_copy_tape_physics
    if config:raise ValueError('fixed reset profile factory has no mutable configuration')
    root=Path('/srv/libero-split-receive-20261004-r01/received-r01/evidence/objects')
    pack=ModelPack(mjb_path=root/'3eea791e54a911d83159c7891050be5447468c62efe026611f91414273156b42',
        frozen_spec_path=FROZEN_SPEC,frozen_manifest_path=FROZEN_MANIFEST,
        velocity_bias_path=root/'745a77cae16c461e0cab5576249231f8d54e310de721d4d7fa423bcb6186f48c')
    state=dict(qpos=np.asarray(pack.model.qpos0,np.float64).reshape(1,48).copy(),
        qvel=np.zeros((1,43),np.float64),time=np.zeros(1,np.float64),ctrl=np.zeros((1,9),np.float64),
        qacc_warmstart=np.zeros((1,43),np.float64),qfrc_applied=np.zeros((1,43),np.float64),
        xfrc_applied=np.zeros((1,38,6),np.float64))
    bundle=prepare_fixed_copy_tape_physics(pack,state=state,with_controller=True)
    bundle.controller.bootstrap_from_current_physics(bundle.runtime.intent,stream=bundle.stream)
    import warp as wp
    wp.synchronize_stream(bundle.stream)
    bundle.profile_manifest=dict(source_epoch='fixed-parallel-copy-and-tape-cold-r01',
        state_source=dict(scope='declared native reset of original fixed reference model; not task reset',
            model_sha256=hashlib.sha256(pack.mjb_path.read_bytes()).hexdigest(),
            fields={name:dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest()) for name,a in state.items()}),
        controller_bootstrapped=True,copy_equivalence_report='37399458bb21827efef7681997841e65dee334453cef49d0a5641aaf0bc36fb1',
        production_task_acceptance=False)
    return bundle
