"""Guarded ControlEnv selection with receipt-bound construction options."""
from .factory import register_gpu_task_factory
from .task_creation_domain_r01 import create_task_for_provider, retry_creation_cleanup

def _control_env_default_options():
    # Source defaults and actual source OSC loader; no environment constructor.
    import inspect
    from libero.libero.envs.env_wrapper import ControlEnv
    import robosuite as suite
    defaults = {name: parameter.default for name, parameter in
                inspect.signature(ControlEnv.__init__).parameters.items()
                if name not in ("self", "bddl_file_name", "physics_backend", "kwargs")
                and parameter.default is not inspect.Parameter.empty}
    controller = defaults.pop("controller")
    defaults["controller_configs"] = suite.load_controller_config(default_controller=controller)
    return defaults

def _check_options(admission, defaults, options):
    from libero.gpu.host.runtime.production.domain_lease import SupportedDomainAdmission
    from libero.gpu.host.runtime.production.admission import QualificationError
    if not isinstance(admission, SupportedDomainAdmission):
        raise TypeError("verified supported-domain admission object required")
    import json
    sealed = getattr(admission, "_candidate_json", None)
    if not isinstance(sealed, str):
        raise QualificationError("supported admission candidate must be sealed JSON text")
    try:
        candidate = json.loads(sealed)
    except (TypeError, ValueError) as error:
        raise QualificationError("invalid supported admission candidate JSON") from error
    domain = candidate.get("domain") if isinstance(candidate, dict) else None
    expected = domain.get("environment_options") if isinstance(domain, dict) else None
    if not isinstance(expected, dict):
        raise QualificationError("supported admission environment options must be a mapping")
    actual = dict(defaults)
    actual.update(options)
    if actual != expected:
        raise QualificationError("requested construction options outside supported source domain")

def install_provider(problem_name, runtime_factory, *, replace=False):
    if not isinstance(problem_name, str) or not problem_name:
        raise ValueError("nonempty exact problem name required")
    if not callable(runtime_factory):
        raise TypeError("explicit actual runtime provider required")
    admission = getattr(runtime_factory, "admission", None)
    defaults = _control_env_default_options() if admission is not None else None
    def factory(*, bddl_file_name, **options):
        if admission is not None:
            _check_options(admission, defaults, options)
        return create_task_for_provider(problem_name, runtime_factory, bddl_file_name, **options)
    register_gpu_task_factory(problem_name, factory, replace=replace)
    return factory
