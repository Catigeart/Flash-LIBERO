"""Direct model-only GPU construction with an explicit provider; no registry writes."""
def retry_creation_cleanup(error):
    """Retry actual retained owners after a construction error; preserve primary."""
    from .cleanup_ownership_r02 import ReleaseOwner
    pending=list(getattr(error,'pending_gpu_cleanup',()))
    if not all(isinstance(item,ReleaseOwner) for item in pending):
        raise TypeError('typed creation cleanup owners required')
    for item in tuple(pending):
        try:
            item.release()
        finally:
            pending=[lease for lease in pending if not lease.released]
            error.pending_gpu_cleanup=tuple(pending)
    return None

def create_task_for_provider(problem_name, runtime_factory, bddl_file_name, **options):
    from libero.gpu.model import build_task_model
    from .task_domain_r01 import GpuTaskEnvironment
    from .cleanup_ownership_r02 import ReleaseOwner, attach_cleanup
    if not isinstance(problem_name,str) or not problem_name:
        raise ValueError('nonempty exact problem name required')
    if not callable(runtime_factory):
        raise TypeError('explicit runtime factory required')

    def rebuild(xml_string, previous=None):
        built = build_task_model(bddl_file_name, problem_name=problem_name,
                                 xml_string_override=xml_string, **options)
        reuse = getattr(runtime_factory, "rebuild_components", None)
        components = reuse(built, previous) if previous is not None and callable(reuse) else runtime_factory(built)
        try:
            components.validate()
        except BaseException as exc:
            if components is previous:
                # Reuse provider owns rollback/quarantine of its transaction.
                raise
            try:
                components.release()
            except BaseException as close_error:
                exc.gpu_cleanup_errors=list(getattr(exc,'gpu_cleanup_errors',()))+[repr(close_error)]
                attach_cleanup(exc,ReleaseOwner(components,method_name='release'))
            raise
        return components

    components = rebuild(None)
    task=GpuTaskEnvironment.__new__(GpuTaskEnvironment)
    try:
        GpuTaskEnvironment.__init__(task, components, rebuild=rebuild)
        return task
    except BaseException as exc:
        # Constructor failure otherwise loses the only actual GPU owner
        # before ControlEnv returns. Attach diagnostic-only facts before
        # release, preserving the original execution exception.
        reader = getattr(getattr(components.sim, '_runtime', None), 'diagnostic_evidence', None)
        if callable(reader):
            try: exc.gpu_diagnostic_only = reader()
            except BaseException as diagnostic_error:
                exc.gpu_diagnostic_reader_error = repr(diagnostic_error)
        cleanup_errors = []
        # Constructor cold updates may already have replaced the initial owner.
        # Use the partially built task's actual public close path when available.
        owner=task if getattr(task,'components',None) is not None else components
        method_name='close' if owner is task else 'release'
        try:
            getattr(owner,method_name)()
        except BaseException as close_error:
            cleanup_errors.append(repr(close_error))
            attach_cleanup(exc,ReleaseOwner(owner,method_name=method_name))
        if cleanup_errors: exc.gpu_cleanup_errors = list(getattr(exc,'gpu_cleanup_errors',())) + cleanup_errors
        raise
