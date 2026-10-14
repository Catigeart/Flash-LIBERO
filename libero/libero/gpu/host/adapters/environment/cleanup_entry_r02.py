"""Explicit new cleanup lifecycle registration into the original GPU entry."""
from .factory import register_gpu_task_factory

def install_model_only_gpu_factory(problem_name, runtime_factory, *, replace=False):
    """Wire a true runtime assembler into the original ControlEnv GPU entry.

    runtime_factory(built) returns GpuTaskComponents with actual device owners.
    It must reject unsupported model identities/layouts; no CPU task constructor,
    CPU physical data or fake ready fields may be used to supply missing stages.
    """
    if not callable(runtime_factory):
        raise ValueError("runtime_factory must be callable")

    def factory(*, bddl_file_name, **options):
        from libero.gpu.model import build_task_model
        from .task_cleanup_r02 import GpuTaskEnvironment

        def rebuild(xml_string, previous=None):
            built = build_task_model(bddl_file_name, problem_name=problem_name,
                                     xml_string_override=xml_string, **options)
            reuse = getattr(runtime_factory, "rebuild_components", None)
            components = reuse(built, previous) if previous is not None and callable(reuse) else runtime_factory(built)
            try:
                components.validate()
            except BaseException:
                if components is previous:
                    # Reuse provider owns rollback/quarantine of its transaction.
                    raise
                try:
                    components.observations.close()
                finally:
                    components.close_runtime()
                raise
            return components

        components = rebuild(None)
        try:
            return GpuTaskEnvironment(components, rebuild=rebuild)
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
            for close in (components.release,):
                try: close()
                except BaseException as close_error: cleanup_errors.append(repr(close_error))
            if cleanup_errors: exc.gpu_cleanup_errors = cleanup_errors
            raise

    register_gpu_task_factory(problem_name, factory, replace=replace)
    return factory
