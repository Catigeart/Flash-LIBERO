"""Original ControlEnv GPU selection with explicit private provider closures."""
from .factory import register_gpu_task_factory
from .task_creation_r01 import create_task_for_provider, retry_creation_cleanup

def install_provider(problem_name,runtime_factory,*,replace=False):
    if not isinstance(problem_name,str) or not problem_name:
        raise ValueError('nonempty exact problem name required')
    if not callable(runtime_factory):raise TypeError('explicit actual runtime provider required')
    def factory(*,bddl_file_name,**options):
        return create_task_for_provider(problem_name,runtime_factory,bddl_file_name,**options)
    register_gpu_task_factory(problem_name,factory,replace=replace)
    return factory
