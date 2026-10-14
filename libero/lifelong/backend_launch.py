"""Explicit backend wiring for original training and checkpoint evaluation.

Policy, optimizer, checkpoints, horizons and physics algorithms stay original.
GPU runtime provider retains diagnostic/production admission responsibility.
"""
import argparse,ast,hashlib,importlib,json,sys
from pathlib import Path

def wave_video_factory(original,offset,records):
    def create(folder,*args,**kwargs):
        unique=Path(folder)/f'gpu_rollout_{offset:03d}'
        records.append({'rollout':offset,'folder':str(unique),'single_video_path':str(unique/'video.mp4')})
        return original(str(unique),*args,**kwargs)
    return create

class TrackedEnvironment:
    def __init__(self,actual):self.actual=actual;self.closed=False
    def __getattr__(self,name):return getattr(self.actual,name)
    def close(self):
        if not self.closed:
            result=self.actual.close();self.closed=True;self.actual=None;return result

def environment_constructor(backend, runtime_factory=None, *, constructor=None, installer=None, created=None):
    if backend not in ('cpu','gpu'):raise ValueError('explicit cpu/gpu backend required')
    if (backend=='gpu')!=callable(runtime_factory):raise ValueError('GPU provider required; CPU provider prohibited')
    def create(**options):
        if 'physics_backend' in options:raise ValueError('backend cannot be overridden by legacy env arguments')
        if backend=='gpu':
            if installer is None:
                from libero.libero.envs.bddl_utils import get_problem_info
                from libero.gpu.host.adapters.environment.factory import install_model_only_gpu_factory
                install_model_only_gpu_factory(get_problem_info(options['bddl_file_name'])['problem_name'],runtime_factory,replace=True)
            else:installer(options['bddl_file_name'],runtime_factory)
        target=constructor
        if target is None:
            from libero.libero.envs.env_wrapper import OffScreenRenderEnv
            target=OffScreenRenderEnv
        env=target(physics_backend=backend,**options)
        if getattr(env,'physics_backend',None)!=backend:
            try:env.close()
            finally:raise RuntimeError('original environment returned wrong backend')
        if created is not None:
            env=TrackedEnvironment(env);created.append(env)
        return env
    return create

class WaveVector:
    def __init__(self,actual,offset):self.actual=actual;self.offset=offset
    def __getattr__(self,name):return getattr(self.actual,name)
    def seed(self,seed=None):
        if seed is not None and not isinstance(seed,int):raise ValueError('original scalar seed required for20-wave route')
        return self.actual.seed(None if seed is None else seed+self.offset)

def single_world_vector(constructors, *, vector_factory=None, seed_offset=0):
    if len(constructors)!=1:
        raise ValueError('current GPU original-entry execution requires one resident environment; no CPU fallback')
    if vector_factory is None:
        from libero.libero.envs import DummyVectorEnv
        vector_factory=DummyVectorEnv
    result=vector_factory(constructors)
    return WaveVector(result,seed_offset) if seed_offset else result

def specialize_evaluation_main(source, environment_count):
    """Prepare one wave of original checkpoint evaluation, without shrinking20.

    Changes only concurrent env_num, global initialization-state offset and
    returning original stats. Original policy/math/horizon/settle code stays.
    """
    if environment_count!=1:raise ValueError('current resident GPU evaluation supports explicit count1')
    tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
    matches=[x for x in ast.walk(fn) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env_num' for t in x.targets)]
    if len(matches)!=1 or not isinstance(matches[0].value,ast.Constant) or matches[0].value.value!=20:
        raise ValueError('original checkpoint evaluation environment-count source changed')
    matches[0].value=ast.copy_location(ast.Constant(environment_count),matches[0].value)
    indexes=[x for x in ast.walk(fn) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='indices' for t in x.targets)]
    if len(indexes)!=1 or ast.unparse(indexes[0].value)!='np.arange(env_num) % init_states.shape[0]':
        raise ValueError('original initial-state cycling source changed')
    indexes[0].value=ast.parse('(np.arange(env_num) + _gpu_wave_start) % init_states.shape[0]',mode='eval').body
    fn.body.append(ast.Return(value=ast.Name(id='eval_stats',ctx=ast.Load())))
    return ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))

def evaluate_waves(target, specialized, *, total_rollouts=20):
    """Preserve original20 rollout indices; one actual GPU owner per wave.

    Original main reloads/checks the same checkpoint each wave. Policy resets
    at each independent episode, matching independent batch-row histories.
    This is explicit same-process scheduling, not batchedW20 acceptance.
    """
    if total_rollouts!=20:raise ValueError('original checkpoint evaluation has exactly20 rollouts')
    torch=target.torch;saved_save=torch.save;writes=[];stats=[];videos=[]
    def collect(value,path,*args,**kwargs):
        if not isinstance(value,dict) or set(value)!={'loss','success_rate'}:
            raise ValueError('unexpected original evaluation save; no policy serialization interception')
        writes.append((value,path,args,kwargs))
    try:
        torch.save=collect
        for offset in range(total_rollouts):
            namespace=dict(vars(target));namespace['_gpu_wave_start']=offset
            namespace['SubprocVectorEnv']=lambda constructors,offset=offset:single_world_vector(constructors,seed_offset=offset)
            if hasattr(target,'VideoWriter'):namespace['VideoWriter']=wave_video_factory(target.VideoWriter,offset,videos)
            exec(compile(specialized,target.__file__,'exec'),namespace)
            try:stats.append(namespace['main']())
            except SystemExit as error:
                if error.code in (None,0):raise RuntimeError('original evaluation exited before full20 rollout evidence') from error
                raise
    finally:torch.save=saved_save
    if len(writes)!=total_rollouts or len({str(x[1]) for x in writes})!=1:
        raise ValueError('original result path/count changed across GPU waves')
    if any(row['loss']!=stats[0]['loss'] for row in stats):
        raise ValueError('original checkpoint test loss differs between independent waves')
    final={'loss':stats[0]['loss'],'success_rate':sum(row['success_rate'] for row in stats)/total_rollouts}
    _,path,args,kwargs=writes[0];saved_save(final,path,*args,**kwargs)
    if videos:
        if len(videos)!=20 or len({r['folder'] for r in videos})!=20:raise ValueError('all20 unique video owners required')
        Path(str(path)+'.gpu-waves.json').write_text(json.dumps({'scope':'original20 rollouts, independent wave video destinations','videos':videos},indent=2))
    print('Original20 rollout coverage preserved; GPU concurrency1; success_rate='+str(final['success_rate']))
    return final

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target',required=True,choices=('train','evaluate'))
    parser.add_argument('--backend',required=True,choices=('cpu','gpu'))
    parser.add_argument('--runtime-factory')
    parser.add_argument('--environment-count',type=int,default=None,
        help='GPU checkpoint evaluation concurrency1; original20 rollouts remain20')
    args,original_args=parser.parse_known_args(argv)
    if original_args[:1]==['--']:original_args=original_args[1:]
    if (args.backend=='gpu')!=bool(args.runtime_factory):parser.error('GPU requires provider; CPU prohibits provider')
    from libero.lifelong.backend_entry import import_callable
    provider=import_callable(args.runtime_factory) if args.runtime_factory else None
    created=[]
    constructor=environment_constructor(args.backend,provider,created=created)
    metric=importlib.import_module('libero.lifelong.metric')
    target=importlib.import_module('libero.lifelong.main' if args.target=='train' else 'libero.lifelong.evaluate')
    saved=[];oldargv=sys.argv[:]
    def bind(module,name,value):saved.append((module,name,getattr(module,name)));setattr(module,name,value)
    try:
        bind(metric,'OffScreenRenderEnv',constructor)
        if args.backend=='gpu':bind(metric,'SubprocVectorEnv',single_world_vector)
        if args.target=='evaluate':
            bind(target,'OffScreenRenderEnv',constructor)
            if args.backend=='gpu':
                if args.environment_count!=1:parser.error('GPU legacy checkpoint evaluation requires --environment-count 1')
                bind(target,'SubprocVectorEnv',single_world_vector)
                path=Path(target.__file__);source=path.read_text()
                specialized=specialize_evaluation_main(source,args.environment_count)
                print('GPU evaluation 20waves/concurrency1; original source SHA256='+hashlib.sha256(path.read_bytes()).hexdigest())
                invoke=lambda:evaluate_waves(target,specialized)
            else:
                if args.environment_count is not None:parser.error('CPU reference keeps original workload')
                invoke=target.main
        else:
            if args.environment_count is not None:parser.error('training metric count comes from original Hydra eval configuration')
            invoke=target.main
        sys.argv=[target.__file__,*original_args]
        return invoke()
    finally:
        primary=sys.exc_info()[1];cleanup=[]
        for env in reversed(created):
            try:env.close()
            except BaseException as error:cleanup.append(error)
        sys.argv=oldargv
        for module,name,value in reversed(saved):setattr(module,name,value)
        if cleanup:
            if primary is not None:primary.backend_cleanup_errors=[repr(error) for error in cleanup]
            else:raise cleanup[0]

if __name__=='__main__':main()
