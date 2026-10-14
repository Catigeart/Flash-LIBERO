"""Measure complete same-task GPU batch steps, excluding setup/reset/policy."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--task-ids', type=int, nargs='+')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--camera', action='store_true', help='two RGB cameras at 128x128')
    parser.add_argument('--memory-limit-gib', type=float)
    parser.add_argument('--output', type=Path, default=Path('results/gpu_batch.json'))
    args = parser.parse_args()
    if min(args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error('steps/repeats must be positive; warmup must be nonnegative')
    from libero.libero import benchmark
    from libero.libero.envs import GpuBatchEnv
    suite = benchmark.get_benchmark(args.suite)()
    task_ids = list(range(suite.n_tasks)) if args.task_ids is None else args.task_ids
    if any(not 0 <= i < suite.n_tasks for i in task_ids):
        parser.error('task ID outside the selected suite')
    policy = {} if args.memory_limit_gib is None else {
        'limit_bytes': int(args.memory_limit_gib * 1024**3)}
    report = dict(suite=args.suite, num_envs=args.num_envs, steps=args.steps,
                  warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                  use_camera_obs=args.camera, device=args.device, tasks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        print(f'Building task {task_id}: {task.name}', flush=True)
        start = time.perf_counter()
        try:
            env = GpuBatchEnv(suite.get_task_bddl_file_path(task_id), args.num_envs,
                              init_states=suite.get_task_init_states(task_id), seed=args.seed,
                              device=args.device, memory_budget=policy,
                              use_camera_obs=args.camera, ignore_done=True)
        except MemoryError as error:
            row = dict(task_id=task_id, task=task.name, status='insufficient_memory',
                       setup_seconds=time.perf_counter()-start, error=str(error))
            report['tasks'].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            continue
        with env:
            setup = time.perf_counter() - start
            # Pre-generated bounded actions; inference/file IO is outside timing.
            actions = np.stack([
                np.random.RandomState(args.seed+i).uniform(
                    -.05, .05, (args.warmup+args.steps, env.action_dim))
                for i in range(args.num_envs)], axis=1)
            samples = []
            for repeat in range(args.repeats):
                env.reset(seed=args.seed)
                for action in actions[:args.warmup]:
                    env.step(action)
                before = env.diagnostics()
                start = time.perf_counter()
                for action in actions[args.warmup:]:
                    env.step(action)
                elapsed = time.perf_counter() - start  # step already checks/waits.
                after = env.diagnostics()
                delta = np.array(after['actual_integration_count'])-before['actual_integration_count']
                if not np.all(delta == args.steps*env.substeps):
                    raise RuntimeError('actual integration counts differ from measured workload')
                samples.append(dict(seconds=elapsed, ms_per_batch=elapsed*1000/args.steps,
                                    env_steps_per_second=args.num_envs*args.steps/elapsed))
            row = dict(task_id=task_id, task=task.name, status='ok', setup_seconds=setup,
                       samples=samples, diagnostics=after,
                       median_ms_per_batch=float(np.median([s['ms_per_batch'] for s in samples])),
                       median_env_steps_per_second=float(np.median([s['env_steps_per_second'] for s in samples])))
            report['tasks'].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
