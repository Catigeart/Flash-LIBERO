<div align="center">

# Flash-LIBERO

### GPU Simulation for Lifelong Robot Learning

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [Project Website](https://libero-project.github.io/) · [LIBERO Paper](https://arxiv.org/abs/2306.03310) · [LIBERO Documentation](https://lifelong-robot-learning.github.io/LIBERO/)

</div>

**Flash-LIBERO** adds GPU physics to [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), a benchmark for knowledge transfer in multitask and lifelong robot learning. Install this distribution and keep your existing LIBERO environment code: simulation uses the GPU by default.

- Original LIBERO imports, task definitions, environment interfaces, and benchmark suites.
- 130 manipulation tasks across **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-100**, including the **LIBERO-90 / LIBERO-10** split.
- CUDA physics with FP64 arithmetic and cuBLASDx / cuSOLVERDx integration.
- Existing observation, rendering, controller, state, and vector-environment interfaces.
- LIBERO policy architectures, lifelong learning methods, and dataset tools.

## Contents

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Tests](#tests)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Installation

### Requirements

- Python **3.10 or newer**.
- An NVIDIA GPU, NVIDIA driver, and CUDA Toolkit with `nvcc` available.
- CMake and a C/C++ compiler for native dependencies.
- NVIDIA MathDx. If no installation is configured, the runtime downloads its pinned **25.12.1 CUDA12** package on first use.

The current development environment uses Ubuntu 22.04 under WSL2 and CUDA 13.1.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'cmake>=3.22,<4'
python -m pip install .
```

For an editable installation, replace the last command with `python -m pip install -e .`.

This distribution installs the `libero` package. Use an environment containing this installation when running your LIBERO code.

### Task resources

Task BDDL files and scene/object XML definitions are included. Meshes, textures, initial states, and demonstration datasets are stored separately. Existing LIBERO resource configurations continue to work; missing default assets and initial states are fetched into the resource cache.

To prepare assets and initial states before creating an environment:

```bash
python -m libero.libero.resource_cli assets init_states
```

The first environment initialization may compile model-specific CUDA kernels. Subsequent initialization can reuse the build cache.

## Getting Started

Use the same task and environment imports as upstream LIBERO. No GPU backend argument or factory registration is required.

```python
import os

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_spatial"]()
task_id = 0
task = task_suite.get_task(task_id)

task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)
print(task.language)

env = OffScreenRenderEnv(
    bddl_file_name=task_bddl_file,
    camera_heights=128,
    camera_widths=128,
)
try:
    env.seed(0)
    obs = env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    obs = env.set_init_state(init_states[0])

    for _ in range(10):
        obs, reward, done, info = env.step([0.0] * 7)
finally:
    env.close()
```

Available suite names include `libero_spatial`, `libero_object`, `libero_goal`, `libero_90`, and `libero_10`. LIBERO uses sparse task-completion rewards.

### Native GPU batches

Use `GpuBatchEnv` to advance multiple worlds of **one task and one shared static
model** in a native GPU batch. This is separate from the compatible
`DummyVectorEnv` / `SubprocVectorEnv` interfaces.

```python
import numpy as np
from libero.libero import benchmark
from libero.libero.envs import GpuBatchEnv

suite = benchmark.get_benchmark("libero_10")()
with GpuBatchEnv(
    suite.get_task_bddl_file_path(0),
    num_envs=8,
    init_states=suite.get_task_init_states(0),
    seed=0,
    use_camera_obs=False,
) as env:
    obs = env.reset()
    obs, rewards, successes, infos = env.step(np.zeros((8, env.action_dim)))
    env.reset(id=[0, 3])
    print(env.diagnostics())
```

Observations are dictionaries with a leading environment axis. `step` advances
all worlds; `reset` and `set_init_state` accept selected IDs in caller order.
There is no automatic reset. `set_init_state` preserves controller history,
whereas `reset` clears episode/controller/observation state. Fixture poses are
sampled once at construction and shared; reset selects movable/robot states
from the supplied bank. If no bank is supplied, the corresponding official
`.pruned_init` file is loaded. Different fixture layouts or tasks require
separate batches. Model replacement and interactive rendering are not exposed
through this entry. Set `use_camera_obs=True` for offscreen images; image rendering
and task goal evaluation are still performed at the host boundary.
This entry retains the low-level physics factory's diagnostic construction mode.
Set `strict_build_receipts=True` to additionally use the product entry's full
compiler/library dependency receipts and artifact cache; this can add substantial
cold-build time. Runtime numerical, source-binding, index and owner checks remain
in force in both modes. This option does not grant numerical qualification.

Device memory is budgeted from the selected GPU's **current free memory** during
construction, after existing physics allocations. The planner reserves an
estimate for subsequent solver/controller/observation allocations plus the
larger of 512 MiB or 10% of the available budget as headroom. An optional `memory_budget`
dictionary accepts `limit_bytes`, `reserve_bytes`, and `safety_fraction`; the
limit applies to allocations remaining at the planning boundary. The snapshot
is available in `env.memory_budget`. These are estimates, so subsequent CUDA
allocation failures remain possible. Index, model, contact, and kernel resource
checks remain enforced. Insufficient memory raises an error without changing N
or truncating contacts.

To time the complete batch interface, excluding setup, reset, and policy inference:

```bash
python -m benchmark_scripts.benchmark_gpu_batch --suite libero_10 --num-envs 8 --warmup 20 --steps 600 --repeats 3
# Add --camera for two 128x128 RGB cameras, or --task-ids 0 for one task.
```

The JSON report includes setup time, repeated milliseconds per batch control
step, environment steps per second, actual integration counts, and the budget
snapshot. These measurements include host goal/observation work and are not
device-only kernel timings. Tasks rejected by the memory preflight are recorded
with `status="insufficient_memory"`; the runner continues with the other tasks
at the requested N.

For task generation and benchmark usage, see the [upstream documentation](https://lifelong-robot-learning.github.io/LIBERO/).

## Datasets

Download the original human demonstration datasets:

```bash
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

To download a single suite:

```bash
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_spatial --use-huggingface
```

Dataset choices are `libero_spatial`, `libero_object`, `libero_goal`, and `libero_100`. The download location follows your LIBERO dataset configuration. Datasets are also available on [Hugging Face](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).

## Training

The original LIBERO training interface is retained. For example:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
python -m libero.lifelong.main \
    seed=0 \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_transformer_policy \
    lifelong=base
```

Choose a benchmark from `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_90`, or `LIBERO_10`; a policy from `bc_rnn_policy`, `bc_transformer_policy`, or `bc_vilt_policy`; and a method from `base`, `er`, `ewc`, `packnet`, or `multitask`.

## Evaluation

Evaluate a saved checkpoint using the original evaluation command:

```bash
python -m libero.lifelong.evaluate \
    --benchmark libero_spatial \
    --task_id 0 \
    --algo base \
    --policy bc_transformer_policy \
    --seed 0 \
    --ep 50 \
    --load_task 0 \
    --device_id 0
```

Set the suite, task, seed, epoch, and training-task index to match your checkpoint. Use `--experiment_dir` to select its experiment directory.

## Configuration

| Variable | Purpose |
| --- | --- |
| `LIBERO_CONFIG_PATH` | Directory containing your LIBERO resource-path configuration. |
| `LIBERO_RESOURCE_PATH` | Cache location for automatically fetched task assets and initial states. |
| `LIBERO_GPU_CACHE_DIR` | GPU build and runtime cache directory. |
| `LIBERO_NVCC` | Path to the CUDA compiler. |
| `LIBERO_MATHDX_ROOT` | MathDx installation root containing `include/`, `external/cutlass/include/`, and `lib/libcusolverdx.a`. |

For CUDA source layout and low-level entry points, see [GPU source documentation](libero/libero/gpu/README.md).

## Tests

A small regression suite covers installation layout, resource and toolchain paths, source dependencies, vector protocols, and observation/graphics ownership. See [tests/README.md](tests/README.md) for commands and scope.

## Contributing

Development and releases use this Git repository. See [CONTRIBUTING.md](CONTRIBUTING.md) for editable installation, focused branches, CPU checks, and tagged releases. The release tool builds checked wheel/source artifacts directly from the current working tree.

## Acknowledgments

This project builds on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), developed by the Lifelong Robot Learning team. We thank the LIBERO authors and the developers of MuJoCo, libccd, robosuite, NVIDIA Warp, and NVIDIA MathDx for their work.

## License

Project-original Flash-LIBERO contributions are licensed under the **MIT License**, with copyright attributed to the Flash-LIBERO contributors. See the root [LICENSE](LICENSE) for the project license and component scope.

Upstream LIBERO-derived code retains its original [MIT license and copyright notice](licenses/LIBERO-MIT.txt). Adapted MuJoCo code retains **Apache-2.0**, and libccd code retains **BSD-3-Clause**. These third-party terms are preserved and are not replaced by the project MIT license.

See [Third-Party Notices](libero/libero/gpu/THIRD_PARTY_NOTICES.md) for source attribution and license texts. External SDKs and dependencies, including NVIDIA MathDx, retain their own licenses.
