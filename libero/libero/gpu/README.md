# GPU Source Documentation

The source tree keeps the GPU implementation under `libero/libero/gpu/`. Both source checkouts and installed packages expose it through the canonical `libero.gpu` namespace.

## Directory Layout

```text
gpu/
  cuda/              Handwritten CUDA and adapted upstream device code
    position/        Kinematics, coordinate transforms, and mass matrices
    collision/       Collision detection, contact processing, native geometry
    constraints/     Constraint rows, impedance, reference values, projection
    dynamics/        Velocity, forces, and smooth acceleration
    newton/          Newton iterations, Hessians, line search, MathDx bridges
    control/         Joint and OSC controllers
    integration/     Euler integration
    postprocess/     RNE, sensors, and Newton finalization
    observations/    Low-dimensional observations and scene sampling
    runtime/         Publication, restoration, copies, and version checks
    common/          Shared mathematical helpers
  host/              Python host code
    factory.py       Physics preparation and environment assembly entry points
    bindings/        CUDA argument bindings and component installation
    build/           Compilation, parameter ABI, and kernel construction
    runtime/         Owners, workspaces, scheduling, and lifecycle protocols
    assembly/        Composition, pipelines, and MathDx integration
    adapters/        LIBERO environment and observation adapters
    paths.py         Canonical source locations
  model/             Static model construction and metadata
  reference/         Reference operators and runtime construction templates
```

## Entry Points

LIBERO users normally create environments through the original public API:

```python
from libero.libero.envs import OffScreenRenderEnv
```

Low-level callers can use:

```python
from libero.gpu.host.factory import prepare_cuda_physics
```

This entry supports `nworld` and `dataflow_profile="fast"` or `"compatible"`. The physics preparation default is `"fast"`; existing internal component profiles remain independently configured. The same module exposes `create` for environment assembly.

Start with `cuda/` to inspect device mathematics and `host/` to inspect memory, launches, caching, bindings, and restoration. Some cold construction paths read operators and workspace templates from `reference/`, so that directory is part of the runtime source dependency set.

## Configuration and Lifecycle

[Algorithm Configuration](ALGORITHM_NOTES.md) describes algorithm choices, defaults, and capacity boundaries. Batched cone/rebuild selection and Post radix/topology reuse retain their configured conditions. Newton islands and collision temporal caching are explicit experimental choices with their original admission checks.

- Device mathematics uses CUDA; standard matrix operations use cuBLASDx and cuSOLVERDx.
- FMA is enabled; precise division, square root, and non-FTZ behavior are retained.
- Model, source, toolchain, policy, and owner changes require the corresponding cold admission or explicit invalidation.
- Capacity checks, stable ordering, errors, fallback paths, and lifecycle ownership remain part of each implementation's contract.

Independent-function measurements do not establish whole-application speedups or coverage of every model and platform.

## Licensing and Sources

Project-original contributions use the project MIT license, subject to the component boundaries in the root [LICENSE](../../../LICENSE). Upstream code keeps its own notices and terms. [Third-Party Notices](THIRD_PARTY_NOTICES.md) records adapted code, external dependencies, and retained license texts.
