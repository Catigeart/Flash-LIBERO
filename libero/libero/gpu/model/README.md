# Static Task Model Construction

`build_task_model(bddl_file_name, problem_name=None, **domain_options)` returns a `BuiltTaskModel` containing the compiled `native_model`, original task XML, a static `domain_shell`, and serializable metadata. `facade_metadata()` provides copied native arrays, name indices, joint slices, XML, and options for `GpuSimFacade`.

## Construction Boundary

The original task class is allocated, and its simulation, renderer, reset, and observation initialization hooks are replaced on that instance before its constructor runs. Original task code still supplies XML, assets, robots, placement samplers, object wrappers, and goals.

Native MjModel compilation is allowed for static preparation. This path does not create a CPU physics environment and then convert it, execute MjSim dynamics, or patch a global environment class.

## Static Shell and Metadata

The shell is not an executable GPU environment. Controllers and robot runtime references are initialized later by GPU lifecycle code. Step, reset, and observation operations require that assembled runtime.

Static metadata includes ordered object indices, robot/gripper joint and actuator recipes, end-effector information, and camera/observation configuration. Original Observable constructors may sample dynamic sensors; their runtime initialization follows real GPU forward instead of static preparation.

CPU-only native handles remain in preparation. Public facades receive copied metadata and the runtime's shared arrays rather than borrowed CPU dynamics handles. Matching dimensions alone do not satisfy an existing model binding.

## Reset and Parameter Updates

BDDL reset calls `_initialization_step1`, dispatching to GPU `sim.step1` or the explicit CPU branch as configured. Fixture updates use `_set_fixture_pose`; GPU body_pos/body_quat updates pass through `update_model_parameters` and model-generation checks.

`_reset_task_placements` separates original task placement from CPU robot reset, allowing the GPU lifecycle to initialize its controller and robot state before placement. Missing GPU methods raise their existing errors rather than silently running CPU physics. Explicit CPU operation retains its original behavior.

The prepared model and metadata feed the current GPU assembly, owner replacement, restoration, and closure paths. Production GPU state is not supplied by CPU reference results.
