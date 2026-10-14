# Tests

This source checkout includes a small regression suite for installation and public interface maintenance.

| Test | Scope |
| --- | --- |
| `test_installation_layout.py` | Original public packages, canonical GPU namespace, compiler sources, catalogs, and templates. |
| `test_resource_paths.py` | Configuration compatibility, resource caching, mocked downloads, and the resource CLI. |
| `test_product_tool_paths.py` | Compiler and SDK path selection, private child configuration, and build-directory handling. |
| `test_native_capsule_sources.py` | Native source/header dependency closure, source generation, drift, and path confinement. |
| `test_gpu_vector_protocol.py` | Vector protocol, cleanup, shared buffers, grouped workers, and per-world RNG using fake environments. |
| `gpu_product_compatibility/test_camera_contract.py` | Camera keys, shapes, segmentation mapping, and observation clocks using an injected bridge. |
| `gpu_product_compatibility/test_observable_extensions.py` | Original Observable callbacks, filters, clocks, and owner transfer using a fake facade. |
| `gpu_product_compatibility/test_render_model_owner.py` | Isolation of graphics model storage using an injected renderer. |

## Running

Run from the repository root after installing the project and its dependencies:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s tests/gpu_product_compatibility -p 'test_*.py'
python tests/test_gpu_vector_protocol.py
```

The vector protocol file exposes standalone test functions, so run its final command separately from unittest discovery.

These tests use temporary fixtures and mock or injected runtime components. They do not compile CUDA, execute GPU physics, render through a real OpenGL context, or benchmark performance. They do not establish full-application numerical qualification.

Historical algorithm experiments, evidence capture, and private data fixtures remain in the development workspace and are not part of this source release.
