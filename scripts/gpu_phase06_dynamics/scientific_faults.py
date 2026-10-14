"""Test-only actual device output mutations; never a production or truth path."""
import warp as wp
wp.set_module_options({"enable_backward":False,"fuse_fp":True,"fast_math":False})

@wp.kernel
def dirty_f64(a:wp.array(dtype=wp.float64), index:int, mode:int):
    if mode==2:
        a[index]=-a[index]
    elif mode==3:
        a[index]=a[index]+wp.float64(0.125)
    else:
        a[index]=a[index]+wp.float64(1.e100)

@wp.kernel
def dirty_i32(a:wp.array(dtype=wp.int32), index:int, mode:int):
    a[index]=a[index]+1

@wp.kernel
def add_f64(a:wp.array(dtype=wp.float64), index:int, delta:wp.float64):
    # Delta is frozen from the input-only source graph before any native/GPU answer.
    a[index]=a[index]+delta

@wp.kernel
def flip_signbit_u64(a:wp.array(dtype=wp.uint64), index:int):
    # Actual test-only sign-bit XOR; scalar arithmetic negation can implement
    # 0-x and leave +0 unchanged, which is not a signed-zero payload control.
    a[index]=a[index] ^ wp.uint64(9223372036854775808)
