"""Budget admission under changing device pressure, without allocating CUDA."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'libero/libero/gpu/host'


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


budget = load('_test_memory_budget', 'runtime/core/memory_budget.py')
planner = load('_test_collision_budget', 'runtime/collision/runtime_adapter.py')
GiB = 1024**3


class MemoryBudgetTests(unittest.TestCase):
    def device(self, free):
        return SimpleNamespace(is_cuda=True, free_memory=free, total_memory=40*GiB)

    def test_free_memory_pressure_and_explicit_limit(self):
        device = self.device(32*GiB)
        with patch.dict(sys.modules, warp=SimpleNamespace(get_device=lambda _: device)):
            result = budget.resolve_memory_budget(pending_bytes=2*GiB)
            self.assertEqual(result['available_bytes'], 32*GiB)
            self.assertEqual(result['reserved_bytes'], 2*GiB + result['headroom_bytes'])
            policy = budget.MemoryBudget(limit_bytes=12*GiB, reserve_bytes=GiB, safety_fraction=0)
            result = budget.resolve_memory_budget(policy, pending_bytes=2*GiB)
            self.assertEqual(result['collision_budget_bytes'], 9*GiB)
            result = budget.resolve_memory_budget({'limit_bytes': 2*GiB})
            self.assertEqual(result['headroom_bytes'], 512*1024**2)
            self.assertEqual(result['collision_budget_bytes'], int(1.5*GiB))
            device.free_memory = 2*GiB
            with self.assertRaises(MemoryError):
                budget.resolve_memory_budget(policy, pending_bytes=2*GiB)

    def test_more_than_six_gib_is_admitted_but_indices_remain_checked(self):
        static = dict(body_geomnum=[1]*4, geom_size=[1]*8)
        args = dict(slots=2000, call_capacity=1500, contact_capacity=512)
        plan = planner.plan_capacity(static, 64, total_device_budget_bytes=20*GiB,
                                     reserved_other_bytes=2*GiB, **args)
        self.assertGreater(plan.preflight_device_bytes, 6*GiB)
        with self.assertRaises(MemoryError):
            planner.plan_capacity(static, 64, total_device_budget_bytes=8*GiB, **args)
        with self.assertRaisesRegex(ValueError, 'int32'):
            planner.plan_capacity(static, 128, total_device_budget_bytes=10000*GiB,
                                  slots=100000, call_capacity=100000)

    def test_adaptive_plan_records_snapshot_and_revalidates_deterministically(self):
        static = dict(body_geomnum=[1]*4, geom_size=[1]*8)
        device = self.device(32*GiB)
        aliases = {'warp': SimpleNamespace(get_device=lambda _: device),
                   'libero.gpu.host.runtime.core.memory_budget': budget}
        with patch.dict(sys.modules, aliases):
            plan = planner.plan_capacity(static, 64, slots=2000, call_capacity=1500,
                                          pending_bytes=2*GiB)
            self.assertEqual(plan.memory_budget['free_bytes'], 32*GiB)
            device.free_memory = GiB  # Captured plans keep their budget for validation.
            again = planner.plan_capacity(static, 64,
                total_device_budget_bytes=plan.total_device_budget_bytes,
                reserved_other_bytes=plan.reserved_other_bytes, slots=2000, call_capacity=1500)
            self.assertEqual(plan, again)
            with self.assertRaises(MemoryError):
                planner.plan_capacity(static, 64, slots=2000, call_capacity=1500)

    def test_policy_rejects_invalid_values(self):
        for options in ({'limit_bytes': True}, {'limit_bytes': 0}, {'reserve_bytes': -1},
                        {'safety_fraction': float('nan')}, {'safety_fraction': 1}):
            with self.assertRaises(ValueError):
                budget.MemoryBudget(**options)


if __name__ == '__main__':
    unittest.main()
