"""Cold-only fixed-model diagnostic adapter for the qualified LS-copy candidate.

Independent wholeloop component source epoch, not inherited production grant.
Inputs/canonical owners are bound before alias and solver construction. The
original FixedModelRules guard remains mandatory; generic dimensions use a
separate native-model cold factory.
"""
from libero.gpu.host.runtime.newton.runtime_adapter import FixedNewtonRuntimeAdapter
from libero.gpu.host.runtime.newton.solver import FwdConstraintSolver
from libero.gpu.host.runtime.newton.parallel_tape import parallel_tape_solver_type


class DiagnosticParallelTapeFixedAdapter(FixedNewtonRuntimeAdapter):
    def __init__(self, workspace, *, rules, solver=None):
        if solver is not None:
            raise ValueError('cold candidate requires bindings before solver creation')
        super().__init__(workspace, rules=rules)

    def seal(self):
        if not self._sealed:
            self._prepare_public()
            self.ws._build_s07_args()
            solver_type = parallel_tape_solver_type(FwdConstraintSolver)
            self.solver = solver_type(self.ws)
            self._sealed = True
        return self
