import sys

from spacers import constraints as spco
from spacers import objectives as spob
from spacers import utilities
from spacers.model import ModelParams, SolverFailedException, StrobeSpacer
from spacers.pcm import DoennesKohlbacherPcm

print(sys.path)


class BaseTest:
    epitopes = ['MGNKWSKSK', 'MGNKWSKSI', 'ARHHKAREL', 'SSNTEATNA', 'NNCLLHPMS']
    immunogens = [0.082, 0.100, 0.115, 0.047, 0.016]
    min_spacer_length = 2
    max_spacer_length = 4
    vaccine_length = 2
    objective = spob.SimpleImmunogenicityObjective()

    def __init__(self, constraints, correct_immunogen=None, correct_epitopes=None, correct_spacers=None):
        self.constraints = constraints
        self._correct_epitopes = correct_epitopes
        self._correct_spacers = correct_spacers
        self._correct_immunogen = correct_immunogen

    def solve_and_check(self):
        solution = self.solve()
        self.check_solution_consistent()
        return solution

    def solve(self):
        self.params = ModelParams(
            self.epitopes, self.immunogens,
            min_spacer_length=self.min_spacer_length,
            max_spacer_length=self.max_spacer_length,
            vaccine_length=self.vaccine_length,
            pcm=DoennesKohlbacherPcm(),
        )

        self.problem = StrobeSpacer(
            params=self.params,
            vaccine_constraints=self.constraints,
            vaccine_objective=self.objective,
        ).build_model()

        self.solution = self.problem.solve()
        return self.solution

    def check_solution_consistent(self):
        correct_cleavages = self.params.pcm.cleavage_per_position(self.solution.sequence)
        if self._correct_epitopes is not None and self._correct_spacers is not None:
            correct_sequence = ''.join([
                x for epi, spa in zip(self._correct_epitopes, self._correct_spacers + [''])
                for x in (self.epitopes[epi], spa)
            ])
        else:
            correct_sequence = None

        # mandatory controls (spacer length and correct computed cleavage)
        assert all(len(spacer) >= 2 for spacer in self.solution.spacers)
        assert all(
            abs(correct - computed) < 1e-6
            for correct, computed in zip(correct_cleavages, self.solution.cleavage)
        )

        # optional controls
        assert self._correct_epitopes is None or self.solution.epitopes == self._correct_epitopes
        assert self._correct_spacers is None or self.solution.spacers == self._correct_spacers
        assert correct_sequence is None or self.solution.sequence == correct_sequence
        assert self._correct_immunogen is None or abs(self.solution.immunogen - self._correct_immunogen) < 1e-6
