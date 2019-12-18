import strobe_spacers as sspa
from pcm import DoennesKohlbacherPcm


class BaseTest:
    epitopes = ['MGNKWSKSK', 'MGNKWSKSI', 'ARHHKAREL', 'SSNTEATNA', 'NNCLLHPMS']
    immunogens = [0.082, 0.100, 0.115, 0.047, 0.016]
    min_spacer_length = 2
    max_spacer_length = 4
    vaccine_length = 2

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
        self.problem = sspa.StrobeSpacer(
            self.epitopes, self.immunogens,
            min_spacer_length=self.min_spacer_length,
            max_spacer_length=self.max_spacer_length,
            vaccine_length=self.vaccine_length,
            vaccine_constraints=self.constraints,
            pcm=DoennesKohlbacherPcm(),
        )

        self.solution = self.problem.solve()
        return self.solution

    def check_solution_consistent(self):
        correct_cleavages = self.problem._pcm.cleavage_per_position(self.solution.sequence)
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


def test_n_terminus_cleavage_gap():
    gap = 0.5
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[sspa.MinimumNTerminusCleavageGap(gap)],
        correct_immunogen=0.215,
    )

    solution = test.solve_and_check()

    second_epitope_start = 9 + len(solution.spacers[0])
    assert all(
        solution.cleavage[second_epitope_start] >= solution.cleavage[second_epitope_start + i] + gap
        for i in range(-4, 2) if i != 0
    )


def test_min_cleavage_inside_spacers():
    min_cleavage = 0.5
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[sspa.BoundCleavageInsideSpacers(min_cleavage, None)],
        correct_immunogen=0.215,
    )

    solution = test.solve_and_check()

    assert all(
        solution.cleavage[9 + pos] >= min_cleavage
        for pos in range(len(solution.spacers[0]))
    )


def test_max_cleavage_inside_spacers():
    max_cleavage = 0.2
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[sspa.BoundCleavageInsideSpacers(None, max_cleavage)],
    )

    solution = test.solve_and_check()

    assert all(
        solution.cleavage[9 + pos] <= max_cleavage
        for pos in range(len(solution.spacers[0]))
    )


def test_max_cleavage_inside_epitope():
    max_cleavage = 0.8
    test = BaseTest(
        constraints=[sspa.MaximumCleavageInsideEpitopes(max_cleavage)],
        correct_immunogen=0.162,
        correct_epitopes=[3, 2],
        correct_spacers=['CCC']
    )

    solution = test.solve_and_check()

    assert all(solution.cleavage[pos] <= max_cleavage for pos in range(9))
    assert all(solution.cleavage[len(solution.cleavage) - pos] <= max_cleavage for pos in range(1, 10))


def test_n_terminus_cleavage():
    cleavage = 0.5
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[sspa.MinimumNTerminusCleavage(cleavage)],
        correct_immunogen=0.215,
    )

    solution = test.solve_and_check()

    second_epitope_start = 9 + len(solution.spacers[0])
    assert solution.cleavage[second_epitope_start] >= cleavage

