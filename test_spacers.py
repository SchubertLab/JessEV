import strobe_spacers as sspa
from pcm import DoennesKohlbacherPcm
import pyomo.environ as aml


class BaseTest:
    epitopes = ['MGNKWSKSK', 'MGNKWSKSI', 'ARHHKAREL', 'SSNTEATNA', 'NNCLLHPMS']
    immunogens = [0.082, 0.100, 0.115, 0.047, 0.016]
    min_spacer_length = 2
    max_spacer_length = 4
    vaccine_length = 2
    objective = sspa.ImmunogenicityObjective()

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
            vaccine_objective=self.objective,
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
    assert solution.cleavage[0] < cleavage


def test_c_terminus_cleavage():
    cleavage = 0.5
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[sspa.MinimumCTerminusCleavage(cleavage)],
        correct_immunogen=0.215,
    )

    solution = test.solve_and_check()

    spacer_start = 9
    assert solution.cleavage[spacer_start] >= cleavage


def test_coverage():
    # the only way to cover three options is to select the first and last epitopes
    epitope_coverage = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ]
    test = BaseTest(
        constraints=[sspa.MinimumCoverageAverageConservation(
            epitope_coverage, min_coverage=3
        )]
    )
    solution = test.solve_and_check()
    assert set(solution.epitopes) == set([0, 4])


def test_conservation():
    # the only way to have an average conservation of 2 is to select the first two epitopes
    epitope_coverage = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]

    test = BaseTest(
        constraints=[sspa.MinimumCoverageAverageConservation(
            epitope_coverage, min_conservation=2
        )]
    )
    solution = test.solve_and_check()
    assert set(solution.epitopes) == set([0, 1])


def test_effective_immunogenicity():
    test = BaseTest([])
    test.objective = sspa.MonteCarloEffectiveImmunogenicityObjective(
        mc_draws=10, cleavage_prior=0.1
    )

    solution = test.solve_and_check()
    model = test.problem._model

    counts = [0] * len(solution.sequence)
    effective_immunogen = 0.0
    for i in range(10):
        # perform simulation using the same random numbers as the milp
        # compute cleavage positions
        cuts = []
        last = -1
        for p in range(len(solution.sequence)):
            cleavage = solution.cleavage[p]
            is_cut = 0
            if cleavage >= model.McRandoms[i, p]:
                assert int(aml.value(model.McBernoulliTrials[i, p])) == 1
                if p - last > 4:
                    is_cut = 1
                    counts[p] += 1
                    last = p
            else:
                assert int(aml.value(model.McBernoulliTrials[i, p])) == 0

            cuts.append(is_cut)

        computed_cuts = [
            int(aml.value(model.McCleavageTrials[i, p]))
            for p in range(len(solution.sequence))
        ]
        assert computed_cuts == cuts

        # compute epitope recovery
        recovery = [1, 1]
        for j in range(10):  # check epitope and c-terminus
            if (j < 8 and cuts[j] > 0) or (j == 9 and cuts[j] < 1):
                recovery[0] = 0
                break

        second_start = 9 + len(solution.spacers[0])
        for j in range(9):
            k = j + second_start
            if (j == 0 and cuts[k] < 1) or (j > 0 and cuts[k] > 0):
                recovery[1] = 0
                break

        assert int(aml.value(model.McRecoveredEpitopes[i, 0])) == recovery[0]
        assert int(aml.value(model.McRecoveredEpitopes[i, 1])) == recovery[1]

        # compute effective immunogenicity
        test_immunogen = sum(
            recovery[i] * test.immunogens[solution.epitopes[i]]
            for i in range(2)
        )
        assert abs(aml.value(model.McEffectiveImmunogen[i]) - test_immunogen) < 1e-6

        effective_immunogen += test_immunogen

    # now test recovery probabilities
    for i in range(len(solution.sequence)):
        computed = aml.value(model.McCleavageProbs[i])
        assert abs(counts[i] / 10 - computed) < 1e-6

    assert abs(solution.immunogen - effective_immunogen / 10) < 1e-6
