import pyomo.environ as aml
import sys
print(sys.path)

from spacers.monte_carlo import constraints as ssc
from spacers.monte_carlo import objectives as sso
from spacers.model import ModelParams, StrobeSpacer, SolverFailedException
from spacers.pcm import DoennesKohlbacherPcm
from spacers.test.base_test import BaseTest


def test_effective_immunogenicity():
    test = BaseTest([])
    test.objective = sso.MonteCarloEffectiveImmunogenicityObjective(
        mc_draws=10, cleavage_prior=0.1
    )

    solution = test.solve_and_check()
    model = test.problem._implementation._model

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
