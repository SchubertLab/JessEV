import sys

import pyomo.environ as aml

from spacers import constraints as spco
from spacers import objectives as spob
from spacers import utilities
from spacers.model import ModelParams, SolverFailedException, StrobeSpacer
from spacers.pcm import DoennesKohlbacherPcm
from spacers.test.base_test import BaseTest


def test_effective_immunogenicity():
    test = BaseTest([])
    test.objective = spob.EffectiveImmunogenicityObjective(
        mc_draws=10, cleavage_prior=0.1
    )

    solution = test.solve_and_check()
    model = test.problem._model

    counts = [0] * len(solution.sequence)
    effective_immunogen = 0.0
    recovery = []
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
        recovery.append([1, 1])
        for j in range(10):  # check epitope and c-terminus
            if (j < 8 and cuts[j] > 0) or (j == 9 and cuts[j] < 1):
                recovery[-1][0] = 0
                break

        second_start = 9 + len(solution.spacers[0])
        for j in range(9):
            k = j + second_start
            if (j == 0 and cuts[k] < 1) or (j > 0 and cuts[k] > 0):
                recovery[-1][1] = 0
                break

        assert int(aml.value(model.McRecoveredEpitopePositions[i, 0])) == recovery[-1][0]
        assert int(aml.value(model.McRecoveredEpitopePositions[i, 1])) == recovery[-1][1]

    # compute epitope recovery frequencies
    recovery_freqs = [
        sum(r[0] for r in recovery) / len(recovery),
        sum(r[1] for r in recovery) / len(recovery),
    ]
    for i in range(len(recovery_freqs)):
        assert abs(aml.value(model.McRecoveredEpitopesFrequency[i]) - recovery_freqs[i]) < 1e-6

    # compute position immunogenicity
    pos_immunogen = [test.immunogens[e] for e in solution.epitopes]
    for i in range(len(pos_immunogen)):
        assert abs(aml.value(model.PositionImmunogenicity[i]) - pos_immunogen[i]) < 1e-6

    # compute effective immunogenicity
    computed_effective_ig = sum(i * f for i, f in zip(pos_immunogen, recovery_freqs))
    assert abs(aml.value(model.EffectiveImmunogenicity) - computed_effective_ig) < 1e-6

    # now test recovery probabilities
    for i in range(len(solution.sequence)):
        computed = aml.value(model.McCleavageProbs[i])
        assert abs(counts[i] / 10 - computed) < 1e-6

    assert abs(solution.immunogen - effective_immunogen / 10) < 1e-6
