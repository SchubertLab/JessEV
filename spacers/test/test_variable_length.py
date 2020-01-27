import sys

from spacers import constraints as spco
from spacers import objectives as spob
from spacers import utilities
from spacers.model import ModelParams, SolverFailedException, StrobeSpacer
from spacers.pcm import DoennesKohlbacherPcm
from spacers.test.base_test import BaseTest


def test_n_terminus_cleavage_gap():
    gap = 0.5
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[spco.MinimumNTerminusCleavageGap(gap)],
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
        constraints=[spco.BoundCleavageInsideSpacers(min_cleavage, None)],
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
        constraints=[spco.BoundCleavageInsideSpacers(None, max_cleavage)],
    )

    solution = test.solve_and_check()

    assert all(
        solution.cleavage[9 + pos] <= max_cleavage
        for pos in range(len(solution.spacers[0]))
    )


def test_max_cleavage_inside_epitope():
    max_cleavage = 0.8
    test = BaseTest(
        constraints=[spco.MaximumCleavageInsideEpitopes(max_cleavage)],
        correct_immunogen=0.162,
        correct_epitopes=[3, 2],
    )

    solution = test.solve_and_check()

    assert all(solution.cleavage[pos] <= max_cleavage for pos in range(9))
    assert all(solution.cleavage[len(solution.cleavage) - pos] <= max_cleavage for pos in range(1, 10))


def test_n_terminus_cleavage():
    cleavage = 0.5
    # there are several optimal solutions, so we only check the objective
    test = BaseTest(
        constraints=[spco.MinimumNTerminusCleavage(cleavage)],
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
        constraints=[spco.MinimumCTerminusCleavage(cleavage)],
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
        constraints=[spco.MinimumCoverageAverageConservation(
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
        constraints=[spco.MinimumCoverageAverageConservation(
            epitope_coverage, min_conservation=2
        )]
    )
    solution = test.solve_and_check()
    assert set(solution.epitopes) == set([0, 1])
