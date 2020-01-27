import math
import random
from abc import ABC, abstractmethod

from spacers.model import (insert_indicator_sum_beyond_threshold,
                           insert_disjunction_constraints,
                           insert_conjunction_constraints)
import pyomo.environ as aml

from spacers import utilities
from spacers.model import VaccineObjective


class MonteCarloVaccineObjective(VaccineObjective):
    '''
    base class for objectives applied to the Monte Carlo model
    '''


class MonteCarloEffectiveImmunogenicityObjective(MonteCarloVaccineObjective):
    '''
    this objective estimates the effective immunogenicity based on the epitopes
    that are cleaved correctly. this latter part is estimated using monte-carlo
    trials of Bernoulli variables whose probability is derived from the cleavage
    scores. nb: cleavage only happens at least four positions apart
    '''

    def __init__(self, mc_draws=100, cleavage_prior=0.1):
        self._mc_draws = mc_draws
        if cleavage_prior < 0 or cleavage_prior > 1:
            raise ValueError('cleavage prior must be between 0 and 1')
        self._cleavage_prior = math.log(cleavage_prior)

    def _compute_bernoulli_trials(self, model):
        ''' runs Bernoulli trials for every position
            based on the cleavage scores
        '''

        # a Bernoulli trial in position p successful if the cleavage score
        # is larger than the random number at position p
        insert_indicator_sum_beyond_threshold(
            model, 'McBernoulliTrials',
            model.McDrawIndices * model.SequencePositions,
            larger_than_is=1,
            get_variables_bounds_fn=lambda model, i, p: (
                [model.i[p]],
                25,
                model.McRandoms[i, p],
            )
        )

    def _compute_cleavage_locations(self, model):
        ''' compute cleavage positions from the Bernoulli trials and
            cleavage in the previous positions.
        '''

        # this variable contains the actual cleavage indicators
        model.McCleavageTrials = aml.Var(model.McDrawIndices * model.SequencePositions,
                                         domain=aml.Binary, initialize=0)

        # this variable indicates whether position k is not blocking cleavage at position p
        # k does not block p iff
        #  - it is further than 4 amino acids, or
        #  - it does not contain an amino acid, or
        #  - it was not cleaved.
        insert_disjunction_constraints(
            model, 'McCleavageNotBlocked',
            model.McDrawIndices * model.SequencePositions * model.SequencePositions,
            lambda model, i, p, k: [
                1 - model.d[p, k],
                1 - model.c[k],
                1 - model.McCleavageTrials[i, k],
            ] if k < p else [],
            default=1.0
        )

        # cleavage happens at position p if
        #  - p is not in the first four locations of the vaccine, and
        #  - there was a successful Bernoulli trial at p, and
        #  - none of the other positions block cleavage at p
        def get_vars_cleavage_trials(model, i, p):
            variables = []
            if p > 3:
                variables = [
                    model.c[p],
                    model.McBernoulliTrials[i, p],
                ] + [
                    model.McCleavageNotBlocked[i, p, j]
                    for j in range(0, p)
                ]
            return variables

        insert_conjunction_constraints(
            model, model.McCleavageTrials, None,
            get_vars_cleavage_trials, default=0
        )

    @staticmethod
    def _compute_cleavage_frequencies(model):
        ''' compute cleavage frequencies, i.e. the average for every
            position along the monte-carlo trials
        '''
        model.McCleavageProbs = aml.Var(model.SequencePositions)
        model.McComputeProbs = aml.Constraint(
            model.SequencePositions,
            rule=lambda model, p: model.McCleavageProbs[p] == sum(
                model.McCleavageTrials[i, p] for i in model.McDrawIndices
            ) / model.McDrawCount
        )

    def _compute_recovered_epitopes(self, model):
        ''' for every epitope position, decide if it was recovered or not
        '''

        # the epitope in position p is recovered if
        #  - there was no cleavage inside the epitope itself, and
        #  - there was cleavage at the n-terminus or p is the first epitope, and
        #  - there was cleavage at the c-terminus or p is the last epitope
        def get_conj_vars(model, i, p):
            nterminus_position = p * (model.EpitopeLength + model.MaxSpacerLength)
            cterminus_position = p * (model.EpitopeLength + model.MaxSpacerLength) + model.EpitopeLength

            inside_not_cleavage = [
                1 - model.McCleavageTrials[i, j]
                for j in range(nterminus_position + 1, cterminus_position)
            ]

            if p == model.VaccineLength - 1:
                return inside_not_cleavage + [model.McCleavageTrials[i, nterminus_position]]
            elif p == 0:
                return inside_not_cleavage + [model.McCleavageTrials[i, cterminus_position]]
            else:
                return inside_not_cleavage + [
                    model.McCleavageTrials[i, cterminus_position],
                    model.McCleavageTrials[i, nterminus_position],
                ]

        insert_conjunction_constraints(
            model, 'McRecoveredEpitopes',
            model.McDrawIndices * model.EpitopePositions,
            get_conj_vars,
        )

    @staticmethod
    def _compute_effective_immunogen(model):
        ''' compute the effective immunogenicity, i.e. the average immunogenicity of the
            epitopes in the vaccine, weighted by the recovery probability of each epitope
        '''

        # effective immunogenicity for each Monte Carlo simulation (useful for debugging)
        model.McEffectiveImmunogen = aml.Var(model.McDrawIndices, domain=aml.Reals, initialize=0)
        model.McAssignEffectiveImmunogen = aml.Constraint(
            model.McDrawIndices * model.EpitopePositions,
            rule=lambda model, i, p: model.McEffectiveImmunogen[i] == sum(
                model.McRecoveredEpitopes[i, p] * model.x[e, p] * model.EpitopeImmunogen[e]
                for e in model.Epitopes for p in model.EpitopePositions
            )
        )

        model.EffectiveImmunogenicity = aml.Var(domain=aml.Reals, initialize=0)
        model.AssignEffectiveImmunogenicity = aml.Constraint(rule=lambda model: sum(
            model.McEffectiveImmunogen[i] for i in model.McDrawIndices
        ) / model.McDrawCount == model.EffectiveImmunogenicity)

    def insert_objective(self, model):
        model.McDrawIndices = aml.RangeSet(0, self._mc_draws - 1)
        model.McDrawCount = aml.Param(initialize=self._mc_draws)
        model.McRandoms = aml.Param(
            model.McDrawIndices * model.SequencePositions,
            initialize=lambda *_: math.log(random.random()) - self._cleavage_prior
        )

        self._compute_bernoulli_trials(model)
        self._compute_cleavage_locations(model)
        self._compute_cleavage_frequencies(model)
        self._compute_recovered_epitopes(model)
        self._compute_effective_immunogen(model)

        model.Objective = aml.Objective(rule=lambda model: model.EffectiveImmunogenicity, sense=aml.maximize)
        return model.Objective
