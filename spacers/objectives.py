import math
import random
from abc import ABC, abstractmethod

import pyomo.environ as aml

from spacers import utilities
from spacers.model import (VaccineObjective, insert_conjunction_constraints,
                           insert_disjunction_constraints,
                           insert_indicator_sum_beyond_threshold)


class SimpleImmunogenicityObjective(VaccineObjective):
    '''
    this objective maximizes the sum of the immunogenicities of the selected epitopes
    '''

    def insert_objective(self, model):
        model.Immunogenicity = aml.Var()
        model.AssignImmunogenicity = aml.Constraint(rule=lambda model: model.Immunogenicity == sum(
            model.x[i, j] * model.EpitopeImmunogen[i] for i in model.Epitopes for j in model.EpitopePositions
        ))

        model.Objective = aml.Objective(rule=lambda model: model.Immunogenicity, sense=aml.maximize)
        return model.Objective


class EffectiveImmunogenicityObjective(VaccineObjective):
    '''
    this objective estimates the effective immunogenicity based on the epitopes
    that are cleaved correctly. this latter part is estimated using monte-carlo
    trials of Bernoulli variables whose probability is derived from the cleavage
    scores. nb: cleavage only happens at least four positions apart
    '''

    def __init__(self, mc_draws, cleavage_prior):
        if mc_draws < 1:
            raise ValueError('use at least one monte-carlo draws (and many more for reliable results)')
        self._mc_draws = mc_draws

        if cleavage_prior < 0 or cleavage_prior > 1:
            raise ValueError('cleavage prior must be between 0 and 1')
        self._cleavage_prior = math.log(cleavage_prior)

    def _compute_bernoulli_trials(self, model):
        '''
        runs Bernoulli trials for every position based on the cleavage scores
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
        '''
        compute cleavage positions from the Bernoulli trials and cleavage in the previous positions
        '''

        if model.MinSpacerLength == model.MaxSpacerLength:
            self._compute_cleavage_locations_fixed_spacer_length(model)
        else:
            self._compute_cleavage_locations_variable_spacer_length(model)

    def _compute_cleavage_locations_fixed_spacer_length(self, model):
        '''
        compute cleavage positions from the Bernoulli trials and cleavage in the previous positions
        when the spacers have fixed length
        '''

        # cleavage happens at position p if
        #  - p is not in the first four locations of the vaccine, and
        #  - there was a successful Bernoulli trial at p, and
        #  - there was no cleavage in the previous four positions
        def get_vars_cleavage_trials(model, i, p):
            variables = []
            if p > 3:
                variables = [
                    model.McBernoulliTrials[i, p],
                ] + [
                    1 - model.McCleavageTrials[i, p + j]
                    for j in range(-4, 0)
                    if p + j <= model.SequenceLength
                ]
            return variables

        insert_conjunction_constraints(
            model, 'McCleavageTrials',
            model.McDrawIndices * model.SequencePositions,
            get_vars_cleavage_trials, default=0
        )

    def _compute_cleavage_locations_variable_spacer_length(self, model):
        '''
        compute cleavage positions from the Bernoulli trials and cleavage in the previous positions
        when the spacers have variable length
        '''

        # this variable contains the actual cleavage indicators
        model.McCleavageTrials = aml.Var(model.McDrawIndices * model.SequencePositions,
                                         domain=aml.Binary, initialize=0)

        # this variable indicates whether position k is not blocking cleavage at position p
        # k does not block p iff
        #  - it is further than 4 amino acids, or
        #  - it does not contain an amino acid, or
        #  - it was not cleaved.
        # (easier to understand its negation via de morgan)
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
        #  - there is an amino acid at position p, and
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
        '''
        compute cleavage frequencies, i.e. the average for every
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
        '''
        for every epitope position, decide if it was recovered or not
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
            model, 'McRecoveredEpitopePositions',
            model.McDrawIndices * model.EpitopePositions,
            get_conj_vars,
        )

    @staticmethod
    def _compute_effective_immunogen(model):
        '''
        compute the effective immunogenicity, i.e. the average immunogenicity of the
        epitopes in the vaccine, weighted by the recovery probability of each epitope
        '''

        # compute the frequency of epitope recovery for every position
        model.McRecoveredEpitopesFrequency = aml.Var(model.EpitopePositions,
                                                     domain=aml.Binary, initialize=0)
        model.AssingRecoveredEpitopesFrequency = aml.Constraint(
            model.EpitopePositions, rule=lambda model, p: model.McRecoveredEpitopesFrequency[p] == sum(
                model.McRecoveredEpitopePositions[i, p] for i in model.McDrawIndices
            ) / model.McDrawCount
        )

        # compute the epitope immunogenicity in each position
        model.PositionImmunogenicity = aml.Var(model.EpitopePositions,
                                               domain=aml.Reals, initialize=0)
        model.AssignPositionImmunogenicity = aml.Constraint(
            model.EpitopePositions, rule=lambda model, p: model.PositionImmunogenicity[p] == sum(
                model.x[e, p] * model.EpitopeImmunogen[e]
                for e in model.Epitopes
            )
        )

        # compute the effective immunogenicity as the epitope immunogenicity times frequency of recovery
        #
        # nb: we could compute this as sum(
        #     McRecoveredEpitopePositions[i, p] * x[e, p] * EpitopeImmungen[e]
        #     for i, p, e
        # ) but this would blow up in our face, as linearizin the quadratic constraints
        # will require o(i| x |p| x |e|) ~ 1e6 constraints and variables !
        model.EffectiveImmunogenicity = aml.Var(domain=aml.Reals, initialize=0)
        model.AssignEffectiveImmunogenicity = aml.Constraint(rule=lambda model: sum(
            model.PositionImmunogenicity[p] * model.McRecoveredEpitopesFrequency[p]
            for p in model.EpitopePositions
        ) == model.EffectiveImmunogenicity)

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
