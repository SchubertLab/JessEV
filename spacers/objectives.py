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

    _constraint_names = ['AssignImmunogenicity']
    _variable_names = ['Immunogenicity']

    def insert_objective(self, model, solver):
        model.Immunogenicity = aml.Var()
        model.AssignImmunogenicity = aml.Constraint(rule=lambda model: model.Immunogenicity == sum(
            model.x[i, j] * model.EpitopeImmunogen[i] for i in model.Epitopes for j in model.EpitopePositions
        ))

        self._objective_variable = model.SimpleImmunogenicityObjective = aml.Objective(
            rule=lambda model: model.Immunogenicity, sense=aml.maximize
        )

        super().insert_objective(model, solver)
        return self._objective_variable


class EffectiveImmunogenicityObjective(VaccineObjective):
    '''
    this objective estimates the effective immunogenicity based on the epitopes
    that are cleaved correctly. this latter part is estimated using monte-carlo
    trials of Bernoulli variables whose probability is derived from the cleavage
    scores. nb: cleavage only happens at least four positions apart

    this objective must be used in conjunction with MonteCarloRecoveryEstimation.
    specifying an upper bound can greatly speed up solving time, but might result in
    a sub-optimal solution if the bound is too tight
    '''

    _constraint_names = ['AssignPositionImmunogenicity', 'AssignEffectiveImmunogenicity']
    _variable_names = ['PositionImmunogenicity', 'EffectiveImmunogenicity']

    def __init__(self, upper_bound=None):
        self._ub = upper_bound

    def insert_objective(self, model, solver):
        self._compute_effective_immunogen(model)
        self._objective_variable = model.EffectiveImmunogenicityObjective = aml.Objective(
            rule=lambda model: model.EffectiveImmunogenicity, sense=aml.maximize
        )

        super().insert_objective(model, solver)

        return model.EffectiveImmunogenicityObjective

    def _compute_effective_immunogen(self, model):
        '''
        compute the effective immunogenicity, i.e. the average immunogenicity of the
        epitopes in the vaccine, weighted by the recovery probability of each epitope
        '''

        # compute the epitope immunogenicity in each position
        model.PositionImmunogenicity = aml.Var(model.EpitopePositions, domain=aml.NonNegativeReals, initialize=0)
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
        # ) but this would blow up in our face, as linearizing the quadratic constraints
        # will require o(i| x |p| x |e|) ~ 1e6 constraints and variables !
        # the trade-off is that now this is a full-fledged quadratic program..
        model.EffectiveImmunogenicity = aml.Var(domain=aml.NonNegativeReals, initialize=0, bounds=(0, self._ub))
        model.AssignEffectiveImmunogenicity = aml.Constraint(rule=lambda model: sum(
            model.PositionImmunogenicity[p] * model.McRecoveredEpitopesFrequency[p]
            for p in model.EpitopePositions
        ) == model.EffectiveImmunogenicity)
