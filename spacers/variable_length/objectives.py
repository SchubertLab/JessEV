import math
import random
from abc import ABC, abstractmethod

import pyomo.environ as aml
from spacers.objectives import VaccineObjective

from spacers import utilities


class VariableLengthVaccineObjective(VaccineObjective):
    '''
    base class for objectives applied to the variable length model
    '''


class ImmunogenicityObjective(VariableLengthVaccineObjective):
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
