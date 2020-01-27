import math

from spacers.model import VaccineConstraint
import pyomo.environ as aml


class MonteCarloVaccineConstraint(VaccineConstraint):
    '''
    base class for constraints applied to the Monte Carlo model
    '''
