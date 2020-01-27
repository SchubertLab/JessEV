import logging
import csv
import multiprocessing as mp
import sys
from collections import namedtuple
from typing import List
from abc import ABC, abstractmethod

import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition

from abc import ABC, abstractmethod
from spacers import utilities
from spacers.pcm import DoennesKohlbacherPcm


VaccineResult = namedtuple('VaccineResult', [
    'epitopes', 'spacers', 'sequence', 'immunogen', 'cleavage'
])


class VaccineResult:
    def __init__(self, epitopes, spacers, sequence, immunogen, cleavage):
        self.epitopes = epitopes
        self.spacers = spacers
        self.sequence = sequence
        self.immunogen = immunogen
        self.cleavage = cleavage

    def to_csv(self, file_name):
        with open(file_name, 'w') as f:
            writer = csv.DictWriter(f, ('immunogen', 'vaccine', 'spacers', 'cleavage'))
            writer.writeheader()
            writer.writerow({
                'immunogen': self.immunogen,
                'vaccine': self.sequence,
                'spacers': ';'.join(self.spacers),
                'cleavage': ';'.join('%.3f' % c for c in self.cleavage)
            })


class VaccineObjective(ABC):
    '''
    base class for the milp objective
    '''

    @abstractmethod
    def insert_objective(self, model):
        '''
        insert the objective in the model
        '''


class VaccineConstraint(ABC):
    '''
    base class for adding constraints to the milp model
    '''

    @abstractmethod
    def insert_constraint(self, model):
        '''
        simply modify the model as appropriate
        '''


class SolverFailedException(Exception):
    def __init__(self, condition):
        self.condition = condition


class ModelParams:
    def __init__(
        self,

        # list of all epitopes (strings)
        all_epitopes,

        # list of immunogenicities, one for each epitope
        epitope_immunogen,

        # maximum length of the spacers
        max_spacer_length,

        # number of epitopes in the vaccine
        vaccine_length,

        # object containing the pcm matrix, or None for default
        pcm=None,
    ):
        if max_spacer_length <= 0:
            raise ValueError('empty/negative maximum spacer length not supported')
        elif len(set(len(e) for e in all_epitopes)) != 1:
            raise ValueError('only epitopes of the same length are supported')
        elif len(all_epitopes[0]) <= 4:
            raise ValueError('only epitopes longer than four amino acids are supported')

        self.pcm = pcm or DoennesKohlbacherPcm()
        self.max_spacer_length = max_spacer_length
        self.vaccine_length = vaccine_length

        self.all_epitopes, self.epitope_immunogen = [], []
        for epi, imm in zip(all_epitopes, epitope_immunogen):
            try:
                self.all_epitopes.append([self.pcm.get_index(a) for a in epi])
            except KeyError:
                continue
            else:
                self.epitope_immunogen.append(imm)

        self.epitope_length = len(self.all_epitopes[0])
        self.pcm_matrix = self._fetch_pcm_matrix()
        self.built = False

        # TODO check and repair (if possible) inconsistencies between constraints
        # for example, max epitope cleavage & min nterminus cleavage
        # or max spacer cleavage & min cterminus cleavage

    def _fetch_pcm_matrix(self):
        '''
        validates and scrambles the pcm matrix in the format used by the ilp
        i.e. score at the cleavage point and in the following position,
        then cleavage from four positions before onwards.
        makes indexing very intuitive, as e.g. -3 indicates three positions before cleavage
        '''

        mat = self.pcm.get_pcm_matrix()
        assert len(mat) == 20 and all(len(r) == 6 for r in mat)
        return [[amino_pcm[i] for i in [4, 5, 0, 1, 2, 3]] for amino_pcm in mat]


class ModelImplementation(ABC):
    '''
    base class for the actual linear program
    '''

    def __init__(
            self,
            params: ModelParams,
            vaccine_constraints: List[VaccineConstraint],
            vaccine_objective: VaccineObjective
    ):
        self._params = params
        self._constraints = vaccine_constraints
        self._objective = vaccine_objective

    @abstractmethod
    def build_model(self) -> None:
        pass

    @abstractmethod
    def solve(self, params: ModelParams, **kwargs) -> VaccineResult:
        pass


class StrobeSpacer:
    def __init__(self, model_implementation: ModelImplementation):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._implementation = model_implementation
        self._built = False

    def build_model(self):
        if self._built:
            return

        self._logger.info('Building model...')
        self._implementation.build_model()
        self._built = True

        return self

    def solve(self, **kwargs):
        # if logging is configured, gurobipy will print messages there *and* on stdout
        # so we silence its logger and redirect all stdout to our own logger
        logging.getLogger('gurobipy.gurobipy').disabled = True

        class LoggingStdOut:
            def __init__(self):
                self.logger = logging.getLogger('stdout')

            def write(self, message):
                self.logger.debug(message.rstrip())

            def flush(self, *args, **kwargs):
                pass

        sys.stdout = LoggingStdOut()

        try:
            return self._solve(**kwargs)
        except Exception:
            # restore stdout so that handlers can print normally
            # https://docs.python.org/3/library/sys.html#sys.__stdout__
            sys.stdout = sys.__stdout__
            raise
        finally:
            sys.stdout = sys.__stdout__

    def _solve(self, **kwargs):
        if not self._built:
            self.build_model()

        self._logger.info('Solving model...')
        res = self._implementation.solve(**kwargs)
        self._logger.info('Solved successfully!')

        return res


def insert_indicator_sum_beyond_threshold(model, name, indices, larger_than_is,
                                          get_variables_bounds_fn, default=None):
    '''
    creates a variable of the given name and defined over the given indices (or uses the existing one)
    that indicates when the sum of certain variables is larger (larger_than_is=1) or smaller (larger_than_is=0)
    than given threshold, or zero in the opposite case

    get_variables_bound_fn takes as input the model and the indices, and must return a tuple
    containing (in order):
        - an iterable of variables to sum over
        - the upper bound for the sum (not necessarily tight)
        - the threshold above which the indicator is one
    '''
    if isinstance(name, aml.Var):
        result_var, indices = name, name.index_set()
        name = result_var.name
    elif not hasattr(model, name):
        result_var = aml.Var(indices, domain=aml.Binary, initialize=0)
        setattr(model, name, result_var)
    else:
        result_var = getattr(model, name)
        indices = result_var.index_set()

    def positive_rule(model, *idx):
        variables, upper_bound, threshold = get_variables_bounds_fn(model, *idx)
        if not variables:
            return result_var[idx] == default
        elif larger_than_is > 0:
            return sum(variables) - upper_bound * result_var[idx] <= threshold
        else:
            return sum(variables) - upper_bound * (1 - result_var[idx]) <= threshold

    def negative_rule(model, *idx):
        variables, upper_bound, threshold = get_variables_bounds_fn(model, *idx)
        if not variables:
            return result_var[idx] == default
        elif larger_than_is > 0:
            return sum(variables) + upper_bound * (1 - result_var[idx]) >= threshold
        else:
            return sum(variables) + upper_bound * result_var[idx] >= threshold

    setattr(model, name + 'SetPositive', aml.Constraint(indices, rule=positive_rule))
    setattr(model, name + 'SetNegative', aml.Constraint(indices, rule=negative_rule))


def insert_conjunction_constraints(model, name, indices, get_conjunction_vars_fn, default=1):
    ''' creates constraints that assign one to a variable if the conjunction of
        some other variables is true, i.e. y = x1 and x2 and ... and xn
    '''

    def get_vars_and_bounds(model, *idx):
        variables = get_conjunction_vars_fn(model, *idx)
        upper_bound = len(variables) + 1
        threshold = len(variables) - 0.5
        return variables, upper_bound, threshold

    insert_indicator_sum_beyond_threshold(model, name, indices, 1, get_vars_and_bounds, default)


def insert_disjunction_constraints(model, name, indices, get_conjunction_vars_fn, default=1):
    ''' creates constraints that assign one to a variable if the disjunction of
        some other variables is true, i.e. y = x1 or x2 or ... or xn
    '''

    def get_vars_and_bounds(model, *idx):
        variables = get_conjunction_vars_fn(model, *idx)
        upper_bound = len(variables) + 1
        threshold = 0.5
        return variables, upper_bound, threshold

    insert_indicator_sum_beyond_threshold(model, name, indices, 1, get_vars_and_bounds, default)
