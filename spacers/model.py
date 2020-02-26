import csv
import logging
import multiprocessing as mp
import sys
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List

import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver

from spacers import utilities
from spacers.pcm import DoennesKohlbacherPcm


class VaccineResult:
    def __init__(self, epitopes, spacers, sequence, immunogen, cleavage):
        self.epitopes = epitopes
        self.spacers = spacers
        self.sequence = sequence
        self.immunogen = immunogen
        self.cleavage = cleavage

    def to_str_dict(self):
        return {
            'immunogen': self.immunogen,
            'vaccine': self.sequence,
            'spacers': ';'.join(self.spacers),
            'cleavage': ';'.join('%.3f' % c for c in self.cleavage)
        }

    def to_csv(self, file_name):
        with open(file_name, 'w') as f:
            writer = csv.DictWriter(f, ('immunogen', 'vaccine', 'spacers', 'cleavage'))
            writer.writeheader()
            writer.writerow(self.to_str_dict())


class ModelEditor:
    '''
    an utility class that allows constraints and variables to be added and removed
    so that users do not have to worry whether a persistent solver is used or not
    '''

    def __init__(self, model, solver, constraints, variables, raise_exceptions=True):
        self.model = model
        self.solver = solver
        self.constraints = constraints
        self.variables = variables
        self._raise = raise_exceptions

    def callback_for_each_item(self, items, callback, unpack_indexed):
        '''
        calls the given callback for each item in the list, e.g. constraints and variables.
        optionally "unpacks" indexed constraints/variables by iterating on each of their components.
        optionally swallows exceptions
        '''
        for c in items:
            if isinstance(c, str):
                c = getattr(self.model, c)

            if unpack_indexed and hasattr(c, 'is_indexed') and c.is_indexed():
                for v in c.values():
                    try:
                        callback(v)
                    except:
                        if self._raise:
                            raise
            else:
                try:
                    callback(c)
                except:
                    if self._raise:
                        raise

    def remove_constraints(self):
        if self.solver is not None and isinstance(self.solver, PersistentSolver):
            self.callback_for_each_item(self.constraints, self.solver.remove_constraint, unpack_indexed=True)

    def remove_variables(self):
        if self.solver is not None and isinstance(self.solver, PersistentSolver):
            self.callback_for_each_item(self.variables, self.solver.remove_var, unpack_indexed=True)

    def add_constraints(self):
        if self.solver is not None and isinstance(self.solver, PersistentSolver):
            self.callback_for_each_item(self.constraints, self.solver.add_constraint, unpack_indexed=True)

    def add_variables(self):
        if self.solver is not None and isinstance(self.solver, PersistentSolver):
            self.callback_for_each_item(self.variables, self.solver.add_var, unpack_indexed=True)

    def activate_constraints(self):
        self.callback_for_each_item(self.constraints, lambda c: c.activate(), unpack_indexed=False)

    def deactivate_constraints(self):
        self.callback_for_each_item(self.constraints, lambda c: c.deactivate(), unpack_indexed=False)

    def reconstruct_constraints(self):
        self.callback_for_each_item(self.constraints, lambda c: c.reconstruct(), unpack_indexed=False)

    def enable(self):
        '''
        activates constraints and adds constraints and variables to the solver
        '''
        self.add_variables()
        self.activate_constraints()
        self.add_constraints()

    def disable(self):
        '''
        removes variables and disables/removes constraints
        '''
        self.remove_constraints()
        self.deactivate_constraints()
        self.remove_variables()


class VaccineObjective(ABC):
    '''
    base class for the milp objective
    '''

    _constraint_names = []
    _variable_names = []
    _objective_variable = None

    def __init__(self):
        self._editor = None

    @abstractmethod
    def insert_objective(self, model, solver):
        '''
        insert the objective in the model. call super() at the end to automatically initialize
        a model editor and insert constraints into the persistent solver if necessary
        '''

        self._editor = ModelEditor(model, solver, self._constraint_names, self._variable_names)
        self._editor.enable()
        self._model = model
        self._solver = solver

    def activate(self):
        '''
        activate the objective
        '''
        self._editor.enable()
        self._editor.solver.set_objective(self._objective_variable)

    def deactivate(self):
        '''
        deactivate the objective
        '''
        self._editor.disable()

    @property
    def solver(self):
        return self._editor.solver

    @solver.setter
    def solver(self, solver):
        self._editor.solver = solver

    @property
    def model(self):
        return self._editor.model

    @model.setter
    def model(self, model):
        self._editor.model = model


class VaccineConstraint(ABC):
    '''
    base class for adding constraints to the milp model
    '''

    _constraint_names = []
    _variable_names = []

    @abstractmethod
    def insert_constraint(self, model, solver):
        '''
        insert the constraints in the model. call super() at the end to automatically initialize
        a model editor and insert constraints into the persistent solver if necessary
        '''

        self._editor = ModelEditor(model, solver, self._constraint_names, self._variable_names,
                                   raise_exceptions=False)
        self._editor.add_variables()
        self._editor.add_constraints()
        self._model = model
        self._solver = solver

    @abstractmethod
    def update(self, **kwargs):
        '''
        update the parameters of the given constraints. call super() at the end to automatically
        update the constraints in a persistent solver
        '''

        self._editor.remove_constraints()
        self._editor.reconstruct_constraints()
        self._editor.add_constraints()

    def activate(self):
        '''
        activate the constraints
        '''
        self._editor.enable()

    def deactivate(self):
        '''
        deactivate the constraints
        '''
        self._editor.disable()

    @property
    def solver(self):
        return self._editor.solver

    @solver.setter
    def solver(self, solver):
        self._editor.solver = solver

    @property
    def model(self):
        return self._editor.model

    @model.setter
    def model(self, model):
        self._editor.model = model


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

        # minimum length of the spacers
        min_spacer_length,

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
        self.min_spacer_length = min_spacer_length
        self.max_spacer_length = max_spacer_length
        self.vaccine_length = vaccine_length
        self.all_epitopes = [[self.pcm.get_index(a) for a in e] for e in all_epitopes]
        self.epitope_immunogen = epitope_immunogen
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


class StrobeSpacer:
    def __init__(
        self,
        params: ModelParams,
        vaccine_constraints: List[VaccineConstraint],
            vaccine_objective: VaccineObjective,
            solver_type='gurobi',
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._params = params
        self._constraints = vaccine_constraints
        self._objective = vaccine_objective
        self._solver_type = solver_type
        self._built = False
        self._solver = None
        self._model = None
        self._objective_variable = None

    @property
    def has_variable_length_spacers(self):
        return self._params.min_spacer_length != self._params.max_spacer_length

    def build_model(self):
        if self._built:
            return

        self._logger.info('Building model...')
        model = aml.ConcreteModel()

        # base linear program
        self._logger.debug('Building model objects...')
        self._build_model_objects(model)

        self._logger.debug('Building basic constraints...')
        self._insert_basic_constraints(model)

        self._logger.debug('Building sequence constraints...')
        self._insert_sequence_constraints(model)

        self._logger.debug('Building cleavage computation constraints...')
        if self._params.min_spacer_length == self._params.max_spacer_length:
            self._insert_cleavage_constraints_fixed_spacer_length(model)
        else:
            self._insert_cleavage_constraints_variable_spacer_length(self._params, model)

        # custom constraints and objective
        self._logger.debug('Building custom vaccine constraints...')
        for constr in self._constraints:
            constr.insert_constraint(model, None)

        self._logger.debug('Building immunogenicity objective...')
        self._objective_variable = self._objective.insert_objective(model, None)

        # create solver
        self._model = model
        self._solver = SolverFactory(self._solver_type)
        if isinstance(self._solver, PersistentSolver):
            self._solver.set_instance(self._model)

        # set solver in constraints and objectives
        for constr in self._constraints:
            constr.solver = self._solver
        self._objective.solver = self._solver

        self._built = True
        self._logger.info('Model built successfully!')

        self._log_model_size()

        return self

    def _build_model_objects(self, model):
        model.Epitopes = aml.RangeSet(0, len(self._params.epitope_immunogen) - 1)
        model.Aminoacids = aml.RangeSet(0, len(self._params.pcm_matrix) - 1)
        model.EpitopePositions = aml.RangeSet(0, self._params.vaccine_length - 1)
        model.SpacerPositions = aml.RangeSet(0, self._params.vaccine_length - 2)
        model.AminoacidPositions = aml.RangeSet(0, self._params.max_spacer_length - 1)
        model.PcmIdx = aml.RangeSet(-4, 1)
        model.SequenceLength = aml.Param(initialize=(
            self._params.vaccine_length * self._params.epitope_length +
            (self._params.vaccine_length - 1) * self._params.max_spacer_length - 1
        ))
        model.SequencePositions = aml.RangeSet(0, model.SequenceLength)
        model.PositionInsideEpitope = aml.RangeSet(0, self._params.epitope_length - 1)

        model.MinSpacerLength = aml.Param(initialize=self._params.min_spacer_length)
        model.MaxSpacerLength = aml.Param(initialize=self._params.max_spacer_length)
        model.VaccineLength = aml.Param(initialize=self._params.vaccine_length)
        model.EpitopeLength = aml.Param(initialize=self._params.epitope_length)
        model.PssmMatrix = aml.Param(model.Aminoacids * model.PcmIdx,
                                     initialize=lambda model, i, j: self._params.pcm_matrix[i][j])
        model.EpitopeImmunogen = aml.Param(model.Epitopes,
                                           initialize=lambda model, i: self._params.epitope_immunogen[i])
        model.EpitopeSequences = aml.Param(
            model.Epitopes * model.PositionInsideEpitope,
            initialize=lambda model, i, j: self._params.all_epitopes[i][j]
        )

        # x(ij) = 1 iff epitope i is in position j
        model.x = aml.Var(model.Epitopes * model.EpitopePositions, domain=aml.Binary, initialize=0)

        # y(ijk) = 1 iff aminoacid k is in position j of spacer i
        model.y = aml.Var(model.SpacerPositions * model.AminoacidPositions * model.Aminoacids,
                          domain=aml.Binary, initialize=0)

        # a(ij) = 1 iff aminoacid j is in position i of the *whole* sequence (epitopes + spacers)
        model.a = aml.Var(model.SequencePositions * model.Aminoacids, domain=aml.Binary, initialize=0)

        # i(i) is the computed cleavage at position i
        model.i = aml.Var(model.SequencePositions, domain=aml.Reals, initialize=0)

    @staticmethod
    def _insert_basic_constraints(model):
        # exactly one epitope per position
        model.OneEpitopePerPosition = aml.Constraint(
            model.EpitopePositions,
            rule=lambda model, position: sum(
                model.x[i, position] for i in model.Epitopes
            ) == 1
        )

        # each epitope can be used at most once
        model.OnePositionPerEpitope = aml.Constraint(
            model.Epitopes,
            rule=lambda model, epitope: sum(
                model.x[epitope, i] for i in model.EpitopePositions
            ) <= 1
        )

        # at most one aminoacid per position per spacer
        model.OneAminoacidSelected = aml.Constraint(
            model.SpacerPositions * model.AminoacidPositions,
            rule=lambda model, position, spacerpos: sum(
                model.y[position, spacerpos, i] for i in model.Aminoacids
            ) <= 1
        )

        # enforce minimum spacer length
        model.MinSpacerLengthConstraint = aml.Constraint(
            model.SpacerPositions, rule=lambda model, spacer: model.MinSpacerLength <= sum(
                model.y[spacer, p, a]
                for p in model.AminoacidPositions
                for a in model.Aminoacids
            )
        )

    @staticmethod
    def _insert_sequence_constraints(model):
        # fill in the sequence, in two steps
        # 1. ensure that a(ij) = 1 if aminoacid j is in position i
        def sequence_positive_rule(model, seq_pos, amino):
            segment = seq_pos // (model.EpitopeLength + model.MaxSpacerLength)
            offset = seq_pos % (model.EpitopeLength + model.MaxSpacerLength)

            if offset < model.EpitopeLength:
                return model.a[seq_pos, amino] >= sum(
                    model.x[epi, segment] if model.EpitopeSequences[epi, offset] == amino else 0
                    for epi in model.Epitopes
                )
            else:
                return model.a[seq_pos, amino] >= model.y[segment, offset - model.EpitopeLength, amino]

        model.SequencePositive = aml.Constraint(
            model.SequencePositions * model.Aminoacids, rule=sequence_positive_rule
        )

        # 2. ensure that a(ij) = 0 if aminoacid j is not in position i
        def sequence_negative_rule(model, seq_pos):
            segment = seq_pos // (model.EpitopeLength + model.MaxSpacerLength)
            offset = seq_pos % (model.EpitopeLength + model.MaxSpacerLength)

            if offset < model.EpitopeLength:
                return sum(
                    model.a[seq_pos, amino]
                    for amino in model.Aminoacids
                ) <= 1
            else:
                return sum(
                    model.a[seq_pos, amino]
                    for amino in model.Aminoacids
                ) <= sum(
                    model.y[segment, offset - model.EpitopeLength, amino]
                    for amino in model.Aminoacids
                )

        model.SequenceNegative = aml.Constraint(
            model.SequencePositions, rule=sequence_negative_rule
        )

    @staticmethod
    def _insert_cleavage_constraints_fixed_spacer_length(model):
        # compute cleavage for each position
        model.ComputeCleavage = aml.Constraint(
            model.SequencePositions,
            rule=lambda model, pos: model.i[pos] == sum(
                model.a[pos + j, a] * model.PssmMatrix[a, j]
                if pos + j > 0 and pos + j < model.SequenceLength else 0
                for j in range(-4, 2)
                for a in model.Aminoacids
            )
        )

    @staticmethod
    def _insert_cleavage_constraints_variable_spacer_length(params, model):

        # c(i) indicates whether there is an aminoacid at position i
        model.c = aml.Var(model.SequencePositions, domain=aml.Binary, initialize=0)

        # o(ij) counts how many amino acids are selected between position i (not counted) and i+j (counted)
        # where j goes from -4-max_spacer_length to 1+max_spacer_length
        # negative when j < 0 and 0 when j = 0
        model.OffsetAround = aml.RangeSet(-4 - model.MaxSpacerLength, model.MaxSpacerLength + 1)
        model.o = aml.Var(model.SequencePositions * model.OffsetAround,
                          bounds=(-4 - model.MaxSpacerLength, 1 + model.MaxSpacerLength))

        # decision variables used to linearize access to the pcm matrix
        # l(ijk) = 1 if o(ij)=k, d(jk)=1 if a(j)=k
        # l0(ij) = 1 if o(ij) is out of bounds, similarly for d0(j)
        model.l = aml.Var(model.SequencePositions * model.OffsetAround * model.PcmIdx, domain=aml.Binary)
        model.l0 = aml.Var(model.SequencePositions * model.OffsetAround, domain=aml.Binary)

        # these variables are used to decide whether an offset is within the bounds of the pcm indices
        # and to force the corresponding lambda variable to be 1
        # d(ij) = 1 if o(ij) >= -4 and g(ij) = 1 if o(ij) < 2
        model.d = aml.Var(model.SequencePositions * model.OffsetAround, domain=aml.Binary)
        model.g = aml.Var(model.SequencePositions * model.OffsetAround, domain=aml.Binary)

        # p(ijk) has the content of the pssm matrix when the aminoacid in position i is k, and the offset is o[j]
        # or zero if j is out of bounds
        # so-called "cleavage contributions"
        model.p = aml.Var(model.SequencePositions * model.OffsetAround, bounds=(
            min(x for row in params.pcm_matrix for x in row),
            max(x for row in params.pcm_matrix for x in row),
        ))

        # compute coverage indicators
        model.SequenceIndicators = aml.Constraint(
            model.SequencePositions, rule=lambda model, pos: model.c[pos] == sum(
                model.a[pos, a] for a in model.Aminoacids
            )
        )

        # compute offsets
        def offsets_rule(model, dst, offs):
            if offs < 0:
                return model.o[dst, offs] == -sum(
                    model.c[p] for p in range(max(0, dst + offs), dst)
                )
            elif offs > 0:
                return model.o[dst, offs] == sum(
                    model.c[p] for p in range(dst + 1, min(aml.value(model.SequenceLength), dst + offs) + 1)
                )
            else:
                return model.o[dst, offs] == 0

        model.ComputeOffsets = aml.Constraint(
            model.SequencePositions * model.OffsetAround, rule=offsets_rule
        )

        # set d = 1 when o >= -4
        insert_indicator_sum_beyond_threshold(
            model, model.d, None, larger_than_is=1,
            get_variables_bounds_fn=lambda model, p1, p2: (
                [model.o[p1, p2]], model.SequenceLength + 5, -4.5
            )
        )

        # and set g = 1 when o <= 1
        insert_indicator_sum_beyond_threshold(
            model, model.g, None, larger_than_is=0,
            get_variables_bounds_fn=lambda model, p1, p2: (
                [model.o[p1, p2]], model.SequenceLength + 2, 1.5
            )
        )

        # force the model to choose one lambda when d = g = 1
        model.ChooseOneLambda = aml.Constraint(
            model.SequencePositions * model.OffsetAround,
            rule=lambda model, p1, p2: sum(
                model.l[p1, p2, k] for k in model.PcmIdx
            ) == model.d[p1, p2] * model.g[p1, p2]
        )

        # and to choose lambda0 when d = 0 or g = 0
        model.LambdaOrLambdaZero = aml.Constraint(
            model.SequencePositions * model.OffsetAround,
            rule=lambda model, p1, p2: model.l0[p1, p2] == 1 - model.d[p1, p2] * model.g[p1, p2]
        )

        # now select the lambda corresponding to the offset if necessary
        model.ReconstructOffset = aml.Constraint(
            model.SequencePositions * model.OffsetAround,
            rule=lambda model, p1, p2: sum(
                model.l[p1, p2, i] * i for i in model.PcmIdx
            ) + model.o[p1, p2] * model.l0[p1, p2] == model.o[p1, p2]
        )

        # read cleavage value from the pcm matrix
        model.AssignP = aml.Constraint(
            model.SequencePositions * model.OffsetAround,
            rule=lambda model, p1, p2: model.p[p1, p2] == sum(
                model.PssmMatrix[k, i] * model.a[p1 + p2, k] * model.l[p1, p2, i]
                for i in model.PcmIdx
                for k in model.Aminoacids
                if 0 <= p1 + p2 <= aml.value(model.SequenceLength)
            )
        )

        # compute cleavage for each position
        model.ComputeCleavage = aml.Constraint(
            model.SequencePositions,
            rule=lambda model, pos: model.i[pos] == model.c[pos] * sum(
                model.p[pos, j] for j in model.OffsetAround
            )
        )

    def _log_model_size(self):
        '''
        print how many variables and constraints are in the linear program
        as inserted in pyomo (e.g., does not count linearization of quadratic constraints)
        '''
        all_vars = [
            (c.name, sum(1 for _ in c.items()))
            for c in self._model.component_objects(aml.Var)
        ]
        all_constrs = [
            (c.name, sum(1 for _ in c.items()))
            for c in self._model.component_objects(aml.Constraint)
        ]

        longest = max(len(c) for c, _ in all_vars + all_constrs)
        fmt = '    %%%ds : %%d' % longest

        self._logger.debug('Object count')
        self._logger.debug('  Variables:')
        for name, count in sorted(all_vars, key=lambda x: -x[1]):
            self._logger.debug(fmt, name, count)
        self._logger.debug(fmt, 'Total', sum(c for _, c in all_vars))

        self._logger.debug('  Constraints:')
        for name, count in sorted(all_constrs, key=lambda x: -x[1]):
            self._logger.debug(fmt, name, count)
        self._logger.debug(fmt, 'Total', sum(c for _, c in all_constrs))

    def solve(self, options=None, tee=1):
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
            return self._solve(options, tee)
        except Exception:
            # restore stdout so that handlers can print normally
            # https://docs.python.org/3/library/sys.html#sys.__stdout__
            sys.stdout = sys.__stdout__
            raise
        finally:
            sys.stdout = sys.__stdout__

    def _solve(self, options, tee):
        if not self._built:
            self.build_model()

        self._logger.info('Solving model...')
        res = self._solver.solve(
            self._model,
            options=options or {'Threads': mp.cpu_count(), 'NonConvex': 2, },  # 'MIPFocus': 3},
            tee=tee, report_timing=True
        )
        if res.solver.termination_condition != TerminationCondition.optimal:
            raise SolverFailedException(res.Solution.status)

        res = self._read_solution_from_model()

        self._logger.info('Solved successfully!')
        return res

    def _read_solution_from_model(self):
        epitopes = [
            j
            for i in self._model.EpitopePositions
            for j in self._model.Epitopes
            if aml.value(self._model.x[j, i]) > 0.9
        ]

        spacers = [
            ''.join(
                self._params.pcm.get_amino(k)
                for j in self._model.AminoacidPositions
                for k in self._model.Aminoacids
                if aml.value(self._model.y[i, j, k]) > 0.9
            )
            for i in self._model.SpacerPositions
        ]

        sequence = ''.join(
            self._params.pcm.get_amino(a)
            for i in self._model.SequencePositions
            for a in [
                k for k in self._model.Aminoacids
                if aml.value(self._model.a[i, k]) > 0.9
            ]
        )

        cleavage = [
            aml.value(self._model.i[i])
            for i in self._model.SequencePositions
            if not self.has_variable_length_spacers or aml.value(self._model.c[i]) > 0.9
        ]

        return VaccineResult(
            epitopes, spacers, sequence,
            aml.value(self._objective_variable),
            cleavage
        )

    def add_constraint(self, constr):
        constr.insert_constraint(self._model, self._solver)
        self._constraints.append(constr)

    def activate_constraint(self, constr_cls):
        for c in self._constraints:
            if isinstance(c, constr_cls):
                c.activate()

    def deactivate_constraint(self, constr_cls):
        for c in self._constraints:
            if isinstance(c, constr_cls):
                c.deactivate()

    def set_objective(self, objective):
        objective.insert_objective(self._model, self._solver)
        objective.activate()
        self._objective.deactivate()
        self._objective = objective


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
