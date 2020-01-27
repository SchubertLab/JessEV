import logging
from typing import List
import multiprocessing as mp
import sys
from collections import namedtuple

import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition

from spacers import utilities
from spacers.pcm import DoennesKohlbacherPcm
from spacers.model import VaccineResult, SolverFailedException, ModelParams, ModelImplementation
from spacers.monte_carlo.objectives import MonteCarloVaccineObjective
from spacers.model import insert_indicator_sum_beyond_threshold, VaccineConstraint


class MonteCarloRecovery(ModelImplementation):
    '''
    Joint optimization of string-of-beads vaccines and spacers
    estimating recovery probabilities using Monte Carlo trials
    '''
    _solver_type = 'gurobi_persistent'

    def __init__(
            self,
            params: ModelParams,
            vaccine_constraints: List[VaccineConstraint],
            vaccine_objective: MonteCarloVaccineObjective
    ):
        super(MonteCarloRecovery, self).__init__(params, vaccine_constraints, vaccine_objective)
        self._model = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def build_model(self) -> None:
        model = aml.ConcreteModel()

        self._logger.debug('Building model objects and parameters...')
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

        self._logger.debug('Building model variables...')

        # x(ij) = 1 iff epitope i is in position j
        model.x = aml.Var(model.Epitopes * model.EpitopePositions, domain=aml.Binary, initialize=0)

        # y(ijk) = 1 iff aminoacid k is in position j of spacer i
        model.y = aml.Var(model.SpacerPositions * model.AminoacidPositions * model.Aminoacids,
                                domain=aml.Binary, initialize=0)

        # a(ij) = 1 iff aminoacid j is in position i of the *whole* sequence (epitopes + spacers)
        model.a = aml.Var(model.SequencePositions * model.Aminoacids, domain=aml.Binary, initialize=0)

        # c(i) indicates whether there is an aminoacid at position i
        model.c = aml.Var(model.SequencePositions, domain=aml.Binary, initialize=0)

        # o(ij) counts how many aminoacids are selected between position i and position j
        # o(ij) < 0 <=> j < i and 0 when i = j. it counts the amino acid in position j, not the one in position i
        model.o = aml.Var(model.SequencePositions * model.SequencePositions,
                                bounds=(-model.SequenceLength, model.SequenceLength))

        # decision variables used to linearize access to the pcm matrix
        # l(ijk) = 1 if o(ij)=k, d(jk)=1 if a(j)=k
        # l0(ij) = 1 if o(ij) is out of bounds, similarly for d0(j)
        model.l = aml.Var(model.SequencePositions * model.SequencePositions * model.PcmIdx,
                                domain=aml.Binary)
        model.l0 = aml.Var(model.SequencePositions * model.SequencePositions, domain=aml.Binary)

        # these variables are used to decide whether an offset is within the bounds of the pcm indices
        # and to force the corresponding lambda variable to be 1
        # d(ij) = 1 if o(ij) >= -4 and g(ij) = 1 if o(ij) < 2
        model.d = aml.Var(model.SequencePositions * model.SequencePositions, domain=aml.Binary)
        model.g = aml.Var(model.SequencePositions * model.SequencePositions, domain=aml.Binary)

        # p(ijk) has the content of the pssm matrix when the aminoacid in position i is k, and the offset is o[j]
        # or zero if j is out of bounds
        model.p = aml.Var(model.SequencePositions * model.SequencePositions, bounds=(
            min(x for row in self._params.pcm_matrix for x in row),
            max(x for row in self._params.pcm_matrix for x in row),
        ))

        # i(i) is the computed cleavage at position i
        model.i = aml.Var(model.SequencePositions, domain=aml.Reals, initialize=0)

        self._logger.debug('Building basic constraints...')
        self._insert_basic_constraints(model)

        self._logger.debug('Building sequence constraints...')
        self._insert_sequence_constraints(model)

        self._logger.debug('Building array access constraints...')
        self._insert_cleavage_constraints(model)

        self._logger.debug('Building custom vaccine constraints...')
        for constr in self._constraints:
            constr.insert_constraint(model)

        self._logger.debug('Building immunogenicity objective...')
        self._objective_variable = self._objective.insert_objective(model)

        self._model = model
        self._solver = SolverFactory(self._solver_type)
        self._solver.set_instance(model)

        self._built = True
        self._logger.info('Model built successfully!')

        return self

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
    def _insert_cleavage_constraints(model):
        # compute coverage indicators
        model.SequenceIndicators = aml.Constraint(
            model.SequencePositions, rule=lambda model, pos: model.c[pos] == sum(
                model.a[pos, a] for a in model.Aminoacids
            )
        )

        # compute offsets
        def offsets_rule(model, dst, src):
            # aminoacid at src is counted, aminoacid at dst is not counted
            if src < dst:
                return model.o[dst, src] == -sum(model.c[p] for p in range(src, dst))
            elif src > dst:
                return model.o[dst, src] == sum(model.c[p] for p in range(dst + 1, src + 1))
            else:
                return model.o[dst, src] == 0

        model.Offsets = aml.Constraint(
            model.SequencePositions * model.SequencePositions, rule=offsets_rule
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
            model.SequencePositions * model.SequencePositions,
            rule=lambda model, p1, p2: sum(
                model.l[p1, p2, k] for k in model.PcmIdx
            ) == model.d[p1, p2] * model.g[p1, p2]
        )

        # and to choose lambda0 when d = 0 or g = 0
        model.LambdaOrLambdaZero = aml.Constraint(
            model.SequencePositions * model.SequencePositions,
            rule=lambda model, p1, p2: model.l0[p1, p2] == 1 - model.d[p1, p2] * model.g[p1, p2]
        )

        # now select the lambda corresponding to the offset if necessary
        model.ReconstructOffset = aml.Constraint(
            model.SequencePositions * model.SequencePositions,
            rule=lambda model, p1, p2: sum(
                model.l[p1, p2, i] * i for i in model.PcmIdx
            ) + model.o[p1, p2] * model.l0[p1, p2] == model.o[p1, p2]
        )

        # read cleavage value from the pcm matrix
        model.AssignP = aml.Constraint(
            model.SequencePositions * model.SequencePositions,
            rule=lambda model, p1, p2: model.p[p1, p2] == sum(
                model.PssmMatrix[k, i] * model.a[p2, k] * model.l[p1, p2, i]
                for i in model.PcmIdx
                for k in model.Aminoacids
            )
        )

        # compute cleavage for each position
        model.ComputeCleavage = aml.Constraint(
            model.SequencePositions,
            rule=lambda model, pos: model.i[pos] == model.c[pos] * sum(
                model.p[pos, j] for j in model.SequencePositions
            )
        )

    def solve(self, options=None, tee=1):
        res = self._solver.solve(
            options=options or {'Threads': mp.cpu_count()},
            tee=tee, save_results=False, report_timing=True
        )
        if res.solver.termination_condition != TerminationCondition.optimal:
            raise SolverFailedException(res.Solution.status)
        self._solver.load_vars()

        return self._read_solution_from_model()

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
            if aml.value(self._model.c[i]) > 0.9
        ]

        return VaccineResult(
            epitopes, spacers, sequence,
            aml.value(self._objective_variable),
            cleavage
        )
