from __future__ import division, print_function

from abc import abstractmethod, ABCMeta
import multiprocessing as mp
import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition
from collections import namedtuple
import logging
from pcm import DoennesKohlbacherPcm


VaccineResult = namedtuple('VaccineResult', [
    'epitopes', 'spacers', 'sequence', 'immunogen', 'cleavage'
])


class VaccineConstraints:
    ''' base class for adding constraints and/or objectives to the milp model
    '''

    __metaclass__ = ABCMeta

    @abstractmethod
    def insert_constraints(self, model):
        ''' simply modify the model as appropriate
        '''
        pass


class MinimumCleavageGap(VaccineConstraints):
    ''' enforces a given minimum cleavage gap between the first position of an epitope
        (which indicates correct cleavage at the end of the preceding spacer)
        and the following amino acids inside the epitope. does not apply to the first epitope
    '''

    # TODO in the future: consider the _preceding_ amino acids instead
    # much harder since the spacer can have gaps...
    def __init__(self, min_gap):
        self.min_gap = min_gap

    def insert_constraints(self, model):
        model.MinCleavageGap = aml.Param(initialize=self.min_gap)
        model.MinCleavageGapConstraint = aml.Constraint(
            model.EpitopePositions * model.PositionInsideEpitope,
            rule=lambda model, epi, pos: (
                model.i[
                    epi * (model.SpacerLength + model.EpitopeLength)
                ] >= model.i[
                    epi * (model.SpacerLength + model.EpitopeLength) + pos
                ] + model.MinCleavageGap
            ) if epi > 0 and pos > 0 else aml.Constraint.Satisfied
        )


class StrobeSpacer:
    '''
    Joint optimization of string-of-beads vaccines with variable-length spacers 

    cleavage is based on a position-specific scoring matrix
    assumed of a fixed structure: one row per amino acid, and six columns, in order:
    cleavage at position -4, -3, -2, -1, 0, 1
    '''
    _solver_type = 'gurobi_persistent'

    def __init__(
        self,

        # list of all epitopes (strings)
        all_epitopes,

        # list of immunogenicities, one for each epitope
        epitope_immunogen,

        # maximum length of the spacers
        spacer_length,

        # number of epitopes in the vaccine
        vaccine_length,

        # constraint(s) to be applied to the vaccine
        vaccine_constraints,

        # object containing the pcm matrix, or None for default
        pcm=None,
    ):

        if spacer_length <= 0:
            raise ValueError('empty/negative maximum spacer length not supported')
        elif len(set(len(e) for e in all_epitopes)) != 1:
            raise ValueError('only epitopes of the same length are supported')
        elif len(all_epitopes[0]) <= 4:
            raise ValueError('only epitopes longer than four amino acids are supported')

        self._pcm = pcm or DoennesKohlbacherPcm()
        self._epitope_immunogen = epitope_immunogen
        self._spacer_length = spacer_length
        self._vaccine_length = vaccine_length
        self._all_epitopes = self._pcm.encode_sequences(all_epitopes)
        self._epitope_length = len(all_epitopes[0])
        self._pcm_matrix = self._fetch_pcm_matrix()
        self._logger = logging.getLogger(self.__class__.__name__)
        self._vaccine_constraints = [vaccine_constraints] if not hasattr(
            vaccine_constraints, '__iter__') else vaccine_constraints
        self._built = False

    def _fetch_pcm_matrix(self):
        ''' validates and scrambles the pcm matrix in the format used by the ilp
            i.e. score at the cleavage point and in the following position,
            then cleavage from four positions before onwards.
            makes indexing very intuitive, as e.g. -3 indicates three positions before cleavage
        '''

        mat = self._pcm.get_pcm_matrix()
        assert len(mat) == 20 and all(len(r) == 6 for r in mat)
        return [[amino_pcm[i] for i in [4, 5, 0, 1, 2, 3]] for amino_pcm in mat]

    def build_model(self):
        if self._built:
            return

        self._logger.info('Building model...')
        self._model = aml.ConcreteModel()

        self._logger.debug('Building model objects and parameters...')
        self._model.Epitopes = aml.RangeSet(0, len(self._epitope_immunogen) - 1)
        self._model.Aminoacids = aml.RangeSet(0, len(self._pcm_matrix) - 1)
        self._model.EpitopePositions = aml.RangeSet(0, self._vaccine_length - 1)
        self._model.SpacerPositions = aml.RangeSet(0, self._vaccine_length - 2)
        self._model.AminoacidPositions = aml.RangeSet(0, self._spacer_length - 1)
        self._model.PcmIdx = aml.RangeSet(-4, 1)
        sequence_length = (
            self._vaccine_length * self._epitope_length +
            (self._vaccine_length - 1) * self._spacer_length - 1
        )
        self._model.SequencePositions = aml.RangeSet(0, sequence_length)
        self._model.PositionInsideEpitope = aml.RangeSet(0, self._epitope_length - 1)

        self._model.SpacerLength = aml.Param(initialize=self._spacer_length)
        self._model.VaccineLength = aml.Param(initialize=self._vaccine_length)
        self._model.EpitopeLength = aml.Param(initialize=self._epitope_length)
        self._model.PssmMatrix = aml.Param(self._model.Aminoacids * self._model.PcmIdx,
                                           initialize=lambda model, i, j: self._pcm_matrix[i][j])
        self._model.EpitopeImmunogen = aml.Param(self._model.Epitopes,
                                                 initialize=lambda model, i: self._epitope_immunogen[i])
        self._model.EpitopeSequences = aml.Param(
            self._model.Epitopes * self._model.PositionInsideEpitope,
            initialize=lambda model, i, j: self._all_epitopes[i][j]
        )
        self._model.SequenceLength = aml.Param(
            initialize=self._vaccine_length * (self._epitope_length + self._spacer_length) - self._spacer_length
        )

        self._logger.debug('Building model variables...')

        # x(ij) = 1 iff epitope i is in position j
        self._model.x = aml.Var(
            self._model.Epitopes * self._model.EpitopePositions, domain=aml.Binary, initialize=0)

        # y(ijk) = 1 iff aminoacid k is in position j of spacer i
        self._model.y = aml.Var(self._model.SpacerPositions * self._model.AminoacidPositions * self._model.Aminoacids,
                                domain=aml.Binary, initialize=0)

        # a(ij) = 1 iff aminoacid j is in position i of the *whole* sequence (epitopes + spacers)
        self._model.a = aml.Var(self._model.SequencePositions * self._model.Aminoacids, domain=aml.Binary, initialize=0)

        # c(i) indicates whether there is an aminoacid at position i
        self._model.c = aml.Var(
            self._model.SequencePositions, domain=aml.Binary, initialize=0)

        # o(ij) counts how many aminoacids are selected between position i and position j
        # o(ij) < 0 <=> j < i and 0 when i = j. it counts the amino acid in position j, not the one in position i
        # nb: we need to index on aminoacids too, read doc for p to know why
        self._model.o = aml.Var(self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
                                bounds=(-sequence_length, sequence_length))

        # p(ijk) has the content of the pssm matrix when the aminoacid in position i is k, and the offset is o[j]
        # or zero if j is out of bounds
        # p is assigned from o with a piecewise linear function, and for this to work they must have the same index
        # that is why we have to include the aminoacids in the indexing for o
        self._model.p = aml.Var(
            self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
            bounds=(
                min(x for row in self._pcm_matrix for x in row),
                max(x for row in self._pcm_matrix for x in row),
            )
        )

        # i(i) is the computed immunogenicity at position i
        self._model.i = aml.Var(self._model.SequencePositions, domain=aml.Reals)

        self._logger.debug('Building basic constraints...')

        # exactly one epitope per position
        self._model.OneEpitopePerPosition = aml.Constraint(
            self._model.EpitopePositions,
            rule=lambda model, position: sum(
                model.x[i, position] for i in model.Epitopes
            ) == 1
        )

        # each epitope can be used at most once
        self._model.OnePositionPerEpitope = aml.Constraint(
            self._model.Epitopes,
            rule=lambda model, epitope: sum(
                model.x[epitope, i] for i in model.EpitopePositions
            ) <= 1
        )

        # at most one aminoacid per position per spacer
        self._model.OneAminoacidSelected = aml.Constraint(
            self._model.SpacerPositions * self._model.AminoacidPositions,
            rule=lambda model, position, spacerpos: sum(
                model.y[position, spacerpos, i] for i in model.Aminoacids
            ) <= 1
        )

        self._logger.debug('Building sequence constraints...')

        # fill in the sequence, in two steps
        # 1. ensure that a(ij) = 1 if aminoacid j is in position i
        def sequence_positive_rule(model, seq_pos, amino):
            segment = seq_pos // (model.EpitopeLength + model.SpacerLength)
            offset = seq_pos % (model.EpitopeLength + model.SpacerLength)

            if offset < model.EpitopeLength:
                return model.a[seq_pos, amino] >= sum(
                    model.x[epi, segment] if model.EpitopeSequences[epi, offset] == amino else 0
                    for epi in model.Epitopes
                )
            else:
                return model.a[seq_pos, amino] >= model.y[segment, offset - model.EpitopeLength, amino]

        self._model.SequencePositive = aml.Constraint(
            self._model.SequencePositions * self._model.Aminoacids, rule=sequence_positive_rule
        )

        # 2. ensure that a(ij) = 0 if aminoacid j is not in position i
        def sequence_negative_rule(model, seq_pos):
            segment = seq_pos // (model.EpitopeLength + model.SpacerLength)
            offset = seq_pos % (model.EpitopeLength + model.SpacerLength)

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

        self._model.SequenceNegative = aml.Constraint(
            self._model.SequencePositions, rule=sequence_negative_rule
        )

        # compute coverage indicators
        self._model.SequenceIndicators = aml.Constraint(
            self._model.SequencePositions, rule=lambda model, pos: model.c[pos] == sum(
                model.a[pos, a] for a in model.Aminoacids
            )
        )

        self._logger.debug('Building offset constraints...')

        # compute offsets
        def offsets_rule(model, dst, src, amino):
            # aminoacid at src is counted, aminoacid at dst is not counted
            if src < dst:
                return model.o[dst, src, amino] == -sum(model.c[p] for p in range(src, dst))
            elif src > dst:
                return model.o[dst, src, amino] == sum(model.c[p] for p in range(dst + 1, src + 1))
            else:
                return model.o[dst, src, amino] == 0

        self._model.Offsets = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids, rule=offsets_rule
        )

        def index_rule(model, dst, src, amino, breakp):
            if -4 <= breakp < 2:
                return model.PssmMatrix[amino, breakp]
            else:
                return 0

        self._logger.debug('Building array access constraints...')

        self._model.AssignP = aml.Piecewise(
            self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
            self._model.p, self._model.o,
            pw_pts=[-sequence_length, -5] + range(-4, 2) + [2, sequence_length],
            pw_constr_type='EQ',
            pw_repn='MC',
            f_rule=index_rule,
            warning_tol=-1.0,
        )

        self._logger.debug('Building cleavage position interactions constraints...')

        # b(ijk) = 1 if the aminoacid in position j is k and should be used to compute cleavage for position i
        self._model.b = aml.Var(self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
                                domain=aml.Binary)
        self._model.AssignB = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
            rule=lambda model, pos, i, k: model.b[pos, i, k] == sum(
                model.a[i, k] * model.a[pos, l] for l in model.Aminoacids
            )
        )

        # compute cleavage for each position
        self._model.ComputeCleavage = aml.Constraint(
            self._model.SequencePositions,
            rule=lambda model, pos: model.i[pos] == sum(
                model.b[pos, i, k] * model.p[pos, i, k]
                for i in model.SequencePositions
                for k in model.Aminoacids
            )
        )

        self._logger.debug('Building custom vaccine constraints...')
        for constr in self._vaccine_constraints:
            constr.insert_constraints(self._model)

        self._logger.debug('Building immunogenicity objective...')

        # store immunogenicity in a variable
        self._model.Immunogenicity = aml.Var()
        self._model.AssignImmunogenicity = aml.Constraint(rule=lambda model: model.Immunogenicity == sum(
            model.x[i, j] * model.EpitopeImmunogen[i] for i in model.Epitopes for j in model.EpitopePositions
        ))

        self._model.Objective = aml.Objective(rule=lambda model: model.Immunogenicity, sense=aml.maximize)

        self._solver = SolverFactory(self._solver_type)
        self._solver.set_instance(self._model)

        self._built = False
        self._logger.info('Model built successfully!')

        return self

    def solve(self, options=None, tee=1):
        if not self._built:
            self.build_model()

        res = self._solver.solve(
            options=options or {'Threads': mp.cpu_count()}, tee=tee, save_results=False, report_timing=True)
        if res.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError('Could not solve problem - %s . Please check your settings' % res.Solution.status)
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
            [
                self._pcm.get_amino(k)
                for j in self._model.AminoacidPositions
                for k in self._model.Aminoacids
                if aml.value(self._model.y[i, j, k]) > 0.9
            ]
            for i in self._model.SpacerPositions
        ]

        sequence = ''.join(
            self._pcm.get_amino(a)
            for i in self._model.SequencePositions
            for a in [
                k for k in self._model.Aminoacids
                if aml.value(self._model.a[i, k]) > 0.9
            ]
        )

        cleavage = [aml.value(self._model.i[i]) for i in self._model.SequencePositions]

        return VaccineResult(epitopes, spacers, sequence, aml.value(self._model.Immunogenicity), cleavage)
