from __future__ import division, print_function

import sys
from abc import abstractmethod, ABC
import multiprocessing as mp
import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition
from collections import namedtuple
import logging
from pcm import DoennesKohlbacherPcm
import math


VaccineResult = namedtuple('VaccineResult', [
    'epitopes', 'spacers', 'sequence', 'immunogen', 'cleavage'
])


class SolverFailedException(Exception):
    def __init__(self, condition):
        self.condition = condition


class VaccineConstraints(ABC):
    ''' base class for adding constraints to the milp model
    '''

    @abstractmethod
    def insert_constraints(self, model):
        ''' simply modify the model as appropriate
        '''


class MinimumNTerminusCleavageGap(VaccineConstraints):
    ''' enforces a given minimum cleavage gap between the first position of an epitope
        (which indicates correct cleavage at the end of the preceding spacer)
        and the cleavage of surrounding amino acids (next one and previous four)
    '''

    def __init__(self, min_gap):
        self._min_gap = min_gap

    @staticmethod
    def _assign_mask(model, epi, pos):
        if epi > 0:
            epi_start = epi * (model.MaxSpacerLength + model.EpitopeLength)
            return model.NTerminusMask[epi, pos] == model.d[epi_start, pos] * model.g[epi_start, pos]
        else:
            return aml.Constraint.Satisfied

    @staticmethod
    def _constraint_rule(model, epi, pos):
        epi_start = epi * (model.MaxSpacerLength + model.EpitopeLength)
        if epi > 0 and pos != epi_start:
            # remove a large constant when c[pos] = 0 to satisfy the constraint
            # when this position is empty
            return (
                model.i[epi_start] * model.NTerminusMask[epi, pos]
            ) >= (
                model.i[pos] + model.MinCleavageGap
            ) * model.NTerminusMask[epi, pos] - 50 * (1 - model.c[pos])
        else:
            return aml.Constraint.Satisfied

    def insert_constraints(self, model):
        model.MinCleavageGap = aml.Param(initialize=self._min_gap)
        model.NTerminusMask = aml.Var(
            model.EpitopePositions * model.SequencePositions,
            domain=aml.Binary, initialize=0
        )
        model.AssignNTerminusMask = aml.Constraint(
            model.EpitopePositions * model.SequencePositions, rule=self._assign_mask
        )
        model.MinCleavageGapConstraint = aml.Constraint(
            model.EpitopePositions * model.SequencePositions, rule=self._constraint_rule
        )


class BoundCleavageInsideSpacers(VaccineConstraints):
    ''' enforces a given minimum/maximum cleavage likelihood inside every spacer
        use None to disable the corresponding constraint
    '''

    def __init__(self, min_cleavage, max_cleavage):
        self._min_cleavage = min_cleavage
        self._max_cleavage = max_cleavage

    def _constraint_rule(self, model, spacer, position):
        # absolute position in the sequence
        pos = (model.EpitopeLength + model.MaxSpacerLength) * spacer + position + model.EpitopeLength

        # I don't know why returning a tuple does not work
        if self._min_cleavage is not None and self._max_cleavage is not None:
            return (
                model.MinSpacerCleavage * model.c[pos]
                <= model.i[pos] <=
                model.MaxSpacerCleavage * model.c[pos]
            )
        elif self._min_cleavage is not None:
            return model.MinSpacerCleavage * model.c[pos] <= model.i[pos]
        elif self._max_cleavage is not None:
            return model.i[pos] <= model.MaxSpacerCleavage * model.c[pos]
        else:
            return aml.Constraint.Satisfied

    def insert_constraints(self, model):
        if self._min_cleavage is not None:
            model.MinSpacerCleavage = aml.Param(initialize=self._min_cleavage)
        if self._max_cleavage is not None:
            model.MaxSpacerCleavage = aml.Param(initialize=self._max_cleavage)

        model.BoundCleavageInsideSpacersConstraint = aml.Constraint(
            model.SpacerPositions * model.AminoacidPositions,
            rule=self._constraint_rule
        )


class MaximumCleavageInsideEpitopes(VaccineConstraints):
    ''' enforces a given maximum cleavage inside the epitopes
        possibly ignoring the first few amino acids

        nb: if this constraint is used together with `MinimumNTerminusCleavage`,
        you should ignore the first amino acid, as it corresponds to the
        n-terminus. if you don't do that, the problem is infeasible as the two
        constraints would contradict each other.

        this is indeed the default behavior, as we are very interested in
        cleavage at the n-terminus

    '''

    def __init__(self, max_cleavage, ignore_first=1):
        self._max_cleavage = max_cleavage
        self._ignore_first = ignore_first

    def _constraint_rule(self, model, epitope, offset):
        if offset >= self._ignore_first:
            pos = (model.EpitopeLength + model.MaxSpacerLength) * epitope + offset
            return model.i[pos] <= model.MaxInnerEpitopeCleavage
        else:
            return aml.Constraint.Satisfied

    def insert_constraints(self, model):
        model.MaxInnerEpitopeCleavage = aml.Param(initialize=self._max_cleavage)
        model.MaxInnerEpitopeCleavageConstraint = aml.Constraint(
            model.EpitopePositions * model.PositionInsideEpitope, rule=self._constraint_rule
        )


class MinimumNTerminusCleavage(VaccineConstraints):
    ''' enforces a given minimum cleavage at the first position of an epitope
        (which indicates correct cleavage at the end of the preceding spacer)

        nb: if this constraint is used together with
        `MaximumCleavageInsideEpitopes`, you should instruct that constraint to
        ignore the first amino acid, as it corresponds to the n-terminus. if
        you don't do that, the problem is infeasible as the two constraints
        would contradict each other
    '''

    def __init__(self, min_cleavage):
        self._min_cleavage = min_cleavage

    def insert_constraints(self, model):
        model.MinNtCleavage = aml.Param(initialize=self._min_cleavage)
        model.MinNtCleavageConstraint = aml.Constraint(
            model.EpitopePositions, rule=self._constraint_rule
        )

    @staticmethod
    def _constraint_rule(model, epi):
        epi_start = epi * (model.MaxSpacerLength + model.EpitopeLength)
        if epi > 0:
            return model.i[epi_start] >= model.MinNtCleavage
        else:
            return aml.Constraint.Satisfied


class MinimumCoverageAverageConservation(VaccineConstraints):
    ''' enforces minimum coverage and/or average epitope conservation
        with respect to a given set of options (i.e., which epitope covers which options)
    '''
    # there can be more instances of this type on the same problem
    # so we use a counter to provide unique default names
    counter = 0

    def __init__(self, epitope_coverage, min_coverage=None, min_conservation=None, name=None):
        self._min_coverage = min_coverage
        self._min_conservation = min_conservation
        self._epitope_coverage = epitope_coverage

        if name is None:
            MinimumCoverageAverageConservation.counter += 1
            self._name = 'CoverageConservation%d' % MinimumCoverageAverageConservation.counter
        else:
            self._name = name

    def insert_constraints(self, model):
        cs = {}  # we create components here, then insert them with the prefixed name

        cs['Options'] = aml.RangeSet(0, len(self._epitope_coverage[0]) - 1)
        cs['Coverage'] = aml.Param(model.Epitopes * cs['Options'],
                                   initialize=lambda _, e, o: self._epitope_coverage[e][o])

        if self._min_coverage is not None:
            cs['IsOptionCovered'] = aml.Var(cs['Options'], domain=aml.Binary, initialize=0)
            cs['AssignIsOptionCovered'] = aml.Constraint(
                cs['Options'], rule=lambda model, option: sum(
                    model.x[e, p] * cs['Coverage'][e, option]
                    for e in model.Epitopes for p in model.EpitopePositions
                ) >= cs['IsOptionCovered'][option]
            )

            cs['MinCoverage'] = aml.Param(initialize=(
                self._min_coverage if isinstance(self._min_coverage, int)
                else math.ceil(self._min_coverage * len(self._epitope_coverage[0]))
            ))

            cs['MinCoverageConstraint'] = aml.Constraint(rule=lambda model: sum(
                cs['IsOptionCovered'][o] for o in cs['Options']
            ) >= cs['MinCoverage'])

        if self._min_conservation is not None:
            cs['MinConservation'] = aml.Param(initialize=(
                self._min_conservation if isinstance(self._min_conservation, int)
                else math.ceil(self._min_conservation * len(self._epitope_coverage[0]))
            ))

            cs['MinConservationConstraint'] = aml.Constraint(rule=lambda model: sum(
                model.x[e, p] * (
                    sum(cs['Coverage'][e, o] for o in cs['Options'])
                    - cs['MinConservation']
                )
                for e in model.Epitopes for p in model.EpitopePositions
            ) >= 0)

        for k, v in cs.items():
            name = '%s_%s' % (self._name, k)
            setattr(model, name, v)


class StrobeSpacer:
    '''
    Joint optimization of string-of-beads vaccines with variable-length spacers

    cleavage is based on a position-specific scoring matrix
    assumed of a fixed structure: one row per amino acid, and six columns, in order:
    cleavage at relative position -4, -3, -2, -1, 0, 1
    '''
    _solver_type = 'gurobi_persistent'

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

        # constraint(s) to be applied to the vaccine
        vaccine_constraints,

        # object containing the pcm matrix, or None for default
        pcm=None,
    ):

        if max_spacer_length <= 0:
            raise ValueError('empty/negative maximum spacer length not supported')
        elif len(set(len(e) for e in all_epitopes)) != 1:
            raise ValueError('only epitopes of the same length are supported')
        elif len(all_epitopes[0]) <= 4:
            raise ValueError('only epitopes longer than four amino acids are supported')
        elif min_spacer_length > max_spacer_length:
            raise ValueError('minimum spacer length cannot be larger than the maximum spacer length')

        self._pcm = pcm or DoennesKohlbacherPcm()
        self._min_spacer_length = min_spacer_length
        self._max_spacer_length = max_spacer_length
        self._vaccine_length = vaccine_length

        self._all_epitopes, self._epitope_immunogen = [], []
        for epi, imm in zip(all_epitopes, epitope_immunogen):
            try:
                self._all_epitopes.append([self._pcm.get_index(a) for a in epi])
            except KeyError:
                continue
            else:
                self._epitope_immunogen.append(imm)

        self._epitope_length = len(self._all_epitopes[0])
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
        self._model.AminoacidPositions = aml.RangeSet(0, self._max_spacer_length - 1)
        self._model.PcmIdx = aml.RangeSet(-4, 1)
        sequence_length = (
            self._vaccine_length * self._epitope_length +
            (self._vaccine_length - 1) * self._max_spacer_length - 1
        )
        self._model.SequencePositions = aml.RangeSet(0, sequence_length)
        self._model.PositionInsideEpitope = aml.RangeSet(0, self._epitope_length - 1)

        self._model.MinSpacerLength = aml.Param(initialize=self._min_spacer_length)
        self._model.MaxSpacerLength = aml.Param(initialize=self._max_spacer_length)
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
            initialize=self._vaccine_length * (self._epitope_length + self._max_spacer_length) - self._max_spacer_length
        )

        self._logger.debug('Building model variables...')

        # x(ij) = 1 iff epitope i is in position j
        self._model.x = aml.Var(self._model.Epitopes * self._model.EpitopePositions, domain=aml.Binary, initialize=0)

        # y(ijk) = 1 iff aminoacid k is in position j of spacer i
        self._model.y = aml.Var(self._model.SpacerPositions * self._model.AminoacidPositions * self._model.Aminoacids,
                                domain=aml.Binary, initialize=0)

        # a(ij) = 1 iff aminoacid j is in position i of the *whole* sequence (epitopes + spacers)
        self._model.a = aml.Var(self._model.SequencePositions * self._model.Aminoacids, domain=aml.Binary, initialize=0)

        # c(i) indicates whether there is an aminoacid at position i
        self._model.c = aml.Var(self._model.SequencePositions, domain=aml.Binary, initialize=0)

        # o(ij) counts how many aminoacids are selected between position i and position j
        # o(ij) < 0 <=> j < i and 0 when i = j. it counts the amino acid in position j, not the one in position i
        self._model.o = aml.Var(self._model.SequencePositions * self._model.SequencePositions,
                                bounds=(-sequence_length, sequence_length))

        # decision variables used to linearize access to the pcm matrix
        # l(ijk) = 1 if o(ij)=k, d(jk)=1 if a(j)=k
        # l0(ij) = 1 if o(ij) is out of bounds, similarly for d0(j)
        self._model.l = aml.Var(self._model.SequencePositions * self._model.SequencePositions * self._model.PcmIdx,
                                domain=aml.Binary)
        self._model.l0 = aml.Var(self._model.SequencePositions * self._model.SequencePositions, domain=aml.Binary)

        # these variables are used to decide whether an offset is within the bounds of the pcm indices
        # and to force the corresponding lambda variable to be 1
        # d(ij) = 1 if o(ij) >= -4 and g(ij) = 1 if o(ij) < 2
        self._model.d = aml.Var(self._model.SequencePositions * self._model.SequencePositions, domain=aml.Binary)
        self._model.g = aml.Var(self._model.SequencePositions * self._model.SequencePositions, domain=aml.Binary)

        # p(ijk) has the content of the pssm matrix when the aminoacid in position i is k, and the offset is o[j]
        # or zero if j is out of bounds
        self._model.p = aml.Var(self._model.SequencePositions * self._model.SequencePositions, bounds=(
            min(x for row in self._pcm_matrix for x in row),
            max(x for row in self._pcm_matrix for x in row),
        ))

        # i(i) is the computed cleavage at position i
        self._model.i = aml.Var(self._model.SequencePositions, domain=aml.Reals, initialize=0)

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
            segment = seq_pos // (model.EpitopeLength + model.MaxSpacerLength)
            offset = seq_pos % (model.EpitopeLength + model.MaxSpacerLength)

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

        self._model.SequenceNegative = aml.Constraint(
            self._model.SequencePositions, rule=sequence_negative_rule
        )

        # compute coverage indicators
        self._model.SequenceIndicators = aml.Constraint(
            self._model.SequencePositions, rule=lambda model, pos: model.c[pos] == sum(
                model.a[pos, a] for a in model.Aminoacids
            )
        )

        # enforce minimum spacer length
        self._model.MinSpacerLengthConstraint = aml.Constraint(
            self._model.SpacerPositions, rule=lambda model, spacer: model.MinSpacerLength <= sum(
                model.c[(spacer + 1) * (model.EpitopeLength + model.MaxSpacerLength) - i]
                for i in range(1, model.MaxSpacerLength + 1)
            )
        )

        self._logger.debug('Building offset constraints...')

        # compute offsets
        def offsets_rule(model, dst, src):
            # aminoacid at src is counted, aminoacid at dst is not counted
            if src < dst:
                return model.o[dst, src] == -sum(model.c[p] for p in range(src, dst))
            elif src > dst:
                return model.o[dst, src] == sum(model.c[p] for p in range(dst + 1, src + 1))
            else:
                return model.o[dst, src] == 0

        self._model.Offsets = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions, rule=offsets_rule
        )

        self._logger.debug('Building array access constraints...')

        # set d = 1 when o >= -4
        # FIXME why is there a 5 and not a 4 !?!
        self._model.OffsetLowerBound = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions,
            rule=lambda model, p1, p2: -5 >= model.o[p1, p2] - (sequence_length + 5) * model.d[p1, p2]
        )

        # and set g = 1 when o <= 1
        # FIXME why is there a 2 and not a 1 !?!
        self._model.OffsetUpperBound = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions,
            rule=lambda model, p1, p2: 2 <= model.o[p1, p2] + (sequence_length + 2) * model.g[p1, p2]
        )

        # force the model to choose one lambda when d = g = 1
        self._model.ChooseOneLambda = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions,
            rule=lambda model, p1, p2: sum(
                model.l[p1, p2, k] for k in model.PcmIdx
            ) == model.d[p1, p2] * model.g[p1, p2]
        )

        # and to choose lambda0 when d = 0 or g = 0
        self._model.LambdaOrLambdaZero = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions,
            rule=lambda model, p1, p2: model.l0[p1, p2] == 1 - model.d[p1, p2] * model.g[p1, p2]
        )

        # now select the lambda corresponding to the offset if necessary
        self._model.ReconstructOffset = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions,
            rule=lambda model, p1, p2: sum(
                model.l[p1, p2, i] * i for i in model.PcmIdx
            ) + model.o[p1, p2] * model.l0[p1, p2] == model.o[p1, p2]
        )

        # read cleavage value from the pcm matrix
        self._model.AssignP = aml.Constraint(
            self._model.SequencePositions * self._model.SequencePositions,
            rule=lambda model, p1, p2: model.p[p1, p2] == sum(
                model.PssmMatrix[k, i] * model.a[p2, k] * model.l[p1, p2, i]
                for i in model.PcmIdx
                for k in model.Aminoacids
            )
        )

        # compute cleavage for each position
        self._model.ComputeCleavage = aml.Constraint(
            self._model.SequencePositions,
            rule=lambda model, pos: model.i[pos] == model.c[pos] * sum(
                model.p[pos, j] for j in model.SequencePositions
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

        self._built = True
        self._logger.info('Model built successfully!')

        return self

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

    def _solve(self, options=None, tee=1):
        if not self._built:
            self.build_model()

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
                self._pcm.get_amino(k)
                for j in self._model.AminoacidPositions
                for k in self._model.Aminoacids
                if aml.value(self._model.y[i, j, k]) > 0.9
            )
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

        cleavage = [
            aml.value(self._model.i[i])
            for i in self._model.SequencePositions
            if aml.value(self._model.c[i]) > 0.9
        ]

        return VaccineResult(epitopes, spacers, sequence, aml.value(self._model.Immunogenicity), cleavage)
