from __future__ import division, print_function

import multiprocessing as mp
import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition
from collections import namedtuple


VaccineResult = namedtuple(
    'VaccineResult', ['epitopes', 'spacers', 'sequence', 'immunogen'])


class StrobeSpacer:
    _solver_type = 'gurobi_persistent'

    def __init__(self,
                 # immunogenicity for each epitope
                 epitope_immunogen,

                 # penalty for having epitope i before/after a spacer (*subtracted* from the total score)
                 epitope_cleavage_before, epitope_cleavage_after,

                 # length of the spacers
                 spacer_length,

                 # minimum cleavage to obtain
                 min_cleavage,

                 # number of epitopes in the vaccine
                 vaccine_length,

                 # matrix[i, j] = cleavage likelihood of aminoacid i in position j
                 # each row starts with the cleavages after the aminoacid, then the cleavages before
                 # [p1', p2', p4, p3, p2, p1]
                 pssm_matrix,

                 # list of all epitopes; each epitope is a list of indices of each amino acid
                 all_epitopes,

                 # minimum cleavage at spacer-epitope junctions
                 min_pre_junction_cleavage,
                 min_post_junction_cleavage,

                 # maximum cleavage inside the chosen epitopes
                 max_epitope_cleavage,
                 ):
        assert spacer_length > 0, 'empty spacers are not supported'

        self._epitope_immunogen = epitope_immunogen
        self._epitope_cleavage_before = epitope_cleavage_before
        self._epitope_cleavage_after = epitope_cleavage_after
        self._spacer_length = spacer_length
        self._min_cleavage = min_cleavage
        self._vaccine_length = vaccine_length
        self._all_epitopes = all_epitopes
        assert len(set(len(e) for e in all_epitopes)
                   ) == 1, 'only epitopes of the same length are supported'

        # not just because I am lazy, a scenario with both epitopes and spacers of one residue could give some headaches
        assert len(
            all_epitopes[0]) > 4, 'only epitopes longer than four amino acids are supported'
        self._epitope_length = len(all_epitopes[0])

        assert all(len(r) == 6 for r in pssm_matrix)
        self._pssm_matrix = pssm_matrix
        self._num_aminoacids = len(pssm_matrix)
        self._min_pre_junction_cleavage = min_pre_junction_cleavage
        self._min_post_junction_cleavage = min_post_junction_cleavage
        self._max_epitope_cleavage = max_epitope_cleavage

    def build_model(self):
        self._model = aml.ConcreteModel()

        self._model.Epitopes = aml.RangeSet(
            0, len(self._epitope_immunogen) - 1)
        self._model.Aminoacids = aml.RangeSet(0, len(self._pssm_matrix) - 1)
        self._model.EpitopePositions = aml.RangeSet(
            0, self._vaccine_length - 1)
        self._model.SpacerPositions = aml.RangeSet(0, self._vaccine_length - 2)
        self._model.AminoacidPositions = aml.RangeSet(
            0, self._spacer_length - 1)
        self._model.PcmIdx = aml.RangeSet(-4, 1)
        sequence_length = (
            self._vaccine_length * self._epitope_length +
            (self._vaccine_length - 1) * self._spacer_length - 1
        )
        self._model.SequencePositions = aml.RangeSet(0, sequence_length)
        self._model.PositionInsideEpitope = aml.RangeSet(
            0, self._epitope_length - 1)

        self._model.MaxCleavage = aml.Param(
            initialize=self._min_cleavage, mutable=True)
        self._model.SpacerLength = aml.Param(initialize=self._spacer_length)
        self._model.VaccineLength = aml.Param(initialize=self._vaccine_length)
        self._model.EpitopeLength = aml.Param(initialize=self._epitope_length)
        self._model.PssmMatrix = aml.Param(self._model.Aminoacids * self._model.PcmIdx,
                                           initialize=lambda model, i, j: self._pssm_matrix[i][j])
        self._model.EpitopeImmunogen = aml.Param(self._model.Epitopes,
                                                 initialize=lambda model, i: self._epitope_immunogen[i])
        self._model.EpitopeCleavageBefore = aml.Param(
            self._model.Epitopes, initialize=lambda model, i: self._epitope_cleavage_before[
                i]
        )
        self._model.EpitopeCleavageAfter = aml.Param(
            self._model.Epitopes, initialize=lambda model, i: self._epitope_cleavage_after[i]
        )
        self._model.EpitopeSequences = aml.Param(
            self._model.Epitopes * self._model.PositionInsideEpitope,
            initialize=lambda model, i, j: self._all_epitopes[i][j]
        )
        self._model.SequenceLength = aml.Param(
            initialize=self._vaccine_length *
            (self._epitope_length + self._spacer_length) - self._spacer_length
        )
        self._model.MinPreJunctionCleavage = aml.Param(
            initialize=self._min_pre_junction_cleavage)
        self._model.MinPostJunctionCleavage = aml.Param(
            initialize=self._min_post_junction_cleavage)
        self._model.MaxEpitopeCleavage = aml.Param(
            initialize=self._max_epitope_cleavage)
        self._model.MinCleavageGap = aml.Param(initialize=0.1)

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
        # FIXME fix bounds from pssm matrix?
        self._model.p = aml.Var(self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
                                bounds=(-10, 10))

        # i(i) is the computed immunogenicity at position i
        self._model.i = aml.Var(self._model.SequencePositions, domain=aml.Reals)

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

        self._model.AssignP = aml.Piecewise(
            self._model.SequencePositions * self._model.SequencePositions * self._model.Aminoacids,
            self._model.p, self._model.o,
            pw_pts=[-sequence_length, -5] + range(-4, 2) + [2, sequence_length],
            pw_constr_type='EQ',
            f_rule=index_rule,
            warning_tol=-1.0,
        )

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

        # make sure cleavage at the C-terminal junction is larger than cleavage inside the epitope
        self._model.CleavageGap = aml.Constraint(
            self._model.SpacerPositions * self._model.PositionInsideEpitope,
            rule=lambda model, spacer, pos_inside_epitope: model.i[
                (spacer + 1) * (model.EpitopeLength + model.SpacerLength) - 1
            ] >= model.i[
                (spacer + 1) * (model.EpitopeLength + model.SpacerLength) + pos_inside_epitope
            ] + model.MinCleavageGap
        )

        # store immunogenicity in a variable
        self._model.Immunogenicity = aml.Var()
        self._model.AssignImmunogenicity = aml.Constraint(rule=lambda model: model.Immunogenicity == sum(
            model.x[i, j] * model.EpitopeImmunogen[i] for i in model.Epitopes for j in model.EpitopePositions
        ))

        self._model.Objective = aml.Objective(
            rule=lambda model: model.Immunogenicity, sense=aml.maximize)

        self._solver = SolverFactory(self._solver_type)
        self._solver.set_instance(self._model)

        return self

    def solve(self, options=None, tee=1):
        res = self._solver.solve(
            options=options or {'Threads': mp.cpu_count()}, tee=tee, save_results=False, report_timing=True)
        if res.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError(
                'Could not solve problem - %s . Please check your settings' % res.Solution.status)
        self._solver.load_vars()

        epitopes, spacers, sequence = self._read_solution_from_model(
            self._model)

        return VaccineResult(epitopes, spacers, sequence, self._model.Immunogenicity.value)

    @staticmethod
    def _read_solution_from_model(model):
        epitopes = [
            j
            for i in model.EpitopePositions
            for j in model.Epitopes
            if model.x[j, i].value > 0.9
        ]

        spacers = [
            [
                k
                for j in model.AminoacidPositions
                for k in model.Aminoacids
                if model.y[i, j, k].value > 0.9
            ]
            for i in model.SpacerPositions
        ]

        sequence = [
            [
                k
                for k in model.Aminoacids
                if model.a[i, k].value > 0.9
            ]
            for i in model.SequencePositions
        ]

        cleavage = [model.i[i].value for i in model.SequencePositions]
        coverage = [model.c[i].value for i in model.SequencePositions]
        offsets5 = [model.o[5, i, 0].value for i in model.SequencePositions]

        print(offsets5)
        print(cleavage)

        import ipdb
        ipdb.set_trace()

        return epitopes, spacers, sequence
