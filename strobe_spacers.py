from __future__ import division, print_function

import pyomo.environ as aml
import pyomo.kernel as pmo
from pyomo.opt import SolverFactory, TerminationCondition
from collections import namedtuple


VaccineResult = namedtuple('VaccineResult', ['epitopes', 'spacers', 'sequence', 'immunogen', 'cleavage'])


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
                 # [p1', p2', p4, p3, p2, p1] makes indexing much nicer: r[-i] = pi and r[i] = pi'
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
        assert len(set(len(e) for e in all_epitopes)) == 1, 'only epitopes of the same length are supported'

        # not just because I am lazy, a scenario with both epitopes and spacers of one residue could give some headaches
        assert len(all_epitopes[0]) > 4, 'only epitopes longer of four amino acids are supported'
        self._epitope_length = len(all_epitopes[0])

        assert all(len(r) == 6 for r in pssm_matrix)
        self._pssm_matrix = pssm_matrix
        self._num_aminoacids = len(pssm_matrix)
        self._min_pre_junction_cleavage = min_pre_junction_cleavage
        self._min_post_junction_cleavage = min_post_junction_cleavage
        self._max_epitope_cleavage = max_epitope_cleavage

    @staticmethod
    def compute_cleavage_objective(model):
        return sum(
            # cleavage for each aminoacid of each spacer
            model.y[i, j, k] * (
                # correct cleavage between epitopes and spacer
                2 * (model.PssmMatrix[k, 0] if j == 0 else 0) +
                2 * (model.PssmMatrix[k, 1] if j == 1 else 0) +
                (model.PssmMatrix[k, -1] if j == model.SpacerLength - 1 else 0) +
                (model.PssmMatrix[k, -2] if j == model.SpacerLength - 2 else 0) +
                (model.PssmMatrix[k, -3] if j == model.SpacerLength - 3 else 0) +
                (model.PssmMatrix[k, -4] if j == model.SpacerLength - 4 else 0) + 0

                # cleavage inside the spacer
                + sum(
                    model.PssmMatrix[k, l] if 0 <= j - l < model.SpacerLength else 0
                    for l in model.PcmIdx
                ) / model.SpacerLength

                # incorrect cleavage inside the preceding epitope (only the part involving the spacer)
                - (model.PssmMatrix[k, 1] if j == 0 else 0)

                # incorrect cleavage inside the following epitope (only the part involving the spacer)
                - sum(
                    model.PssmMatrix[k, l] if model.SpacerLength < j - l < model.SpacerLength + 4 else 0
                    for l in model.PcmIdx
                )
            )
            for i in model.SpacerPositions
            for j in model.AminoacidPositions
            for k in model.Aminoacids
        ) - sum(
            # incorrect cleavage inside epitopes (only the part not involving the spacer)
            model.x[i, j] * (
                (model.EpitopeCleavageBefore[i] if j > 0 else 0) +
                (model.EpitopeCleavageAfter[i] if j < model.VaccineLength - 1 else 0)
            )
            for i in model.Epitopes
            for j in model.EpitopePositions
        )

    @staticmethod
    def compute_pre_junction_cleavage(model, spacer):
        return sum(
            # effect of previous epitope
            model.x[epi, spacer] * sum(
                model.PssmMatrix[model.EpitopeSequences[epi, model.EpitopeLength + j - 1], j]
                for j in range(-4, 0)
            ) +

            # effect of next epitope
            model.x[epi, spacer + 1] * sum(
                model.PssmMatrix[model.EpitopeSequences[epi, j - model.SpacerLength], j]
                for j in range(model.SpacerLength, 2)
            )
            for epi in model.Epitopes
        ) + sum(
            # effect of spacer
            model.y[spacer, i, k] * model.PssmMatrix[k, i]
            for i in range(0, min(2, model.SpacerLength))
            for k in model.Aminoacids
        )

    @staticmethod
    def compute_post_junction_cleavage(model, spacer):
        return sum(
            # effect of spacer
            model.y[spacer, model.SpacerLength - i - 1, k] * model.PssmMatrix[k, -i - 1]
            for i in range(0, min(4, model.SpacerLength))
            for k in model.Aminoacids
        ) + sum(
            # effect of next epitope
            model.x[epi, spacer + 1] * sum(
                model.PssmMatrix[model.EpitopeSequences[epi, i], i]
                for i in range(2)
            )

            # effect of previous epitope
            + model.x[epi, spacer] * sum(
                model.PssmMatrix[
                    model.EpitopeSequences[epi, model.EpitopeLength + model.SpacerLength - i - 1],
                    -i - 1
                ]
                for i in range(model.SpacerLength, 4)
            )
            for epi in model.Epitopes
        )

    @staticmethod
    def compute_max_epitope_cleavage(model, pos):
        return sum(  # effect of the inside of the epitope
            model.x[epi, pos] * (
                model.PssmMatrix[model.EpitopeSequences[epi, i + j], j] if 0 <= i + j < model.EpitopeLength else 0
                for i in range(1, model.EpitopeLength)
                for j in range(-4, 2)
            )
            for epi in model.Epitopes
        ) + sum(
            # effect of the previous spacer
            model.y[pos - 1, model.SpacerLength - i - 1, k] * model.PssmMatrix[k, -i - 1]
            for i in range(0, min(4, model.SpacerLength))
            for k in model.Aminoacids
        ) + sum(
            # effect of the next spacer
            model.y[pos, i, k] * model.PssmMatrix[k, i]
            for i in range(0, min(2, model.SpacerLength))
            for k in model.Aminoacids
        ) + sum(
            # effect of the previous epitope
            sum(
                model.x[epi, pos - 1] * model.PssmMatrix[
                    model.EpitopeSequences[epi, model.EpitopeLength + model.SpacerLength - i - 1],
                    -i - 1
                ] if pos > 0 else 0
                for i in range(model.SpacerLength, 4)
            )

            # effect of the next epitope
            + sum(
                model.x[epi, pos + 1] * model.PssmMatrix[
                    model.EpitopeSequences[epi, i - model.SpacerLength]
                ] if pos < model.VaccineLength - 1 else 0
                for i in range(model.SpacerLength, 2)
            )
            for epi in model.Epitopes
        )

    def build_model(self):
        self._model = aml.ConcreteModel()

        self._model.Epitopes = aml.RangeSet(0, len(self._epitope_immunogen) - 1)
        self._model.Aminoacids = aml.RangeSet(0, len(self._pssm_matrix) - 1)
        self._model.EpitopePositions = aml.RangeSet(0, self._vaccine_length - 1)
        self._model.SpacerPositions = aml.RangeSet(0, self._vaccine_length - 2)
        self._model.AminoacidPositions = aml.RangeSet(0, self._spacer_length - 1)
        self._model.PcmIdx = aml.RangeSet(-4, 1)
        self._model.SequencePositions = aml.RangeSet(
            0,  self._vaccine_length * self._epitope_length + (self._vaccine_length - 1) * self._spacer_length - 1
        )
        self._model.PositionInsideEpitope = aml.RangeSet(0, self._epitope_length - 1)

        self._model.MaxCleavage = aml.Param(initialize=self._min_cleavage, mutable=True)
        self._model.SpacerLength = aml.Param(initialize=self._spacer_length)
        self._model.VaccineLength = aml.Param(initialize=self._vaccine_length)
        self._model.EpitopeLength = aml.Param(initialize=self._epitope_length)
        self._model.PssmMatrix = aml.Param(self._model.Aminoacids * self._model.PcmIdx,
                                           initialize=lambda model, i, j: self._pssm_matrix[i][j])
        self._model.EpitopeImmunogen = aml.Param(self._model.Epitopes,
                                                 initialize=lambda model, i: self._epitope_immunogen[i])
        self._model.EpitopeCleavageBefore = aml.Param(
            self._model.Epitopes, initialize=lambda model, i: self._epitope_cleavage_before[i]
        )
        self._model.EpitopeCleavageAfter = aml.Param(
            self._model.Epitopes, initialize=lambda model, i: self._epitope_cleavage_after[i]
        )
        self._model.EpitopeSequences = aml.Param(
            self._model.Epitopes * self._model.PositionInsideEpitope,
            initialize=lambda model, i, j: self._all_epitopes[i][j]
        )
        self._model.SequenceLength = aml.Param(
            initialize=self._vaccine_length * (self._epitope_length + self._spacer_length) - self._spacer_length
        )
        self._model.MinPreJunctionCleavage = aml.Param(initialize=self._min_pre_junction_cleavage)
        self._model.MinPostJunctionCleavage = aml.Param(initialize=self._min_post_junction_cleavage)
        self._model.MaxEpitopeCleavage = aml.Param(initialize=self._max_epitope_cleavage)

        # a(ij) = 1 iff aminoacid j is in position i of the *whole* sequence (epitopes + spacers)
        # self._model.a = aml.Var(self._model.SequencePositions * self._model.Aminoacids, domain=aml.Binary, initialize=0)

        # c(i) is the computed immunogenicity at position i
        # self._model.c = aml.Var(self._model.SequencePositions, domain=aml.Reals)

        # x(ij) = 1 iff epitope i is in position j
        self._model.x = aml.Var(self._model.Epitopes * self._model.EpitopePositions, domain=aml.Binary, initialize=0)

        # y(ijk) = 1 iff aminoacid k is in position j of spacer i
        self._model.y = aml.Var(self._model.SpacerPositions * self._model.AminoacidPositions * self._model.Aminoacids,
                                domain=aml.Binary, initialize=0)

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

        # exactly one aminoacid per position per spacer
        self._model.OneAminoacidSelected = aml.Constraint(
            self._model.SpacerPositions * self._model.AminoacidPositions,
            rule=lambda model, position, spacerpos: sum(
                model.y[position, spacerpos, i] for i in model.Aminoacids
            ) == 1
        )

        # # fill in the sequence based on the chosen epitopes
        # self._model.AssignSequenceFromEpitopes = aml.Constraint(
        #     self._model.Epitopes * self._model.EpitopePositions * self._model.PositionInsideEpitope,
        #     rule=lambda model, epitope, epi_pos, pos_inside_epi: model.a[
        #         epi_pos * (model.EpitopeLength + model.SpacerLength) + pos_inside_epi,
        #         model.EpitopeSequences[epitope, pos_inside_epi]
        #     ] >= model.x[epitope, epi_pos]
        # )

        # # fill in the sequence based on the chosen spacers
        # self._model.AssignSequenceFromSpacers = aml.Constraint(
        #     self._model.SpacerPositions * self._model.AminoacidPositions * self._model.Aminoacids,
        #     rule=lambda model, spacer_pos, pos_inside_spacer, amino: model.a[
        #         (spacer_pos + 1) * (model.EpitopeLength + model.SpacerLength) - model.SpacerLength + pos_inside_spacer,
        #         amino
        #     ] >= model.y[spacer_pos, pos_inside_spacer, amino]
        # )

        # # ensure only one aminoacid is filled for each position
        # self._model.UniqueSequence = aml.Constraint(
        #     self._model.SequencePositions,
        #     rule=lambda model, pos: sum(model.a[pos, k] for k in model.Aminoacids) <= 1
        # )

        # def assign_sequence_rule(model, seq_pos, amino):
        #     segment = seq_pos // (model.EpitopeLength + model.SpacerLength)
        #     offset = seq_pos % (model.EpitopeLength + model.SpacerLength)

        #     if offset < model.EpitopeLength:
        #         return model.a[seq_pos, amino] >= sum(
        #             model.x[epi, segment] if model.EpitopeSequences[epi, offset] == amino else 0
        #             for epi in model.Epitopes
        #         )
        #     else:
        #         return model.a[seq_pos, amino] >= model.y[segment, offset - model.EpitopeLength, amino]

        # self._model.AssignSequence = aml.Constraint(
        #     self._model.SequencePositions * self._model.Aminoacids, rule=assign_sequence_rule
        # )

        # # compute cleavage for each position
        # self._model.ComputeCleavage = aml.Constraint(
        #     self._model.SequencePositions,
        #     rule=lambda model, pos: sum(
        #         model.a[pos - j, k] * model.PssmMatrix[k, j] if 0 <= pos - j < model.SequenceLength else 0
        #         for j in range(-4, 2)
        #         for k in model.Aminoacids
        #     ) == model.c[pos]
        # )

        # # make sure cleavage at the C-terminal junction is larger than cleavage inside the epitope
        # self._model.CleavageGap = aml.Constraint(
        #     self._model.SpacerPositions * self._model.PositionInsideEpitope,
        #     rule=lambda model, spacer, pos_inside_epitope: model.c[
        #         (spacer + 1) * (model.EpitopeLength + model.SpacerLength) - 1
        #     ] >= model.c[
        #         (spacer + 1) * (model.EpitopeLength + model.SpacerLength) + pos_inside_epitope
        #     ] + model.MinClevageGap
        # )

        # enforce minimum pre-junctions cleavage
        self._model.MinPreJunctionCleavageConstraint = aml.Constraint(
            self._model.SpacerPositions,
            rule=lambda model, spacer: self.compute_pre_junction_cleavage(model, spacer) >= model.MinPreJunctionCleavage
        )

        # enforce minimum post-junction cleavage
        self._model.MinPostJunctionCleavageConstraint = aml.Constraint(
            self._model.SpacerPositions,
            rule=lambda model, spacer: self.compute_post_junction_cleavage(
                model, spacer) >= model.MinPostJunctionCleavage
        )

        # enforce maximum epitope cleavage
        self._model.MaxEpitopeCleavageConstraint = aml.Constraint(
            self._model.EpitopePositions,
            rule=lambda model, pos: self.compute_max_epitope_cleavage(model, pos) >= model.MaxEpitopeCleavage
        )

        # store immunogenicity in a variable
        self._model.Immunogenicity = aml.Var()
        self._model.AssignImmunogenicity = aml.Constraint(rule=lambda model: model.Immunogenicity == sum(
            model.x[i, j] * model.EpitopeImmunogen[i] for i in model.Epitopes for j in model.EpitopePositions
        ))

        # store total cleavage in a variable
        self._model.Cleavage = aml.Var()
        self._model.AssignCleavage = aml.Constraint(
            rule=lambda model: model.Cleavage == self.compute_cleavage_objective
        )

        if self._min_cleavage is not None:
            self._model.MaxCleavageConstraint = aml.Constraint(
                rule=lambda model: model.Cleavage <= model.MaxCleavage
            )
            def objective_rule(model): return model.Immunogenicity
        else:
            def objective_rule(model): return model.Immunogenicity + model.Cleavage

        self._model.Objective = aml.Objective(rule=objective_rule, sense=aml.maximize)

        self._solver = SolverFactory(self._solver_type)
        self._solver.set_instance(self._model)

        return self

    def set_objective(self, immunogen, cleavage):
        if immunogen and cleavage:
            def rule(model): return model.Immunogenicity + model.Cleavage
        elif immunogen:
            def rule(model): return model.Immunogenicity
        elif cleavage:
            def rule(model): return model.Cleavage
        else:
            raise ValueError('at least one of immunogenicity and cleavage must be in the objective')

        del self._model.Objective
        self._model.Objective = aml.Objective(rule=rule, sense=aml.maximize)
        self._solver.set_objective(self._model.Objective)

    def set_max_cleavage(self, cleavage):
        if hasattr(self._model, 'MaxCleavageConstraint'):
            constr = self._model.MaxCleavageConstraint
            self._solver.remove_constraint(constr)
            del constr

        if cleavage is not None:
            self.set_objective(immunogen=True, cleavage=False)
            self._model.MaxCleavage.set_value(cleavage)
            self._model.MaxCleavageConstraint = pmo.constraint(expr=self._model.Cleavage <= self._model.MaxCleavage)
            self._solver.add_constraint(self._model.MaxCleavageConstraint)

    def solve(self, options=None, tee=1):
        res = self._solver.solve(options=options or {}, tee=tee, save_results=False, report_timing=True)
        if res.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError('Could not solve problem - %s . Please check your settings' % res.Solution.status)
        self._solver.load_vars()

        epitopes = [
            j
            for i in self._model.EpitopePositions
            for j in self._model.Epitopes
            if self._model.x[j, i].value > 0.9
        ]

        spacers = [
            [
                k
                for j in self._model.AminoacidPositions
                for k in self._model.Aminoacids
                if self._model.y[i, j, k].value > 0.9
            ]
            for i in self._model.SpacerPositions
        ]

        sequence = [
            [k for k in range(self._num_aminoacids) if self._model.a[i, k].value > 0.9][0]
            for i in range(self._vaccine_length * (self._epitope_length + self._spacer_length) - self._spacer_length)
        ]

        return VaccineResult(epitopes, spacers, sequence, self._model.Immunogenicity.value, self._model.Cleavage.value)
