import math
from abc import ABC, abstractmethod
from spacers.model import VaccineConstraint

import pyomo.environ as aml


class VariableLengthVaccineConstraint(VaccineConstraint):
    '''
    base class for constraints that only apply to the variable length implementation
    '''
    pass


class MinimumNTerminusCleavageGap(VariableLengthVaccineConstraint):
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

    def insert_constraint(self, model):
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


class BoundCleavageInsideSpacers(VariableLengthVaccineConstraint):
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

    def insert_constraint(self, model):
        if self._min_cleavage is not None:
            model.MinSpacerCleavage = aml.Param(initialize=self._min_cleavage)
        if self._max_cleavage is not None:
            model.MaxSpacerCleavage = aml.Param(initialize=self._max_cleavage)

        model.BoundCleavageInsideSpacersConstraint = aml.Constraint(
            model.SpacerPositions * model.AminoacidPositions,
            rule=self._constraint_rule
        )


class MaximumCleavageInsideEpitopes(VariableLengthVaccineConstraint):
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

    def insert_constraint(self, model):
        model.MaxInnerEpitopeCleavage = aml.Param(initialize=self._max_cleavage)
        model.MaxInnerEpitopeCleavageConstraint = aml.Constraint(
            model.EpitopePositions * model.PositionInsideEpitope, rule=self._constraint_rule
        )


class MinimumNTerminusCleavage(VariableLengthVaccineConstraint):
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

    def insert_constraint(self, model):
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


class MinimumCTerminusCleavage(VariableLengthVaccineConstraint):
    ''' enforces a minimum cleavage score at the first position of every spacer
    '''

    def __init__(self, min_cleavage):
        self._min_cleavage = min_cleavage

    def insert_constraint(self, model):
        model.MinCtCleavage = aml.Param(initialize=self._min_cleavage)
        model.MinCtCleavageConstraint = aml.Constraint(
            model.SpacerPositions, rule=self._constraint_rule
        )

    @staticmethod
    def _constraint_rule(model, spacer):
        spacer_start = spacer * (model.MaxSpacerLength + model.EpitopeLength) + model.EpitopeLength
        return model.i[spacer_start] >= model.MinCtCleavage


class MinimumCoverageAverageConservation(VariableLengthVaccineConstraint):
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

    def insert_constraint(self, model):
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
                    sum(cs['Coverage'][e, o] for o in cs['Options']) - cs['MinConservation']
                )
                for e in model.Epitopes for p in model.EpitopePositions
            ) >= 0)

        for k, v in cs.items():
            name = '%s_%s' % (self._name, k)
            setattr(model, name, v)


class MinimumSpacerLength(VariableLengthVaccineConstraint):
    '''
    enforces a given minimum spacer length
    '''

    def __init__(self, min_spacer_length):
        if min_spacer_length < 0:
            raise ValueError('The minimum spacer length must be positive')

        self._min_spacer_length = min_spacer_length

    def insert_constraint(self, model):
        model.MinSpacerLength = aml.Param(initialize=self._min_spacer_length)

        # enforce minimum spacer length
        model.MinSpacerLengthConstraint = aml.Constraint(
            model.SpacerPositions, rule=lambda model, spacer: model.MinSpacerLength <= sum(
                model.c[(spacer + 1) * (model.EpitopeLength + model.MaxSpacerLength) - i]
                for i in range(1, model.MaxSpacerLength + 1)
            )
        )
