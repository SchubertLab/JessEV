import math
from abc import ABC, abstractmethod

import pyomo.environ as aml

from spacers.model import VaccineConstraint


class MaximumCleavageInsideEpitopes(VaccineConstraint):
    '''
    enforces a given maximum cleavage inside the epitopes
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


class MinimumNTerminusCleavage(VaccineConstraint):
    '''
    enforces a given minimum cleavage at the first position of an epitope
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


class MinimumCTerminusCleavage(VaccineConstraint):
    '''
    enforces a minimum cleavage score at the first position of every spacer
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


class MinimumCoverageAverageConservation(VaccineConstraint):
    '''
    enforces minimum coverage and/or average epitope conservation
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

        # for every option, indicates which epitopes cover it
        # FIXME terrible for caching as we iterate over columns
        cs['Coverage'] = aml.Param(cs['Options'], initialize=lambda _, o: set(
            i for i, epi_cov in enumerate(self._epitope_coverage) if epi_cov[o]
        ))

        if self._min_coverage is not None:
            cs['IsOptionCovered'] = aml.Var(cs['Options'], domain=aml.Binary, initialize=0)
            cs['AssignIsOptionCovered'] = aml.Constraint(
                cs['Options'], rule=lambda model, option: sum(
                    model.x[e, p]
                    for e in cs['Coverage'][option]
                    for p in model.EpitopePositions
                ) >= cs['IsOptionCovered'][option]
            )

            cs['MinCoverage'] = aml.Param(initialize=(
                int(self._min_coverage) if self._min_coverage > 1
                else math.ceil(self._min_coverage * len(self._epitope_coverage[0]))
            ))

            cs['MinCoverageConstraint'] = aml.Constraint(rule=lambda model: sum(
                cs['IsOptionCovered'][o] for o in cs['Options']
            ) >= cs['MinCoverage'])

        if self._min_conservation is not None:
            cs['MinConservation'] = aml.Param(initialize=(
                int(self._min_conservation) if self._min_conservation > 1
                else math.ceil(self._min_conservation * len(self._epitope_coverage[0]))
            ))

            cs['MinConservationConstraint'] = aml.Constraint(rule=lambda model: sum(
                model.x[e, p]
                for o in cs['Options']
                for e in cs['Coverage'][o]
                for p in model.EpitopePositions
            ) >= cs['MinConservation'] * model.VaccineLength)

        for k, v in cs.items():
            name = '%s_%s' % (self._name, k)
            setattr(model, name, v)


class DualImplementationConstraints(VaccineConstraint, ABC):
    '''
    base class for constraints whose implementation is different between
    fixed and variable spacer length
    '''

    def insert_constraint(self, model):
        if model.MinSpacerLength == model.MaxSpacerLength:
            constr = self._get_fixed_spacer_length_constraints()
            typ = 'fixed'
        else:
            constr = self._get_variable_spacer_length_constraints()
            typ = 'variable'

        if constr is None:
            raise RuntimeError('these constraints do not support a model with %s length spacers', typ)
        else:
            constr.insert_constraint(model)

    @abstractmethod
    def _get_fixed_spacer_length_constraints(self) -> VaccineConstraint:
        '''
        return the implementation responsible for inserting constraints to the model with fixed length spacers
        '''

    @abstractmethod
    def _get_variable_spacer_length_constraints(self) -> VaccineConstraint:
        '''
        return the implementation responsible for inserting constraints to the model with variable length spacers
        '''


class MinimumNTerminusCleavageGap(DualImplementationConstraints):
    '''
    enforces a given minimum cleavage gap between the first position of an epitope
    (which indicates correct cleavage at the end of the preceding spacer)
    and the cleavage of surrounding amino acids (next one and previous four)
    '''

    def __init__(self, min_gap):
        self._min_gap = min_gap

    def _get_variable_spacer_length_constraints(self):
        return VariableLengthMinimumNTerminusCleavageGap(self._min_gap)

    def _get_fixed_spacer_length_constraints(self):
        return None  # TODO


class VariableLengthMinimumNTerminusCleavageGap(VaccineConstraint):
    '''
    enforces a given minimum cleavage gap between the first position of an epitope
    (which indicates correct cleavage at the end of the preceding spacer)
    and the cleavage of surrounding amino acids (next one and previous four)
    '''

    def __init__(self, min_gap):
        self._min_gap = min_gap

    @staticmethod
    def _assign_mask(model, epi, off):
        if epi > 0:
            epi_start = epi * (model.MaxSpacerLength + model.EpitopeLength)
            return model.NTerminusMask[epi, off] == model.d[epi_start, off] * model.g[epi_start, off]
        else:
            return aml.Constraint.Satisfied

    @staticmethod
    def _constraint_rule(model, epi, offs):
        epi_start = epi * (model.MaxSpacerLength + model.EpitopeLength)
        if epi > 0 and offs != 0:
            # remove a large constant when c[pos] = 0 to satisfy the constraint
            # when this position is empty
            return (
                model.i[epi_start] * model.NTerminusMask[epi, offs]
            ) >= (
                model.i[epi_start + offs] + model.MinCleavageGap
            ) * model.NTerminusMask[epi, offs] - 50 * (1 - model.c[epi_start + offs])
        else:
            return aml.Constraint.Satisfied

    def insert_constraint(self, model):
        model.MinCleavageGap = aml.Param(initialize=self._min_gap)

        model.NTerminusMask = aml.Var(
            model.EpitopePositions * model.OffsetAround,
            domain=aml.Binary, initialize=0
        )
        model.AssignNTerminusMask = aml.Constraint(
            model.EpitopePositions * model.OffsetAround, rule=self._assign_mask
        )
        model.MinCleavageGapConstraint = aml.Constraint(
            model.EpitopePositions * model.OffsetAround, rule=self._constraint_rule
        )


class BoundCleavageInsideSpacers(DualImplementationConstraints):
    '''
    enforces a given minimum/maximum cleavage likelihood inside every spacer
    use None to disable the corresponding constraint
    '''

    def __init__(self, min_cleavage, max_cleavage):
        self._min_cleavage = min_cleavage
        self._max_cleavage = max_cleavage

    def _get_variable_spacer_length_constraints(self):
        return VariableLengthBoundCleavageInsideSpacers(self._min_cleavage, self._max_cleavage)

    def _get_fixed_spacer_length_constraints(self):
        return None  # TODO


class VariableLengthBoundCleavageInsideSpacers(VaccineConstraint):
    '''
    enforces a given minimum/maximum cleavage likelihood inside every spacer
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
