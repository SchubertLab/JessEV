import math
import random
from abc import ABC, abstractmethod

import pyomo.environ as aml

from spacers.model import (ModelEditor, VaccineConstraint, VaccineObjective,
                           insert_conjunction_constraints,
                           insert_disjunction_constraints,
                           insert_indicator_sum_beyond_threshold)


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

    _constraint_names = ['MaxInnerEpitopeCleavageConstraint']

    def __init__(self, max_cleavage, ignore_first=1):
        self._max_cleavage = max_cleavage
        self._ignore_first = ignore_first

    def _constraint_rule(self, model, epitope, offset):
        if offset >= self._ignore_first:
            pos = (model.EpitopeLength + model.MaxSpacerLength) * epitope + offset
            return model.i[pos] <= model.MaxInnerEpitopeCleavage
        else:
            return aml.Constraint.Satisfied

    def insert_constraint(self, model, solver):
        model.MaxInnerEpitopeCleavage = aml.Param(initialize=self._max_cleavage, mutable=True)
        model.MaxInnerEpitopeCleavageConstraint = aml.Constraint(
            model.EpitopePositions * model.PositionInsideEpitope, rule=self._constraint_rule
        )
        super().insert_constraint(model, solver)

    def update(self, max_cleavage=None, ignore_first=None):
        dirty = False
        if ignore_first is not None:
            self._ignore_first = ignore_first
            dirty = True

        self._max_cleavage = max_cleavage
        if max_cleavage is not None:
            self.model.MaxInnerEpitopeCleavage.set_value(max_cleavage)
            dirty = True

        if dirty:
            super().update


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

    _constraint_names = ['MinNtCleavageConstraint']

    def __init__(self, min_cleavage):
        self._min_cleavage = min_cleavage

    def insert_constraint(self, model, solver):
        model.MinNtCleavage = aml.Param(initialize=self._min_cleavage, mutable=True)
        model.MinNtCleavageConstraint = aml.Constraint(
            model.EpitopePositions, rule=self._constraint_rule
        )

        super().insert_constraint(model, solver)

    @staticmethod
    def _constraint_rule(model, epi):
        epi_start = epi * (model.MaxSpacerLength + model.EpitopeLength)
        if epi > 0:
            return model.i[epi_start] >= model.MinNtCleavage
        else:
            return aml.Constraint.Satisfied

    def update(self, min_cleavage=None):
        self._min_cleavage = min_cleavage
        if min_cleavage is not None:
            self.model.MinNtCleavage.set_value(min_cleavage)
            super().update()


class MinimumCTerminusCleavage(VaccineConstraint):
    '''
    enforces a minimum cleavage score at the first position of every spacer
    '''

    _constraint_names = ['MinCtCleavageConstraint']

    def __init__(self, min_cleavage):
        self._min_cleavage = min_cleavage

    def insert_constraint(self, model, solver):
        model.MinCtCleavage = aml.Param(initialize=self._min_cleavage, mutable=True)
        model.MinCtCleavageConstraint = aml.Constraint(
            model.SpacerPositions, rule=self._constraint_rule
        )

        super().insert_constraint(model, solver)

    @staticmethod
    def _constraint_rule(model, spacer):
        spacer_start = spacer * (model.MaxSpacerLength + model.EpitopeLength) + model.EpitopeLength
        return model.i[spacer_start] >= model.MinCtCleavage

    def update(self, min_cleavage=None):
        self._min_cleavage = min_cleavage
        if min_cleavage is not None:
            self.model.MinCtCleavage.set_value(min_cleavage)
            super().update()


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

    def insert_constraint(self, model, solver):
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
            ), mutable=True)

            cs['MinCoverageConstraint'] = aml.Constraint(rule=lambda model: sum(
                cs['IsOptionCovered'][o] for o in cs['Options']
            ) >= cs['MinCoverage'])

        if self._min_conservation is not None:
            cs['MinConservation'] = aml.Param(initialize=(
                int(self._min_conservation) if self._min_conservation > 1
                else math.ceil(self._min_conservation * len(self._epitope_coverage[0]))
            ), mutable=True)

            cs['MinConservationConstraint'] = aml.Constraint(rule=lambda model: sum(
                model.x[e, p]
                for o in cs['Options']
                for e in cs['Coverage'][o]
                for p in model.EpitopePositions
            ) >= cs['MinConservation'] * model.VaccineLength)

        for k, v in cs.items():
            name = '%s_%s' % (self._name, k)
            setattr(model, name, v)
            if isinstance(v, aml.Constraint):
                self._constraint_names.append(name)
            elif isinstance(v, aml.Var):
                self._variable_names.append(name)

        self._cs = cs
        super().insert_constraint(model, solver)

    def update(self, min_coverage=None, min_conservation=None):
        dirty = False

        self._min_coverage = min_coverage
        if min_coverage is not None:
            self._cs['MinCoverage'].set_value(min_coverage)
            dirty = True

        self._min_conservation = min_conservation
        if min_conservation is not None:
            self._cs['MinConservation'].set_value(min_conservation)
            dirty = True

        if dirty:
            super().update()


# not an example of good engineering...
class DualImplementationConstraints(VaccineConstraint, ABC):
    '''
    base class for constraints whose implementation is different between
    fixed and variable spacer length
    '''

    def __init__(self):
        super().__init__()
        self._instance = None

    def insert_constraint(self, model, solver):
        if self._instance is None:
            if model.MinSpacerLength == model.MaxSpacerLength:
                self._instance = self._get_fixed_spacer_length_constraints()
                typ = 'fixed'
            else:
                self._instance = self._get_variable_spacer_length_constraints()
                typ = 'variable'

            if self._instance is None:
                raise RuntimeError(
                    'these constraints do not support a model with %s length spacers' % typ
                )

        self._instance.insert_constraint(model, solver)

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

    @property
    def _editor(self):
        return self._instance._editor

    def update(self, **kwargs):
        return self._instance.update(**kwargs)


class MinimumNTerminusCleavageGap(DualImplementationConstraints):
    '''
    enforces a given minimum cleavage gap between the first position of an epitope
    (which indicates correct cleavage at the end of the preceding spacer)
    and the cleavage of surrounding amino acids (next one and previous four)
    '''

    def __init__(self, min_gap):
        super().__init__()
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

    _constraint_names = ['AssignNTerminusMask', 'MinCleavageGapConstraint']
    _variable_names = ['NTerminusMask']

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

    def insert_constraint(self, model, solver):
        model.MinCleavageGap = aml.Param(initialize=self._min_gap, mutable=True)

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

        super().insert_constraint(model, solver)

    def update(self, min_gap=None):
        self._min_gap = min_gap
        if min_gap is not None:
            self.model.MinCleavageGap.set_value(self._min_gap)
            super().update()


class BoundCleavageInsideSpacers(DualImplementationConstraints):
    '''
    enforces a given minimum/maximum cleavage likelihood inside every spacer
    use None to disable the corresponding constraint
    '''

    def __init__(self, min_cleavage, max_cleavage):
        super().__init__()
        self._min_cleavage = min_cleavage
        self._max_cleavage = max_cleavage

    def _get_variable_spacer_length_constraints(self):
        return VariableLengthBoundCleavageInsideSpacers(self._min_cleavage, self._max_cleavage)

    def _get_fixed_spacer_length_constraints(self):
        return FixedLengthBoundCleavageInsideSpacers(self._min_cleavage, self._max_cleavage)


class FixedLengthBoundCleavageInsideSpacers(VaccineConstraint):
    '''
    enforces a given minimum/maximum cleavage likelihood inside every spacer
    except at the c-terminus
    use None to disable the corresponding constraint
    '''
    _constraint_names = ['BoundCleavageInsideSpacersConstraint']

    def __init__(self, min_cleavage, max_cleavage):
        self._min_cleavage = min_cleavage
        self._max_cleavage = max_cleavage

    def _constraint_rule(self, model, spacer, position):
        if position == 0:
            return aml.Constraint.Satisfied

        # absolute position in the sequence
        pos = (model.EpitopeLength + model.MaxSpacerLength) * spacer + position + model.EpitopeLength

        # I don't know why returning a tuple does not work
        if self._min_cleavage is not None and self._max_cleavage is not None:
            return (
                model.MinSpacerCleavage
                <= model.i[pos] <=
                model.MaxSpacerCleavage
            )
        elif self._min_cleavage is not None:
            return model.MinSpacerCleavage <= model.i[pos]
        elif self._max_cleavage is not None:
            return model.i[pos] <= model.MaxSpacerCleavage
        else:
            return aml.Constraint.Satisfied

    def insert_constraint(self, model, solver):
        model.MinSpacerCleavage = aml.Param(initialize=self._min_cleavage, mutable=True)
        model.MaxSpacerCleavage = aml.Param(initialize=self._max_cleavage, mutable=True)

        model.BoundCleavageInsideSpacersConstraint = aml.Constraint(
            model.SpacerPositions * model.AminoacidPositions,
            rule=self._constraint_rule
        )

        super().insert_constraint(model, solver)

    def update(self, model, solver, min_cleavage=None, max_cleavage=None):
        dirty = False
        self._min_cleavage = min_cleavage
        if min_cleavage is not None:
            dirty = True
            model.MinSpacerCleavage.set_value(self._min_cleavage)

        self._max_cleavage = max_cleavage
        if max_cleavage is not None:
            dirty = True
            model.MaxSpacerCleavage.set_value(self._max_cleavage)

        if dirty:
            super().update()


class VariableLengthBoundCleavageInsideSpacers(VaccineConstraint):
    '''
    enforces a given minimum/maximum cleavage likelihood inside every spacer
    use None to disable the corresponding constraint
    '''

    _constraint_names = ['BoundCleavageInsideSpacersConstraint']

    def __init__(self, min_cleavage, max_cleavage):
        self._min_cleavage = min_cleavage
        self._max_cleavage = max_cleavage

    def _constraint_rule(self, model, spacer, position):
        # TODO this *does* consider the c-terminus, it is thus incompatible with a minimum cleavage there

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

    def insert_constraint(self, model, solver):
        if self._min_cleavage is not None:
            model.MinSpacerCleavage = aml.Param(initialize=self._min_cleavage)
        if self._max_cleavage is not None:
            model.MaxSpacerCleavage = aml.Param(initialize=self._max_cleavage)

        model.BoundCleavageInsideSpacersConstraint = aml.Constraint(
            model.SpacerPositions * model.AminoacidPositions,
            rule=self._constraint_rule
        )

        super().insert_constraint(model, solver)

    def update(self, model, solver, min_cleavage=None, max_cleavage=None):
        dirty = False
        self._min_cleavage = min_cleavage
        if min_cleavage is not None:
            dirty = True
            model.MinSpacerCleavage.set_value(self._min_cleavage)

        self._max_cleavage = max_cleavage
        if max_cleavage is not None:
            dirty = True
            model.MaxSpacerCleavage.set_value(self._max_cleavage)

        if dirty:
            super().update()


class MonteCarloRecoveryEstimation(VaccineConstraint):
    '''
    performs several Monte Carlo cleavage simulations to estimate the recovery probability
    of each epitope
    '''

    # these are for fixed length, more are added below for variable length
    _constraint_names = [
        'McBernoulliTrialsSetPositive',
        'McBernoulliTrialsSetNegative',
        'McCleavageTrialsSetPositive',
        'McCleavageTrialsSetNegative',
        'McRecoveredEpitopePositionsSetPositive',
        'McRecoveredEpitopePositionsSetNegative',
        'McComputeProbs',
        'AssingRecoveredEpitopesFrequency',
    ]

    _variable_names = [
        'McRecoveredEpitopesFrequency',
        'McBernoulliTrials',
        'McCleavageTrials',
        'McCleavageProbs',
        'McRecoveredEpitopePositions'
    ]

    def __init__(self, mc_draws, cleavage_prior):
        self._editor = None

        if mc_draws < 1:
            raise ValueError('use at least one monte-carlo draws (and many more for reliable results)')
        self._mc_draws = mc_draws

        if cleavage_prior < 0 or cleavage_prior > 1:
            raise ValueError('cleavage prior must be between 0 and 1')
        self._cleavage_prior = math.log(cleavage_prior)

    def insert_constraint(self, model, solver):
        model.McDrawIndices = aml.RangeSet(0, self._mc_draws - 1)
        model.McDrawCount = aml.Param(initialize=self._mc_draws)
        model.McRandoms = aml.Param(
            model.McDrawIndices * model.SequencePositions,
            initialize=lambda *_: math.log(random.random()) - self._cleavage_prior
        )

        self._compute_bernoulli_trials(model)
        self._compute_cleavage_locations(model)
        self._compute_cleavage_frequencies(model)
        self._compute_recovered_epitopes(model)

        super().insert_constraint(model, solver)

    def _compute_bernoulli_trials(self, model):
        '''
        runs Bernoulli trials for every position based on the cleavage scores
        '''

        # a Bernoulli trial in position p successful if the cleavage score
        # is larger than the random number at position p
        insert_indicator_sum_beyond_threshold(
            model, 'McBernoulliTrials',
            model.McDrawIndices * model.SequencePositions,
            larger_than_is=1,
            get_variables_bounds_fn=lambda model, i, p: (
                [model.i[p]],
                25,
                model.McRandoms[i, p],
            )
        )

    def _compute_cleavage_locations(self, model):
        '''
        compute cleavage positions from the Bernoulli trials and cleavage in the previous positions
        '''

        # this is technically a dual implementation but there is so much duplicated code I prefer to keep them united
        if model.MinSpacerLength == model.MaxSpacerLength:
            self._compute_cleavage_locations_fixed_spacer_length(model)
        else:
            self._compute_cleavage_locations_variable_spacer_length(model)
            self._variable_names.append('McCleavageNotBlocked')
            self._constraint_names.extend(['McCleavageNotBlockedSetPositive', 'McCleavageNotBlockedSetNegative'])

    def _compute_cleavage_locations_fixed_spacer_length(self, model):
        '''
        compute cleavage positions from the Bernoulli trials and cleavage in the previous positions
        when the spacers have fixed length
        '''

        # cleavage happens at position p if
        #  - p is not in the first four locations of the vaccine, and
        #  - there was a successful Bernoulli trial at p, and
        #  - there was no cleavage in the previous four positions
        def get_vars_cleavage_trials(model, i, p):
            variables = []
            if p > 3:
                variables = [
                    model.McBernoulliTrials[i, p],
                ] + [
                    1 - model.McCleavageTrials[i, p + j]
                    for j in range(-3, 0)
                ]
            return variables

        insert_conjunction_constraints(
            model, 'McCleavageTrials',
            model.McDrawIndices * model.SequencePositions,
            get_vars_cleavage_trials, default=0
        )

    def _compute_cleavage_locations_variable_spacer_length(self, model):
        '''
        compute cleavage positions from the Bernoulli trials and cleavage in the previous positions
        when the spacers have variable length
        '''

        # this variable contains the actual cleavage indicators
        model.McCleavageTrials = aml.Var(model.McDrawIndices * model.SequencePositions,
                                         domain=aml.Binary, initialize=0)

        # this variable indicates whether position p+k is not blocking cleavage at position p
        # p+k does not block p iff
        #  - it is further than 4 amino acids, or
        #  - it does not contain an amino acid, or
        #  - it was not cleaved.
        # (easier to understand its negation via de morgan)
        # FIXME should be until -3, not -4 !!
        insert_disjunction_constraints(
            model, 'McCleavageNotBlocked',
            model.McDrawIndices * model.SequencePositions * model.OffsetAround,
            lambda model, i, p, k: [
                1 - model.d[p, k],
                1 - model.c[p+k],
                1 - model.McCleavageTrials[i, p+k],
            ] if k < 0 and k + p >= 0 else [],
            default=1.0
        )

        # cleavage happens at position p if
        #  - there is an amino acid at position p, and
        #  - p is not in the first four locations of the vaccine, and
        #  - there was a successful Bernoulli trial at p, and
        #  - none of the other positions block cleavage at p
        def get_vars_cleavage_trials(model, i, p):
            variables = []
            if p > 3:
                variables = [
                    model.c[p],
                    model.McBernoulliTrials[i, p],
                ] + [
                    model.McCleavageNotBlocked[i, p, o]
                    for o in model.OffsetAround
                    if o < 0
                ]
            return variables

        insert_conjunction_constraints(
            model, model.McCleavageTrials, None,
            get_vars_cleavage_trials, default=0
        )

    @staticmethod
    def _compute_cleavage_frequencies(model):
        '''
        compute cleavage frequencies, i.e. the average for every
        position along the monte-carlo trials
        '''
        model.McCleavageProbs = aml.Var(model.SequencePositions)
        model.McComputeProbs = aml.Constraint(
            model.SequencePositions,
            rule=lambda model, p: model.McCleavageProbs[p] == sum(
                model.McCleavageTrials[i, p] for i in model.McDrawIndices
            ) / model.McDrawCount
        )

    def _compute_recovered_epitopes(self, model):
        '''
        for every epitope position, decide if it was recovered or not
        '''

        # the epitope in position p is recovered if
        #  - there was no cleavage inside the epitope itself, and
        #  - there was cleavage at the n-terminus or p is the first epitope, and
        #  - there was cleavage at the c-terminus or p is the last epitope
        def get_conj_vars(model, i, p):
            nterminus_position = p * (model.EpitopeLength + model.MaxSpacerLength)
            cterminus_position = p * (model.EpitopeLength + model.MaxSpacerLength) + model.EpitopeLength

            inside_not_cleavage = [
                1 - model.McCleavageTrials[i, j]
                for j in range(nterminus_position + 1, cterminus_position)
            ]

            if p == model.VaccineLength - 1:
                return inside_not_cleavage + [model.McCleavageTrials[i, nterminus_position]]
            elif p == 0:
                return inside_not_cleavage + [model.McCleavageTrials[i, cterminus_position]]
            else:
                return inside_not_cleavage + [
                    model.McCleavageTrials[i, cterminus_position],
                    model.McCleavageTrials[i, nterminus_position],
                ]

        insert_conjunction_constraints(
            model, 'McRecoveredEpitopePositions',
            model.McDrawIndices * model.EpitopePositions,
            get_conj_vars,
        )

        # compute the frequency of epitope recovery for every position
        model.McRecoveredEpitopesFrequency = aml.Var(model.EpitopePositions, domain=aml.RealSet(0, 1), initialize=0)
        model.AssingRecoveredEpitopesFrequency = aml.Constraint(
            model.EpitopePositions, rule=lambda model, p: model.McRecoveredEpitopesFrequency[p] == sum(
                model.McRecoveredEpitopePositions[i, p] for i in model.McDrawIndices
            ) / model.McDrawCount
        )

    def update(self, model, solver):
        raise NotImplementedError('better to remove this and create new ones')
