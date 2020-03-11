import csv
import logging
import multiprocessing as mp

import click
import pyomo.environ as aml

from spacers import constraints as spco
from spacers import objectives as spob
from spacers import utilities
from spacers.model import ModelParams, SolverFailedException, StrobeSpacer
from spacers.pcm import DoennesKohlbacherPcm


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('output-vaccine', type=click.Path())
@click.option('--rounds', '-r', default=10, help='How many rounds to perform')
@click.option('--increment', '-i', default=0.05, help='Effective conservation increment per round')
@click.option('--cleavage-prior', '-p', default=0.1, help='Prior cleavage probability')
@click.option('--mc-draws', '-n', default=100, help='How many Monte Carlo experiments to use')
@click.option('--num-epitopes', '-e', default=5, help='How many epitopes in the vaccine')
@click.option('--upper-bound', '-u', type=float, help='Upper bound for the effective immunogenicity')
@click.option('--solution-epitopes', '-se', multiple=True, help='Initial epitopes for the warm start')
@click.option('--solution-spacers', '-ss', multiple=True, help='Initial spacers for the warm start')
@click.option('--log-file', type=click.Path(), help='Where to save the logs')
@click.option('--verbose', is_flag=True, help='Print debug messages')
@click.pass_context
def main(ctx=None, **kwargs):
    '''
    repeatedly optimizes the effective immunogenicity while gradually
    increasing the minimum effective conservation
    '''
    global LOGGER
    LOGGER = utilities.init_logging(kwargs.get('verbose', False), kwargs.get('log_file', None))
    utilities.main_dispatcher(pareto_cli, LOGGER, ctx, kwargs)


def pareto_cli(input_epitopes, output_vaccine, rounds, increment, upper_bound, solution_epitopes,
               solution_spacers, cleavage_prior, num_epitopes, mc_draws, **kwargs):
    # discard epitopes containing invalid amino acids
    epitope_data = utilities.load_epitopes(input_epitopes, None, None, None)
    pcm = DoennesKohlbacherPcm()
    valid_epitopes = [epi for epi in epitope_data if all(a in pcm.AMINOS for a in epi)]
    epitope_data = {e: epitope_data[e] for e in valid_epitopes}
    protein_coverage = utilities.compute_protein_coverage(epitope_data.values())
    LOGGER.info(f'Loaded {len(epitope_data)} epitopes')

    epitopes = list(epitope_data.keys())
    immunogens = [epitope_data[e]['immunogen'] for e in epitopes]

    problem = StrobeSpacer(
        params=ModelParams(
            all_epitopes=epitopes,
            epitope_immunogen=immunogens,
            min_spacer_length=4,
            max_spacer_length=4,
            vaccine_length=num_epitopes,
        ),
        vaccine_constraints=[
            spco.MinimumNTerminusCleavage(1.5),
            spco.MinimumCTerminusCleavage(1.5),
            spco.MaximumCleavageInsideEpitopes(0.5, 3),
            spco.MonteCarloRecoveryEstimation(mc_draws, cleavage_prior),
        ],
        vaccine_objective=spob.SimpleImmunogenicityObjective(),
        solver_type='gurobi_persistent',
    ).build_model()

    LOGGER.info('Optimizing simple immunogenicity first...')
    problem.solve()

    LOGGER.info('Now optimizing effective immunogenicity...')
    problem.set_objective(spob.EffectiveImmunogenicityObjective(upper_bound))
    problem.deactivate_constraint(spco.MinimumNTerminusCleavage)
    problem.deactivate_constraint(spco.MinimumCTerminusCleavage)
    problem.deactivate_constraint(spco.MaximumCleavageInsideEpitopes)
    problem.add_constraint(spco.MinimumCoverageAverageConservation(protein_coverage, min_coverage=2, name='proteins'))
    consconstr = spco.MinimumEffectiveConservation(0, name='proteins')
    problem.add_constraint(consconstr)
    problem.solve()

    #consconstr = None
    for i in range(1, rounds):
        cons = i * increment
        if consconstr is None:
            problem.add_constraint(
                spco.MinimumCoverageAverageConservation(protein_coverage, min_coverage=2, name='proteins'),
            )
            consconstr = spco.MinimumEffectiveConservation(cons, name='proteins')
            problem.add_constraint(consconstr)
        else:
            consconstr.update(min_conservation=cons)

        LOGGER.info('Solving with minimum effective conservation of %.3f', cons)
        solution = problem.solve(options={
            'Threads': mp.cpu_count(),
            'NonConvex': 2,
            'MIPFocus': 1,  # if upper_bound is not None else 3,
            # 'NodeMethod': 1,
            'MIPGap': 0.01,
        })
        LOGGER.info('Effective immunogenicity: %.3f', aml.value(solution.immunogen))

        solution.to_csv(output_vaccine.format(cons))

    return True


if __name__ == '__main__':
    main()
