import csv
import logging

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
@click.option('--cleavage-prior', '-p', default=0.1, help='Prior cleavage probability')
@click.option('--mc-draws', '-n', default=100, help='How many Monte Carlo experiments to use')
@click.option('--num-epitopes', '-e', default=5)
@click.option('--log-file', type=click.Path(), help='Where to save the logs')
@click.option('--verbose', is_flag=True, help='Print debug messages')
@click.pass_context
def main(ctx=None, **kwargs):
    '''
    design a vaccine optimizing the effective immunogenicity objective
    '''
    global LOGGER
    LOGGER = utilities.init_logging(kwargs.get('verbose', False), kwargs.get('log_file', None))
    utilities.main_dispatcher(seqdesign_cli, LOGGER, ctx, kwargs)


def seqdesign_cli(input_epitopes, output_vaccine, cleavage_prior, num_epitopes, mc_draws, **kwargs):
    # discard epitopes containing invalid amino acids
    epitope_data = utilities.load_epitopes(input_epitopes, None, None, None)
    pcm = DoennesKohlbacherPcm()
    valid_epitopes = [epi for epi in epitope_data if all(a in pcm.AMINOS for a in epi)]
    epitope_data = {e: epitope_data[e] for e in valid_epitopes}
    LOGGER.info(f'Loaded {len(epitope_data)} epitopes')

    constraints = [
        spco.MinimumNTerminusCleavage(1.5),
        spco.MinimumCTerminusCleavage(1.5),
        spco.MaximumCleavageInsideEpitopes(0.0, 3),
    ]

    epitopes = list(epitope_data.keys())
    immunogens = [epitope_data[e]['immunogen'] for e in epitopes]

    params = ModelParams(
        all_epitopes=epitopes,
        epitope_immunogen=immunogens,
        min_spacer_length=4,
        max_spacer_length=4,
        vaccine_length=num_epitopes,
    )

    problem = StrobeSpacer(
        params=params,
        vaccine_constraints=constraints,
        vaccine_objective=spob.SimpleImmunogenicityObjective(),
        solver_type='gurobi_persistent',
    ).build_model()

    LOGGER.info('Optimizing simple immunogenicity first...')
    _ = problem.solve()

    LOGGER.info('Now optimizing effective immunogenicity...')
    problem.add_constraint(spco.MonteCarloRecoveryEstimation(mc_draws, cleavage_prior))
    problem.set_objective(spob.EffectiveImmunogenicityObjective())
    problem.deactivate_constraint(spco.MinimumNTerminusCleavage)
    problem.deactivate_constraint(spco.MinimumCTerminusCleavage)
    problem.deactivate_constraint(spco.MaximumCleavageInsideEpitopes)

    solution = problem.solve()
    solution.to_csv(output_vaccine)

    return True


if __name__ == '__main__':
    main()
