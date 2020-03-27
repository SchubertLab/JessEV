import csv
import logging

import click

from spacers import constraints as spco
from spacers import objectives as spob
from spacers import utilities
from spacers.model import ModelParams, SolverFailedException, StrobeSpacer
from spacers.pcm import DoennesKohlbacherPcm


def design(epitope_data, min_spacer_length, max_spacer_length, num_epitopes,
           constraints, objective, pcm=None, solver_type='gurobi'):
    epitopes = list(epitope_data.keys())
    immunogens = [epitope_data[e]['immunogen'] for e in epitopes]

    params = ModelParams(
        all_epitopes=epitopes,
        epitope_immunogen=immunogens,
        min_spacer_length=min_spacer_length,
        max_spacer_length=max_spacer_length,
        vaccine_length=num_epitopes,
        pcm=pcm,
    )

    problem = StrobeSpacer(
        params=params,
        vaccine_constraints=constraints,
        vaccine_objective=objective,
        solver_type='gurobi',
    ).build_model()

    return problem.solve()


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('output-vaccine', type=click.Path())
# vaccine properties
@click.option('--min-spacer-length', '-s', default=0, help='Minimum length of the spacer to be designed')
@click.option('--max-spacer-length', '-S', default=4, help='Maximum length of the spacer to be designed')
@click.option('--num-epitopes', '-e', default=2, help='Number of epitopes in the vaccine')
# selection constraints
@click.option('--top-immunogen', help='Only consider the top epitopes by immunogenicity', type=float)
@click.option('--top-proteins', help='Only consider the top epitopes by protein coverage', type=float)
@click.option('--top-alleles', help='Only consider the top epitopes by allele coverage', type=float)
@click.option('--min-alleles', help='Vaccine must cover at least this many alleles', type=float)
@click.option('--min-proteins', help='Vaccine must cover at least this many proteins', type=float)
@click.option('--min-avg-prot-conservation', type=float,
              help='On average, epitopes in the vaccine must cover at least this many proteins')
@click.option('--min-avg-alle-conservation', type=float,
              help='On average, epitopes in the vaccine must cover at least this many alleles')
# cleavage constraints
@click.option('--min-nterminus-gap', '-g', help='Minimum cleavage gap', type=float)
@click.option('--min-nterminus-cleavage', '-n', help='Minimum cleavage at the N-terminus', type=float)
@click.option('--min-cterminus-cleavage', '-ct', help='Minimum cleavage at the C-terminus', type=float)
@click.option('--min-spacer-cleavage', '-c', help='Minimum cleavage inside the spacers', type=float)
@click.option('--max-spacer-cleavage', '-C', help='Maximum cleavage inside the spacers', type=float)
@click.option('--max-epitope-cleavage', '-E', help='Maximum cleavage inside epitopes', type=float)
@click.option('--epitope-cleavage-ignore-first', '-i', type=int, default=1,
              help='Ignore first amino acids for epitope cleavage')
# misc
@click.option('--log-file', type=click.Path(), help='Where to save the logs')
@click.option('--verbose', is_flag=True, help='Print debug messages')
@click.option('--solver-type', default='gurobi', help='Which linear programming solver to use')
@click.pass_context
def main(ctx=None, **kwargs):
    global LOGGER
    LOGGER = utilities.init_logging(kwargs.get('verbose', False), kwargs.get('log_file', None))
    utilities.main_dispatcher(design_cli, LOGGER, ctx, kwargs)


def design_cli(input_epitopes, output_vaccine, max_spacer_length,
               min_spacer_length, num_epitopes, top_immunogen, top_alleles,
               top_proteins, min_nterminus_gap, min_spacer_cleavage,
               max_epitope_cleavage, min_nterminus_cleavage, solver_type,
               epitope_cleavage_ignore_first, max_spacer_cleavage, min_alleles,
               min_proteins, min_avg_prot_conservation,
               min_avg_alle_conservation, min_cterminus_cleavage, **kwargs):

    epitope_data = utilities.load_epitopes(input_epitopes, top_immunogen, top_alleles, top_proteins)

    # discard epitopes containing invalid amino acids
    pcm = DoennesKohlbacherPcm()
    valid_epitopes = [epi for epi in epitope_data if all(a in pcm.AMINOS for a in epi)]
    epitope_data = {e: epitope_data[e] for e in valid_epitopes}
    LOGGER.info(f'Loaded {len(epitope_data)} epitopes')

    objective = spob.SimpleImmunogenicityObjective()

    constraints = []

    if min_nterminus_gap is not None:
        constraints.append(spco.MinimumNTerminusCleavageGap(min_nterminus_gap))

    if min_spacer_cleavage is not None or max_spacer_cleavage is not None:
        constraints.append(spco.BoundCleavageInsideSpacers(min_spacer_cleavage, max_spacer_cleavage))

    if max_epitope_cleavage is not None:
        constraints.append(spco.MaximumCleavageInsideEpitopes(
            max_epitope_cleavage, epitope_cleavage_ignore_first or 1))

    if min_nterminus_cleavage is not None:
        constraints.append(spco.MinimumNTerminusCleavage(min_nterminus_cleavage))

    if min_cterminus_cleavage is not None:
        constraints.append(spco.MinimumCTerminusCleavage(min_cterminus_cleavage))

    if min_alleles is not None or min_avg_alle_conservation is not None:
        allele_coverage = utilities.compute_allele_coverage(epitope_data.values())
        constraints.append(spco.MinimumCoverageAverageConservation(
            allele_coverage, min_alleles, min_avg_alle_conservation, name='Alleles'
        ))

    if min_proteins is not None or min_avg_prot_conservation is not None:
        protein_coverage = utilities.compute_protein_coverage(epitope_data.values())
        constraints.append(spco.MinimumCoverageAverageConservation(
            protein_coverage, min_proteins, min_avg_prot_conservation, name='Proteins'
        ))

    try:
        solution = design(epitope_data, min_spacer_length, max_spacer_length,
                          num_epitopes, constraints, objective, pcm, solver_type)
    except SolverFailedException as exc:
        LOGGER.error('Could not solve the problem: %s', exc.condition)
        return False, str(exc.condition)

    solution.to_csv(output_vaccine)
    LOGGER.info('Saved to %s', output_vaccine)

    return True


if __name__ == '__main__':
    main()
