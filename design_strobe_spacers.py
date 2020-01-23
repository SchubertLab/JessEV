import csv

import click

import strobe_spacers as sspa
import utilities

LOGGER = None


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('output-vaccine', type=click.Path())
# vaccine properties
@click.option('--max-spacer-length', '-S', default=4, help='Maximum length of the spacer to be designed')
@click.option('--min-spacer-length', '-s', default=0, help='Minimum length of the spacer to be designed')
@click.option('--num-epitopes', '-e', default=2, help='Number of epitopes in the vaccine')
# selection constraints
@click.option('--top-immunogen', help='Only consider the top epitopes by immunogenicity', type=float)
@click.option('--top-proteins', help='Only consider the top epitopes by protein coverage', type=float)
@click.option('--top-alleles', help='Only consider the top epitopes by allele coverage', type=float)
@click.option('--min-alleles', help='Vaccine must cover at least this many alleles')
@click.option('--min-proteins', help='Vaccine must cover at least this many proteins')
@click.option('--min-avg-prot-conservation', type=float,
              help='On average, epitopes in the vaccine must cover at least this many proteins')
@click.option('--min-avg-alle-conservation', type=float,
              help='On average, epitopes in the vaccine must cover at least this many alleles')
# cleavage constraints
@click.option('--min-nterminus-gap', '-g', help='Minimum cleavage gap', type=float)
@click.option('--min-nterminus-cleavage', '-n', help='Minimum cleavage at the n-terminus', type=float)
@click.option('--min-cterminus-cleavage', '-ct', help='Minimum cleavage at the n-terminus', type=float)
@click.option('--min-spacer-cleavage', '-c', help='Minimum cleavage inside the spacers', type=float)
@click.option('--max-spacer-cleavage', '-C', help='Maximum cleavage inside the spacers', type=float)
@click.option('--max-epitope-cleavage', '-E', help='Maximum cleavage inside epitopes', type=float)
@click.option('--epitope-cleavage-ignore-first', '-i', type=int,
              help='Ignore first amino acids for epitope cleavage')
# immunogenicity computation options
@click.option('--immunogen-mc-trials', '-mct', type=int,
              help='Number of Monte-Carlo trials to estimate the immunogenicity')
@click.option('--prior-cleavage-probability', '-cpp', type=float, default=0.1,
              help='Prior cleavage probability to use in the Monte Carlo trials')
# logging
@click.option('--log-file', type=click.Path(), help='Where to save the logs')
@click.option('--verbose', is_flag=True, help='Print debug messages')
@click.pass_context
def main(ctx=None, **kwargs):
    global LOGGER
    LOGGER = utilities.init_logging(kwargs.get('verbose', False), kwargs.get('log_file', None))
    utilities.main_dispatcher(design_strobe_spacers, LOGGER, ctx, kwargs)


def design_strobe_spacers(
        input_epitopes, output_vaccine, max_spacer_length, min_spacer_length, num_epitopes, top_immunogen,
        top_alleles, top_proteins, min_nterminus_gap, min_spacer_cleavage, max_epitope_cleavage,
        min_nterminus_cleavage, epitope_cleavage_ignore_first, max_spacer_cleavage, min_alleles,
        min_proteins, min_avg_prot_conservation, min_avg_alle_conservation, min_cterminus_cleavage,
        immunogen_mc_trials, prior_cleavage_probability,
        **kwargs
    ):

    epitope_data = utilities.load_epitopes(input_epitopes, top_immunogen, top_alleles, top_proteins)
    epitopes = list(epitope_data.keys())
    immunogens = [epitope_data[e]['immunogen'] for e in epitopes]

    constraints = []
    if min_nterminus_gap is not None:
        constraints.append(sspa.MinimumNTerminusCleavageGap(min_nterminus_gap))
    if min_spacer_cleavage is not None or max_spacer_cleavage is not None:
        constraints.append(sspa.BoundCleavageInsideSpacers(min_spacer_cleavage, max_spacer_cleavage))
    if max_epitope_cleavage is not None:
        constraints.append(sspa.MaximumCleavageInsideEpitopes(
            max_epitope_cleavage, epitope_cleavage_ignore_first or 0))
    if min_nterminus_cleavage is not None:
        constraints.append(sspa.MinimumNTerminusCleavage(min_nterminus_cleavage))
    if min_cterminus_cleavage is not None:
        constraints.append(sspa.MinimumCTerminusCleavage(min_cterminus_cleavage))
    if min_alleles is not None or min_avg_alle_conservation is not None:
        allele_coverage = utilities.compute_allele_coverage(epitope_data.values())
        constraints.append(sspa.MinimumCoverageAverageConservation(
            allele_coverage, min_alleles, min_avg_alle_conservation, name='Alleles'
        ))
    if min_proteins is not None or min_avg_prot_conservation is not None:
        protein_coverage = utilities.compute_protein_coverage(epitope_data.values())
        constraints.append(sspa.MinimumCoverageAverageConservation(
            protein_coverage, min_proteins, min_avg_prot_conservation, name='Proteins'
        ))

    if immunogen_mc_trials is not None:
        if immunogen_mc_trials < 10:
            raise ValueError('too few Monte Carlo trials')

        objective = sspa.MonteCarloEffectiveImmunogenicityObjective(
            immunogen_mc_trials, prior_cleavage_probability
        )
    else:
        objective = sspa.ImmunogenicityObjective()

    problem = sspa.StrobeSpacer(
        all_epitopes=epitopes,
        epitope_immunogen=immunogens,
        min_spacer_length=min_spacer_length,
        max_spacer_length=max_spacer_length,
        vaccine_length=num_epitopes,
        vaccine_constraints=constraints,
        vaccine_objective=objective,
    ).build_model()

    try:
        solution = problem.solve()
    except sspa.SolverFailedException as exc:
        LOGGER.error('Could not solve the problem: %s', exc.condition)
        return False, str(exc.condition)

    with open(output_vaccine, 'w') as f:
        writer = csv.DictWriter(f, ('immunogen', 'vaccine', 'spacers', 'cleavage'))
        writer.writeheader()

        writer.writerow({
            'immunogen': solution.immunogen,
            'vaccine': solution.sequence,
            'spacers': ';'.join(solution.spacers),
            'cleavage': ';'.join('%.3f' % c for c in solution.cleavage)
        })

    LOGGER.info('Saved to %s', output_vaccine)
    return True


if __name__ == '__main__':
    main()
