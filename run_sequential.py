import csv

import click

import pcm
import utilities
from Fred2.CleavagePrediction.PSSM import PCM
from Fred2.Core import Allele, Protein
from Fred2.Core.Peptide import Peptide
from Fred2.EpitopeAssembly import EpitopeAssembly, EpitopeAssemblyWithSpacer
from Fred2.EpitopePrediction.PSSM import BIMAS
from Fred2.EpitopeSelection import OptiTope

LOGGER = None


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('input-alleles', type=click.Path())
@click.argument('input-affinities', type=click.Path())
@click.argument('output-vaccine', type=click.Path())
# vaccine properties
@click.option('--max-spacer-length', '-S', default=4, help='Maximum length of the spacer to be designed')
@click.option('--min-spacer-length', '-s', default=0, help='Minimum length of the spacer to be designed')
@click.option('--num-epitopes', '-e', default=2, help='Number of epitopes in the vaccine')
# selection constraints
@click.option('--min-alleles', help='Vaccine must cover at least this many alleles')
@click.option('--min-proteins', help='Vaccine must cover at least this many proteins')
# logging
@click.option('--log-file', type=click.Path(), help='Where to save the logs')
@click.option('--verbose', is_flag=True, help='Print debug messages')
# misc
@click.option('--solver', default='gurobi', help='Which linear programming solver to use')
@click.pass_context
def main(ctx=None, **kwargs):
    global LOGGER
    LOGGER = utilities.init_logging(kwargs.get('verbose', False), kwargs.get('log_file', None))
    #utilities.main_dispatcher(run_sequential, LOGGER, ctx, kwargs)
    run_sequential(**kwargs)


def run_sequential(input_epitopes, input_alleles, input_affinities,
                   output_vaccine, num_epitopes, min_alleles, min_proteins,
                   solver, **kwargs):

    epitope_data = {
        k: v for k, v in utilities.load_epitopes(input_epitopes).items()
        if 'X' not in k
    }
    LOGGER.info('Loaded %d epitopes', len(epitope_data))

    peptide_coverage = {
        # we don't really need the actual protein sequence, just fill it with the id to make it unique
        Peptide(r['epitope']): set(Protein(gid, gene_id=gid) for gid in r['proteins'])
        for r in epitope_data.values()
    }

    allele_data = utilities.get_alleles_and_thresholds(input_alleles).to_dict('index')
    alleles = [Allele(allele.replace('HLA-', ''), prob=data['frequency'] / 100)
               for allele, data in allele_data.items()]
    threshold = {allele.replace('HLA-', ''): data['threshold'] for allele, data in allele_data.items()}
    LOGGER.info('Loaded %d alleles', len(threshold))

    affinities = utilities.affinities_from_csv(input_affinities, allele_data, peptide_coverage=peptide_coverage)
    LOGGER.info('Loaded %d affinities', len(affinities))

    LOGGER.info('Selecting epitopes...')
    model = OptiTope(affinities, threshold, k=num_epitopes, solver=solver)
    if min_alleles is not None:
        model.activate_allele_coverage_const(min_alleles)
    if min_proteins is not None:
        model.activate_antigen_coverage_const(min_proteins)
    selected_epitopes = model.solve()

    LOGGER.info('Creating spacers...')
    vaccine = EpitopeAssemblyWithSpacer(
        selected_epitopes, PCM(), BIMAS(), alleles,
        threshold=threshold, solver=solver
    ).solve()

    immunogen = sum(epitope_data[str(e)]['immunogen'] for e in vaccine[::2])
    sequence = ''.join(map(str, vaccine))
    cleavage = pcm.DoennesKohlbacherPcm().cleavage_per_position(sequence)

    with open(output_vaccine, 'w') as f:
        writer = csv.DictWriter(f, ('immunogen', 'vaccine', 'spacers', 'cleavage'))
        writer.writeheader()
        writer.writerow({
            'immunogen': immunogen,
            'vaccine': sequence,
            'spacers': ';'.join(str(e) for e in vaccine[1::2]),
            'cleavage': ';'.join('%.3f' % c for c in cleavage)
        })


if __name__ == '__main__':
    main()
