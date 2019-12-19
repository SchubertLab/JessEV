from __future__ import print_function, division

import subprocess
import traceback
import sys
import pprint
import json
import datetime
import csv
import strobe_spacers as sspa
import utilities
import click
import logging


LOGGER = None


def save_run_info(ctx, result):
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        git_head = str(git_head).strip()
    except subprocess.CalledProcessError:
        git_head = None

    run_info = {
        'datetime': datetime.datetime.utcnow().isoformat(),
        'file': __file__,
        'command': ctx.command.name,
        'params': ctx.params,
        'head': git_head,
        'result': result
    }

    LOGGER.debug('Run info:')
    for row in pprint.pformat(run_info).split('\n'):
        LOGGER.debug(row)

    with open('dev/experiment-history.jsonl', 'a') as f:
        f.write(json.dumps(run_info))
        f.write('\n')


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('output-vaccine', type=click.Path())
# vaccine properties
@click.option('--max-spacer-length', '-S', default=4, help='Maximum length of the spacer to be designed')
@click.option('--min-spacer-length', '-s', default=0, help='Minimum length of the spacer to be designed')
@click.option('--num-epitopes', '-e', default=2, help='Number of epitopes in the vaccine')
# vaccine constraints
@click.option('--min-nterminus-gap', '-g', help='Minimum cleavage gap', type=float)
@click.option('--min-nterminus-cleavage', '-n', help='Minimum cleavage at the n-terminus', type=float)
@click.option('--min-spacer-cleavage', '-c', help='Minimum cleavage inside the spacers', type=float)
@click.option('--max-spacer-cleavage', '-C', help='Maximum cleavage inside the spacers', type=float)
@click.option('--max-epitope-cleavage', '-E', help='Maximum cleavage inside epitopes', type=float)
@click.option('--epitope-cleavage-ignore-first', '-i', help='Ignore first amino acids for epitope cleavage', type=int)
# epitope pre-selection
@click.option('--top-immunogen', help='Only consider the top epitopes by immunogenicity', type=float)
@click.option('--top-proteins', help='Only consider the top epitopes by protein coverage', type=float)
@click.option('--top-alleles', help='Only consider the top epitopes by allele coverage', type=float)
# misc
@click.option('--log-file', type=click.Path(), help='Where to save the logs')
@click.option('--verbose', is_flag=True, help='Print debug messages')
@click.pass_context
def main(ctx, **kwargs):
    global LOGGER
    LOGGER = utilities.init_logging(kwargs.get('verbose', False), kwargs.get('log_file', None))

    try:
        ret = design_strobe_spacers(**kwargs)
        save_run_info(ctx, {'completed': True, 'result': ret})
    except Exception as exc:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        save_run_info(ctx, {
            'completed': False,
            'traceback': traceback.format_tb(exc_traceback),
            'exception': traceback.format_exception_only(exc_type, exc_value),
        })
        raise


def design_strobe_spacers(
        input_epitopes, output_vaccine, max_spacer_length, min_spacer_length, num_epitopes, top_immunogen,
        top_alleles, top_proteins, min_nterminus_gap, min_spacer_cleavage, max_epitope_cleavage, log_file,
        min_nterminus_cleavage, verbose, epitope_cleavage_ignore_first, max_spacer_cleavage
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

    problem = sspa.StrobeSpacer(
        all_epitopes=epitopes,
        epitope_immunogen=immunogens,
        min_spacer_length=min_spacer_length,
        max_spacer_length=max_spacer_length,
        vaccine_length=num_epitopes,
        vaccine_constraints=constraints,
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
