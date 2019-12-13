from __future__ import print_function, division

import subprocess
import json
import datetime
import csv
from strobe_spacers import StrobeSpacer, MinimumCleavageGap
import utilities
import click
from pcm import DoennesKohlbacherPcm


def save_run_info(ctx):
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
    }

    with open('dev/experiment-history.jsonl', 'a') as f:
        f.write(json.dumps(run_info))
        f.write('\n')


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('output-vaccine', type=click.Path())
@click.option('--spacer-length', '-s', default=3, help='Length of the spacer to be designed')
@click.option('--num-epitopes', '-e', default=2, help='Number of epitopes in the vaccine')
@click.option('--top-immunogen', help='Only consider the top epitopes by immunogenicity', type=float)
@click.option('--top-proteins', help='Only consider the top epitopes by protein coverage', type=float)
@click.option('--top-alleles', help='Only consider the top epitopes by allele coverage', type=float)
@click.option('--min-cleavage-gap', '-g', default=0.25, help='Minimum cleavage gap')
@click.pass_context
def main(ctx, input_epitopes, output_vaccine, spacer_length, num_epitopes, top_immunogen, top_alleles,
         top_proteins, min_cleavage_gap):

    save_run_info(ctx)

    epitope_data = utilities.load_epitopes(input_epitopes, top_immunogen, top_alleles, top_proteins)
    epitopes = list(epitope_data.keys())
    immunogens = [epitope_data[e]['immunogen'] for e in epitopes]

    problem = StrobeSpacer(
        all_epitopes=epitopes,
        epitope_immunogen=immunogens,
        spacer_length=spacer_length,
        vaccine_length=num_epitopes,
        vaccine_constraints=MinimumCleavageGap(min_cleavage_gap),
    ).build_model()

    solution = problem.solve()

    with open(output_vaccine, 'w') as f:
        writer = csv.DictWriter(f, ('immunogen', 'vaccine'))
        writer.writeheader()

        writer.writerow({
            'immunogen': solution.immunogen,
            'vaccine': solution.sequence
        })


if __name__ == '__main__':
    main()
