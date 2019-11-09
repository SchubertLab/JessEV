from __future__ import print_function, division

import numpy as np
import csv
from strobe_spacers import StrobeSpacer
import utilities
import click
from pcm import PCM_MATRIX, AMINOS, AMINO_IDX, get_pcm_matrix


def compute_epitopes_cleavage(epitopes, spacer_length):
    good_epitopes, epi_idx, before, after = [], [], [], []
    for epi in epitopes:
        try:
            indices = [AMINO_IDX[a] for a in epi]
        except KeyError:
            continue

        epi_idx.append(indices)
        good_epitopes.append(epi)

        # total score for cleavage inside the epitope
        base_score = sum(
            PCM_MATRIX[indices[i + j - 4]][j]
            for j in range(6)
            for i in range(4, len(epi) - 1)
        )

        before.append(
            base_score

            # account for the effect of this epitope on the cleavage of the next four amino acids:
            #  - add if cleavage in the spacer or in the following epitope
            #  - subtract if cleavage happens at the edges of the spacer
            + sum(
                PCM_MATRIX[indices[len(epi) - 4 + i + j]][j] * (
                    -1 if i == 0 or i == spacer_length else 1
                ) for i in range(4) for j in range(4 - i)
            )
        )

        after.append(
            base_score

            # account for the effect of this epitope on the cleavage of the previous two amino acids
            # again, add or subtract according to where the amino acid is
            + sum(
                PCM_MATRIX[indices[i]][j] * (
                    -1 if i == 0 or i == spacer_length + 1 else 1
                ) for i in range(2) for j in range(4, 6 - i)
            )
        )
    
    if len(good_epitopes) < len(epitopes):
        print('WARN: discarded %d epitopes with invalid amino acids' % (len(epitopes) - len(good_epitopes)))
        if not good_epitopes:
            raise RuntimeError('all epitopes contain invalid aminoacids!')
    return good_epitopes, epi_idx, before, after


def evaluate(seq, p1, p2):
    return sum(
        PCM_MATRIX[AMINO_IDX[seq[p1 + i - 4]]][i] + PCM_MATRIX[AMINO_IDX[seq[p2 + i - 4]]][i]
        for i in range(6)
    )


@click.command()
@click.argument('input-epitopes', type=click.Path())
@click.argument('output-tradeoff', type=click.Path())
@click.option('--spacer-length', '-l', default=3, help='Length of the spacer to be designed')
@click.option('--num-epitopes', '-e', default=2, help='Number of epitopes in the vaccine')
@click.option('--top-immunogen', help='Only consider the top epitopes by immunogenicity', type=float)
@click.option('--top-proteins', help='Only consider the top epitopes by protein coverage', type=float)
@click.option('--top-alleles', help='Only consider the top epitopes by allele coverage', type=float)
def main(input_epitopes, output_tradeoff, spacer_length, num_epitopes, top_immunogen, top_alleles, top_proteins):
    epitope_data = utilities.load_epitopes(input_epitopes, top_immunogen, top_alleles, top_proteins)
    epitopes, epitopes_indicized, cleavages_before, cleavages_after = compute_epitopes_cleavage(
        epitope_data.keys(), spacer_length
    )
    immunogens = [epitope_data[e]['immunogen'] for e in epitopes]

    solver = StrobeSpacer(
        epitope_immunogen=immunogens,
        epitope_cleavage_before=cleavages_before,
        epitope_cleavage_after=cleavages_after,
        spacer_length=spacer_length,
        vaccine_length=num_epitopes,
        pssm_matrix=get_pcm_matrix(),
        min_cleavage=None,
        all_epitopes=epitopes_indicized,
        min_cleavage_gap=0.25,
    ).build_model()

    solver.set_objective(immunogen=True, cleavage=False)
    result_immunog = solver.solve(tee=1)
    print('Max immunogen %.3f at cleavage %.3f' % (result_immunog.immunogen, result_immunog.cleavage))
    
    solver.set_objective(immunogen=False, cleavage=True)
    result_cleav = solver.solve(tee=0)
    print('Max cleavage %.3f at immunogen %.3f' % (result_cleav.cleavage, result_cleav.immunogen))

    with open(output_tradeoff, 'w') as f:
        writer = csv.DictWriter(f, ('immunogen', 'cleavage', 'threshold', 'vaccine'))
        writer.writeheader()

        for alpha in np.arange(0, 1.01, 0.05):
            solver.set_max_cleavage(result_immunog.cleavage + alpha * (result_cleav.cleavage - result_immunog.cleavage))
            solver.set_objective(immunogen=True, cleavage=True)
            result = solver.solve(tee=0)
            print('Immunogen %.3f at %.1f%% cleavage %.3f' % (result.immunogen, 100 * alpha, result.cleavage))

            seq = []
            for i in range(num_epitopes):
                seq.append(epitopes[result.epitopes[i]])
                if i < num_epitopes - 1:
                    seq.append(''.join(AMINOS[j] for j in result.spacers[i]))
            vax = ' '.join(seq)
            print('   ', vax)

            writer.writerow({
                'immunogen': result.immunogen,
                'cleavage': result.cleavage,
                'threshold': alpha,
                'vaccine': vax,
            })


if __name__ == '__main__':
    main()