from __future__ import division, print_function

import pandas as pd
import csv
import click
from Fred2.CleavagePrediction import CleavageSitePredictorFactory
from Fred2.Core import Protein


@click.command()
@click.argument('input-vaccine', type=click.Path())
def main(input_vaccine):
    with open(input_vaccine) as f:
        vaccines = list(csv.DictReader(f))
    
    pred = CleavageSitePredictorFactory('netchop')
    res = [pred.predict(Protein(v['vaccine'].replace(' ', ''))) for v in vaccines]

    pd.concat(res).to_csv('dev/chopped.csv')


if __name__ == '__main__':
    main()