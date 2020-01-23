import csv
import datetime
import json
import logging
import pprint
import subprocess
import sys
import traceback

import pandas as pd
import pyomo.environ as aml

from Fred2.Core import Allele, Peptide
from Fred2.EpitopePrediction import EpitopePredictionResult


def compute_allele_coverage(epitope_data):
    ''' compute allele coverage matrix
    '''
    alleles = [''] + list(set(a for e in epitope_data for a in e['alleles']))
    return [
        [int(a in e['alleles']) for a in alleles]
        for e in epitope_data
    ]


def compute_protein_coverage(epitope_data):
    ''' compute protein coverage matrix
    '''
    proteins = [''] + list(set(p for e in epitope_data for p in e['proteins']))
    return [
        [int(p in e['proteins']) for p in proteins]
        for e in epitope_data
    ]


def load_epitopes(epitopes_file, top_immunogen=None, top_alleles=None, top_proteins=None):
    ''' loads the epitopes from the given file, returning a dictionary mapping the epitope string to its data.
        optionally filters the epitopes by only taking the top N with the highest immunogenicity,
        or with the largest allele/protein coverage. if multiple options are given, the union of the
        matching epitopes is returned.
    '''
    with open(epitopes_file) as f:
        epitope_data = {}
        for row in csv.DictReader(f):
            row['immunogen'] = float(row['immunogen'])
            row['proteins'] = set(row['proteins'].split(';'))
            row['alleles'] = set(row['alleles'].split(';'))
            epitope_data[row['epitope']] = row

    if top_immunogen is None and top_alleles is None and top_proteins is None:
        return epitope_data

    top_immunogen = max(0, top_immunogen or 0)
    top_alleles = max(0, top_alleles or 0)
    top_proteins = max(0, top_proteins or 0)

    def filter_epitopes(epitopes, top_count, top_key):
        assert top_count > 0
        count = int(top_count) if top_count > 1 else int(top_count * len(epitopes))
        best = sorted(epitopes, key=lambda e: top_key(epitopes[e]), reverse=True)
        return set(best[:count])

    top_epitopes = set()
    if top_immunogen > 0:
        top_epitopes.update(filter_epitopes(epitope_data, top_immunogen, lambda e: e['immunogen']))
    if top_alleles > 0:
        top_epitopes.update(filter_epitopes(epitope_data, top_alleles, lambda e: len(e['alleles'])))
    if top_proteins > 0:
        top_epitopes.update(filter_epitopes(epitope_data, top_proteins, lambda e: len(e['proteins'])))

    return {e: epitope_data[e] for e in top_epitopes}


def init_logging(verbose, log_file, log_append=False):
    ''' initializes logging, optionally saving to a file as well
    '''
    level = (logging.DEBUG) if verbose else logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)

    if log_file:
        fh = logging.FileHandler(log_file, 'a' if log_append else 'w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.addHandler(sh)

    return logger


def save_run_info(logger, click_ctx, result):
    ''' saves the parameters and outcome of an execution to the experiment log
    '''
    try:
        git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        git_head = str(git_head).strip()
    except subprocess.CalledProcessError:
        git_head = None

    run_info = {
        'datetime': datetime.datetime.utcnow().isoformat(),
        'file': __file__,
        'command': click_ctx.command.name,
        'params': click_ctx.params,
        'head': git_head,
        'result': result
    }

    logger.debug('Run info:')
    for row in pprint.pformat(run_info).split('\n'):
        logger.debug(row)

    with open('dev/experiment-history.jsonl', 'a') as f:
        f.write(json.dumps(run_info))
        f.write('\n')


def main_dispatcher(main_fn, logger, click_ctx, main_kwargs):
    ''' calls the specified main function with the given kwargs
        and updates the experiment history to save parameters
        and outcome.
    '''

    try:
        ret = main_fn(**main_kwargs)
        save_run_info(logger, click_ctx, {'completed': True, 'result': ret})
    except Exception as exc:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        save_run_info(logger, click_ctx, {
            'completed': False,
            'traceback': traceback.format_tb(exc_traceback),
            'exception': traceback.format_exception_only(exc_type, exc_value),
        })
        raise


def get_alleles_and_thresholds(allele_file):
    df = pd.read_csv(allele_file, index_col=['allele'])
    return df


def affinities_from_csv(bindings_file, allele_data=None, peptide_coverage=None):
    ''' Loads binding affinities from a csv file. Optionally, augments alleles with probability
        and peptides with protein coverage. Discards all peptides for which coverage is not provided.
    '''
    df = pd.read_csv(bindings_file)

    df['Seq'] = df.Seq.apply(Peptide)
    if peptide_coverage is not None:
        keep = []
        for pep in df.Seq:
            if pep not in peptide_coverage:
                keep.append(False)
                continue

            keep.append(True)
            for prot in peptide_coverage[str(pep)]:
                pep.proteins[prot] = prot

        df = df[keep]

    df = df.set_index(['Seq', 'Method'])

    if allele_data is not None:
        df.columns = [Allele(c, allele_data[c]['frequency'] / 100) for c in df.columns]
    else:
        df.columns = [Allele(c) for c in df.columns]

    return EpitopePredictionResult(df)


def insert_indicator_sum_beyond_threshold(model, name, indices, larger_than_is,
                                          get_variables_bounds_fn, default=None):
    '''
    creates a variable of the given name and defined over the given indices (or uses the existing one)
    that indicates when the sum of certain variables is larger (larger_than_is=1) or smaller (larger_than_is=0)
    than given threshold, or zero in the opposite case

    get_variables_bound_fn takes as input the model and the indices, and must return a tuple
    containing (in order):
        - an iterable of variables to sum over
        - the upper bound for the sum (not necessarily tight)
        - the threshold above which the indicator is one
    '''
    if isinstance(name, aml.Var):
        result_var, indices = name, name.index_set()
        name = result_var.name
    elif not hasattr(model, name):
        result_var = aml.Var(indices, domain=aml.Binary, initialize=0)
        setattr(model, name, result_var)
    else:
        result_var = getattr(model, name)
        indices = result_var.index_set()

    def positive_rule(model, *idx):
        variables, upper_bound, threshold = get_variables_bounds_fn(model, *idx)
        if not variables:
            return result_var[idx] == default
        elif larger_than_is > 0:
            return sum(variables) - upper_bound * result_var[idx] <= threshold
        else:
            return sum(variables) - upper_bound * (1 - result_var[idx]) <= threshold

    def negative_rule(model, *idx):
        variables, upper_bound, threshold = get_variables_bounds_fn(model, *idx)
        if not variables:
            return result_var[idx] == default
        elif larger_than_is > 0:
            return sum(variables) + upper_bound * (1 - result_var[idx]) >= threshold
        else:
            return sum(variables) + upper_bound * result_var[idx] >= threshold

    setattr(model, name + 'SetPositive', aml.Constraint(indices, rule=positive_rule))
    setattr(model, name + 'SetNegative', aml.Constraint(indices, rule=negative_rule))


def insert_conjunction_constraints(model, name, indices, get_conjunction_vars_fn, default=1):
    ''' creates constraints that assign one to a variable if the conjunction of
        some other variables is true, i.e. y = x1 and x2 and ... and xn
    '''

    def get_vars_and_bounds(model, *idx):
        variables = get_conjunction_vars_fn(model, *idx)
        upper_bound = len(variables) + 1
        threshold = len(variables) - 0.5
        return variables, upper_bound, threshold

    insert_indicator_sum_beyond_threshold(model, name, indices, 1, get_vars_and_bounds, default)


def insert_disjunction_constraints(model, name, indices, get_conjunction_vars_fn, default=1):
    ''' creates constraints that assign one to a variable if the disjunction of
        some other variables is true, i.e. y = x1 or x2 or ... or xn
    '''

    def get_vars_and_bounds(model, *idx):
        variables = get_conjunction_vars_fn(model, *idx)
        upper_bound = len(variables) + 1
        threshold = 0.5
        return variables, upper_bound, threshold

    insert_indicator_sum_beyond_threshold(model, name, indices, 1, get_vars_and_bounds, default)
