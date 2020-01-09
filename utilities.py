import numpy as np
import multiprocessing as mp
import logging
import pandas as pd
from Fred2.Core import Allele, Peptide, Protein
from Fred2.IO import FileReader
from Fred2.EpitopePrediction import EpitopePredictionResult
import csv


class Trie:
    def __init__(self):
        self.children = {}

    def _get_child(self, letter, create=True):
        if create and self.children.get(letter) is None:
            self.children[letter] = Trie()
        return self.children.get(letter)

    def insert(self, string, pos_in_string=0):
        if pos_in_string >= len(string):
            return

        child = self._get_child(string[pos_in_string], create=True)
        child.insert(string, pos_in_string + 1)

    def reachable_strings(self, string, mistakes_allowed, pos_in_string=0, mistakes_done=0):
        ''' yields all strings in the trie that can be reached from the given strings
            by changing at most `mistakes_allowed` characters, and the number of characters changed
        '''
        if not isinstance(string, list):
            string = list(string)

        if pos_in_string >= len(string):
            yield ''.join(string), mistakes_done
            return

        if mistakes_allowed - mistakes_done <= 0:
            child = self._get_child(string[pos_in_string], create=False)
            if child is not None:
                reachable = child.reachable_strings(string, mistakes_allowed,
                                                    pos_in_string + 1, mistakes_done)
                for s in reachable:
                    yield s
        else:
            for letter, child in self.children.iteritems():
                if letter == string[pos_in_string]:
                    reachable = child.reachable_strings(string, mistakes_allowed,
                                                        pos_in_string + 1, mistakes_done)
                    for s in reachable:
                        yield s
                else:
                    correct = string[pos_in_string]
                    string[pos_in_string] = letter
                    reachable = child.reachable_strings(string, mistakes_allowed,
                                                        pos_in_string + 1, mistakes_done + 1)
                    for s in reachable:
                        yield s
                    string[pos_in_string] = correct


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


def compute_coverage_matrix(epitope_data, min_alleles, min_proteins,
                            min_prot_conservation, min_alle_conservation,
                            num_proteins, num_alleles):
    def make_absolute_and_append(value, maxval, lst):
        lst.append(int(value) if value > 1 else int(value * maxval))

    type_coverage, min_type_coverage, min_type_conservation = [], [], []

    # allele coverage
    if min_alleles > 0 or min_alle_conservation > 0:
        allele_coverage = compute_allele_coverage(epitope_data)
        type_coverage.append(np.array(
            [0] * len(allele_coverage[0]) + allele_coverage
        ))

        make_absolute_and_append(min_alleles, num_alleles, min_type_coverage)
        make_absolute_and_append(min_alle_conservation, num_alleles, min_type_conservation)

    # protein coverage
    if min_proteins > 0 or min_prot_conservation > 0:
        protein_coverage = compute_protein_coverage(epitope_data)
        type_coverage.append(np.array(
            [[0] * len(protein_coverage[0])] + protein_coverage
        ))
        make_absolute_and_append(min_proteins, num_proteins, min_type_coverage)
        make_absolute_and_append(min_prot_conservation, num_proteins, min_type_conservation)

    # must pad all matrices to the same size
    if len(type_coverage) > 1:
        max_rows = max(a.shape[0] for a in type_coverage)
        max_cols = max(a.shape[1] for a in type_coverage)

        type_coverage = [
            np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), 'constant', constant_values=0)
            for arr in type_coverage
        ]

    return type_coverage, min_type_coverage, min_type_conservation


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


def get_alleles_and_thresholds(allele_file):
    df = pd.read_csv(allele_file, index_col=['allele'])
    return df


def read_annotated_proteins(proteins_file):
    ''' Reads proteins from a fasta file and extracts their metadata from the header.
        Currently follows the format of the HIV database
    '''
    proteins = FileReader.read_fasta(proteins_file, in_type=Protein)
    for prot in proteins:
        parts = prot.transcript_id.split('.')
        prot.transcript_id = parts[-1]
    return proteins


def affinities_from_csv(bindings_file, allele_data=None, peptide_coverage=None, proteins=None):
    ''' Loads binding affinities from a csv file. Optionally, augments alleles with probability
        and peptides with protein coverage.
    '''
    df = pd.read_csv(bindings_file)

    df['Seq'] = df.Seq.apply(Peptide)
    if peptide_coverage is not None:
        for pep in df.Seq:
            for prot in peptide_coverage[str(pep)]:
                pep.proteins[prot] = prot

    df = df.set_index(['Seq', 'Method'])

    if allele_data is not None:
        df.columns = [Allele(c, allele_data[c]['frequency'] / 100) for c in df.columns]
    else:
        df.columns = [Allele(c) for c in df.columns]

    return EpitopePredictionResult(df)


def init_logging(verbose, log_file, log_append=False):
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


def compute_all_pairs_suffix_prefix_cost(strings):
    all_costs = np.zeros((len(strings), len(strings)))
    for i, string_from in enumerate(strings):
        for j, string_to in enumerate(strings):
            cost = None
            if j == 0 or i == j:
                cost = 0
            elif i == 0:
                cost = len(string_to)
            else:
                all_costs[i, j] = compute_suffix_prefix_cost(str(string_from), str(string_to))
    return all_costs


def compute_suffix_prefix_cost(string_from, string_to):
    k = 1
    overlap = 0
    while k <= len(string_from) and k <= len(string_to):
        if string_from[-k:] == string_to[:k]:
            overlap = k
        k += 1
    return len(string_to) - overlap


def itake(it, n):
    res = []
    for i in range(n):
        try:
            res.append(next(it))
        except StopIteration:
            break
    return res


def batches(it, bsize):
    res = 3
    while res:
        res = itake(it, bsize)
        if res:
            yield res


def parallel_apply(apply_fn, task_generator, processes, preload=64, timeout=99999):
    pool = mp.Pool(processes=processes if processes > 0 else (mp.cpu_count() + processes))

    try:
        tasks = []
        task_count = processed_count = 0
        for task in itake(task_generator, preload):
            tasks.append(pool.apply_async(apply_fn, task))
            task_count += 1

        cursor = 0
        while processed_count < task_count:
            result = tasks[cursor].get(timeout)
            yield result
            tasks[cursor] = None

            processed_count += 1
            cursor += 1
            next_task = itake(task_generator, 1)
            if next_task:
                tasks.append(pool.apply_async(apply_fn, next_task[0]))
                task_count += 1

    except:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()


def load_overlaps(input_overlaps, min_overlap):
    # load overlaps
    # we don't use the csv module to be much quicker, but less flexible:
    # we assume overlaps are sorted by cost and columns are ordered as from,to,cost
    current_cost = None
    with open(input_overlaps) as f:
        header_checked = False
        for row in f:
            parts = row.strip().split(',')
            if header_checked:
                cost = float(parts[2])
                if current_cost is not None and cost < current_cost:
                    raise RuntimeError('overlap file not sorted! sort it by cost')
                elif cost > 9 - min_overlap:
                    break
                elif parts[0] != parts[1]:
                    yield parts[0], parts[1], cost
                current_cost = cost
            elif parts[0] != 'from' or parts[1] != 'to' or parts[2] != 'cost':
                raise RuntimeError('Make sure the columns are ordered as follows: from,to,cost')
            else:
                header_checked = True


def load_edges_from_overlaps(input_overlaps, min_overlap, epitopes):
    if epitopes[0] != '':
        epitopes = [''] + epitopes

    epitope_index = {e: i for i, e in enumerate(epitopes)}

    # create edges to/from the dummy vertex
    edges = {}
    for e, i in epitope_index.iteritems():
        edges[(0, i + 1)] = len(e)
        edges[(i + 1, 0)] = 0

    for epi_from, epi_to, cost in load_overlaps(input_overlaps, min_overlap):
        i, j = epitope_index.get(epi_from), epitope_index.get(epi_to)
        if i is not None and j is not None:
            edges[(i, j)] = cost

    # the overlap file does not contain pairs that do not overlap
    # so we have to add them manually if needed
    if min_overlap <= 0:
        for i in range(1, len(epitope_index)):
            for j in range(1, len(epitope_index)):
                if i != j and (i, j) not in edges:
                    edges[(i, j)] = 9

    return edges
