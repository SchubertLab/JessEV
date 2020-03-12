from collections import Counter
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from collections import defaultdict
from matplotlib import gridspec
from scipy.stats import pearsonr, spearmanr
from scipy import stats


def set_font_size(font_size):
    plt.rc('font', size=font_size)          # controls default text sizes
    plt.rc('axes', titlesize=font_size)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)    # legend fontsize
    plt.rc('figure', titlesize=font_size)   # fontsize of the figure title


def read_results(fname):
    '''
    reads a single result file with the design of a vaccine
    '''
    with open(fname) as f:
        return next(csv.DictReader(f))


def read_log(prefix):
    '''
    reads a collection of result files that all belong to the same experiment.
    they should all start with the given prefix, followed by a number of parameters
    separated by a dash.
    returns a dictionary keyed by the parameters and valued by the individual results
    '''
    log = {}
    for fname in os.listdir('./dev'):
        if not fname.startswith(prefix) or not fname.endswith('.csv'):
            continue

        # problem: I was dumb and used '-' as separator, not considering that some numbers
        # could be negative, so that now we have to parse things such as res--0.1-0.5.csv
        # whith parts -0.1 and 0.5
        # so what we do here is to replace '-' with '_', except for '--' that becomes '_-'
        # the previous example becomes res_-0.1_0.5.csv
        parts_name = fname.replace('--', '$%#').replace('-','_').replace('$%#', '_-')
        parts = tuple(map(float, parts_name[len(prefix):-len('.csv')].split('_')))
        log[parts] = read_results('./dev/' + fname)
    return log


def recover_epitopes_spacers_positions(log):
    '''
    given a single result item, returns the start-end indices in the vaccine
    sequence for each epitope and spacer.
    '''
    cursor = 9
    epitopes, spacers = [(0, cursor)], []
    for i, spa in enumerate(log['spacers'].split(';')):
        s, e = cursor, cursor + len(spa)
        cursor += 9 + len(spa)

        spacers.append((s, e))
        epitopes.append((e, cursor))
    return epitopes, spacers


def monte_carlo(log, count=1000, baseline=0.1):
    '''
    given a single result item, performs several Monte Carlo simulations.
    for each simulation, returns a list of indices where cleavage happened.
    '''
    cleavages = [float(c) for c in log['cleavage'].split(';')]
    for _ in range(count):
        cuts = [0]
        last = 0
        for i, (a, c) in enumerate(zip(log['vaccine'], cleavages)):
            if i - last < 4:
                continue

            # c = log( p(cut) / p(baseline) )
            pcut = np.exp(c) * baseline
            if np.random.random() <= pcut:
                cuts.append(i)
                last = i

        if i - last >= 4:
            cuts.append(len(log['vaccine']))

        yield cuts


def effective_immunogen(log, num_mc=1000, baseline=0.1):
    '''
    computes the effective immunogenicity of a result item.
    '''
    with open('dev/full-epitopes.csv') as f:
        immunogens = {
            row['epitope']: float(row['immunogen'])
            for row in csv.DictReader(f)
        }

    epitope_boundaries, _ = recover_epitopes_spacers_positions(log)
    vaccine_epitopes = [log['vaccine'][a:b] for a, b in epitope_boundaries]

    for cuts in monte_carlo(log, num_mc, baseline):
        ig = 0
        for i in range(1, len(cuts)):
            seq = log['vaccine'][cuts[i-1]:cuts[i]]
            if seq in vaccine_epitopes:
                ig += immunogens[seq]
        yield ig


def sample_recovery(log, num_mc=1000, baseline=0.1):
    '''
    perform a number of Monte Carlo simulations and returns, for each of them,
    the number of epitopes recovered from the vaccine.
    '''
    epi_pos, _ = recover_epitopes_spacers_positions(log)
    epitopes = set(log['vaccine'][a:b] for a, b in epi_pos)

    samples = []
    for cuts in monte_carlo(log, num_mc, baseline):
        count = 0
        for i in range(1, len(cuts)):
            part = log['vaccine'][cuts[i-1]:cuts[i]]
            if part in epitopes:
                count += 1
        samples.append(count)
    return samples


def read_epitope_data():
    '''
    reads epitopes, their immunogenicities and pathogen/hla coverage
    '''
    with open('dev/full-epitopes.csv') as f:
        immunogens, proteins, alleles = {}, {}, {}
        for row in csv.DictReader(f):
            immunogens[row['epitope']] = float(row['immunogen'])

    # only use these epitopes to evaluate coverage
    # because this is what we used to optimize the vaccine
    with open('dev/made-epitopes.csv') as f:
        proteins, alleles = {}, {}
        for row in csv.DictReader(f):
            proteins[row['epitope']] = set(row['proteins'].split(';'))
            alleles[row['epitope']] = set(
                row['alleles'].split(';')
            ) if row['alleles'] else set()
    return immunogens, proteins, alleles


def process_log(log, num_mc, baseline):
    '''
    given a result item, performs Monte Carlo simulations and computes several
    metric for each simulation, in order: number of fragments, effective
    immunogenicity, average fragment length, total immunogenicity, covered
    proteins, covered alleles, and simulation index.
    '''
    epitope_boundaries, _ = recover_epitopes_spacers_positions(log)
    vaccine_epitopes = [log['vaccine'][a:b] for a, b in epitope_boundaries]
    immunogens, proteins, alleles = read_epitope_data()

    for trial, cuts in enumerate(monte_carlo(log, num_mc, baseline)):
        count = ig = length = 0
        covered_prots, covered_alls = set(), set()
        for i in range(1, len(cuts)):
            part = log['vaccine'][cuts[i-1]:cuts[i]]
            length += len(part)

            if part in vaccine_epitopes:
                count += 1
                ig += immunogens[part]
            if part in proteins:
                covered_prots.update(proteins[part])
            if part in alleles:
                covered_alls.update(alleles[part])

        yield (
            count, ig, length / len(cuts),
            float(log['immunogen']),
            len(covered_prots),
            len(covered_alls), trial
        )


def process_results(fname, key, basepc, res, num_mc):
    '''
    given a file name, performs Monte Carlo simulations and appends metrics to res
    '''
    seq = read_results(fname)
    for vals in process_log(seq, num_mc, basepc):
        res.append((basepc, key, np.nan, np.nan, np.nan, np.nan) + vals)


def compute_mc_experiments(baselines=None):
    '''
    reads the results of several experiments, runs Monte Carlo experiments for
    each of them and returns a data frame with the results.
    '''
    res = []
    bases = [
        #('res-comb2-', 2), ('res-spac-', 2), ('res-spacmin-', 2),
        # ('res-comb-nc1-', 2), ('res-conservation-', 3),
        ('res-cov-', 3), ('res-comb-nc-', 2), ('res-eig-', 3)
    ]
    num_mc = 1000

    baselines = baselines or (list(np.arange(0, 1.01, 0.05)) + [0.025, 0.075])
    for basepc in baselines:
        for base, parts in bases:
            for k, v in read_log(base).items():
                epitope_boundaries, _ = recover_epitopes_spacers_positions(v)
                for vals in process_log(v, num_mc, basepc):
                    k = k + (np.nan,) * (4 - len(k))  # same length
                    res.append((basepc, base,) + k + vals)

        process_results('./dev/sequential-full.csv', 'sequential', basepc, res, num_mc)
        process_results('./dev/sequential-cov.csv', 'sequential-cov', basepc, res, num_mc)

    return pd.DataFrame(res, columns=[
        'baseline', 'experiment', 'param_1', 'param_2', 'param_3', 'param_4',
        'recovered', 'effective_immunogen', 'avg_len', 'immunogen',
        'proteins', 'alleles', 'trial'
    ])


def summarize_experiment(g):
    '''
    given a data frame with Monte Carlo experiments, computes mean, standard
    deviation, 25-th and 75-th percentile of effective immunogenicity, fragment
    length, recovered epitopes, proteins covered and alleles covered.
    '''
    return pd.Series({
        'mean_eig': g.effective_immunogen.mean(),
        'std_eig': g.effective_immunogen.std(),
        'eig_q25': g.effective_immunogen.quantile(0.25),
        'eig_q75': g.effective_immunogen.quantile(0.75),
        'mean_len': g.avg_len.mean(),
        'std_len': g.avg_len.std(),
        'len_q25': g.avg_len.quantile(0.25),
        'len_q75': g.avg_len.quantile(0.75),
        'mean_rec': g.recovered.mean(),
        'std_rec': g.recovered.std(),
        'rec_q25': g.recovered.quantile(0.25),
        'rec_q75': g.recovered.quantile(0.75),
        'mean_prot': g.proteins.mean(),
        'std_prot': g.proteins.std(),
        'prot_q25': g.proteins.quantile(0.25),
        'prot_q75': g.proteins.quantile(0.75),
        'mean_alle': g.alleles.mean(),
        'std_alle': g.alleles.std(),
        'alle_q25': g.alleles.quantile(0.25),
        'alle_q75': g.alleles.quantile(0.75),
    })


def compare_experiments(monte_carlo_df, column):
    '''
    for each experiment, finds the parameter settings that resulted in the largest metric
    '''
    return monte_carlo_df.groupby([
        'baseline', 'experiment', 'param_1', 'param_2', 'param_3', 'param_4'
    ]).apply(
        summarize_experiment
    ).reset_index().groupby([
        'baseline', 'experiment'
    ]).apply(
        lambda g: g.loc[g[column].idxmax()]
    ).reset_index(drop=True)


def compute_probability_of_improvement(monte_carlo_df, comparison, column,
                                       exp1, exp2, bootstraps=5000):
    '''
    computes the probability of improvement of the metric `column` of
    experiment `exp2` with respect to `exp1`.
    '''
    def probability_of_improvement(g):
        ig1 = g[g.experiment == exp1][column].values
        ig2 = g[g.experiment == exp2][column].values

        idx1 = np.random.choice(ig1, bootstraps)
        idx2 = np.random.choice(ig2, bootstraps)

        return pd.Series({
            'gt': np.mean(idx2 > idx1),
            'ge': np.mean(idx2 >= idx1),
            'eq': np.mean(np.abs(idx2 - idx1) < 1e-2),
            'eqnz': np.mean((np.abs(idx2 - idx1) < 1e-2) & (idx2 > 0)),
        })

    poi = monte_carlo_df.merge(comparison, on=[
        'baseline', 'experiment', 'param_1', 'param_2', 'param_3', 'param_4'
    ]).groupby(
        'baseline'
    ).apply(
        probability_of_improvement
    )
    poi.index = poi.index / 1000

    return poi


def compute_expected_improvement(comparison, column, exp1, exp2):
    '''
    compute the expected improvement of the metric `column` of experiment
    `exp2` with respect to experiment `exp1`.
    '''
    improv = comparison[(
        comparison.experiment == exp1
    ) | (
        comparison.experiment == exp2
    )].pivot('baseline', 'experiment', column)
    improv.index = improv.index / 1000
    return improv


def decode_str_color(color):
    '''
    returns the RGB components of a color in string form, e.g. #aabbcc
    '''
    assert len(color) == 7 and color[0] == '#'
    return (
        int(color[1:3], 16),
        int(color[3:5], 16),
        int(color[5:7], 16),
    )


def decode_tuple_color(color):
    '''
    ensures that the color in tuple-format contains integers in 0-255
    '''
    return (
        int(color[0]) if color[0] > 1 else int(255 * color[0]),
        int(color[1]) if color[1] > 1 else int(255 * color[1]),
        int(color[2]) if color[2] > 1 else int(255 * color[2]),
    )


def rgba2rgb(rgb_fore, rgb_back, alpha):
    '''
    converts a RGB color + alpha with the specified background to RGB
    '''

    rf, gf, bf = decode_str_color(rgb_fore) if isinstance(rgb_fore, str) else decode_tuple_color(rgb_fore)
    rb, gb, bb = decode_str_color(rgb_back) if isinstance(rgb_back, str) else decode_tuple_color(rgb_fore)

    # blend colors
    rr = int(rf * alpha + (1 - alpha) * rb)
    gr = int(gf * alpha + (1 - alpha) * gb)
    br = int(bf * alpha + (1 - alpha) * bb)

    return '#%02x%02x%02x' % (rr, gr, br)


def plot_vaccine(fname, hline=None, gray_first=(0,-1), ylim=(-2.1, 2.1),
                 savefig=False, style='bar', ax=None):
    '''
    plots a vaccine
    '''
    with open(fname) as f:
        log = next(csv.DictReader(f))

    c2 = mpl.rcParams['axes.prop_cycle'].by_key()['color'][2]
    spacer_color = rgba2rgb(c2, '#ffffff', alpha=0.4)

    c0 = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]
    epi_color = rgba2rgb(c0, '#ffffff', alpha=0.4)

    if ax is None:
        fig = plt.figure(figsize=(15, 3), dpi=96)
        ax = fig.subplots()
    else:
        fig = None

    cleavages = list(map(float, log['cleavage'].split(';')))
    if style == 'line':
        ax.plot(cleavages)
        ax.grid(axis='x')

    if hline is not None:
        for k, v in hline.items():
            ax.plot([0, len(log['vaccine'])], [v, v], label=k)
        ax.legend()

    if style == 'line':
        ax.plot([0, len(log['vaccine'])], [0, 0], 'k--')

    # highlight spacers
    cursor = 9
    spacers = []
    cols = [0] * cursor
    for i, spa in enumerate(log['spacers'].split(';')):
        s, e = cursor, cursor + len(spa)
        spacers.append((s, e))
        cursor += 9 + len(spa)

        if style == 'bar':
            cols.extend([1] * len(spa))
            cols.extend([0] * gray_first[0])
            cols.extend([2] * gray_first[1])
            cols.extend([0] * (9 - max(gray_first[0], 0) - max(gray_first[1], 0)))

        if style == 'line':
            ax.fill_between([s - 0.5, e - 0.5], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                            color='C2', alpha=0.3, label='Spacers' if i == 0 else None)
            if gray_first is not None:
                ax.fill_between([e - 0.5 + gray_first[0], e + gray_first[1] - 0.5],
                                [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='#dddddd')

    if style == 'bar':
        ax.bar(range(len(cleavages)), cleavages, color=[
            [epi_color, spacer_color, '#dddddd'][c] for c in cols
        ])

    # highlight cleaved positions on the x ticks
    bgc = rgba2rgb(c2, '#ffffff', alpha=0.25)
    ax.set_xticks(range(len(log['vaccine'])))
    ax.set_xticklabels(list(log['vaccine']))
    for i, t in enumerate(ax.get_xticklabels()):
#         if cleavages[i] > 0:
#             t.set_color('red')
#             t.set_fontweight('bold')

        # highlight spacers x ticks
        if style == 'bar':
            for ss, se in spacers:
                if ss <= i < se:
                    t.set_color('C2')
                    t.set_fontweight('bold')
                elif se + gray_first[0] <= i <= se + gray_first[1]:
                    t.set_backgroundcolor('#dddddd')

    ax.set_xlabel('Vaccine Sequence')
    ax.set_ylabel('Cleavage')
    ax.set_title('Immunogenicity: %.3f' % float(log['immunogen']))
    ax.set_ylim(ylim)
    ax.set_xlim(-0.5, len(log['vaccine']) - 0.5)
    ax.grid(False, axis='x')
    ax.set_yticks([-1.5, 0., 1.5])

    # monte carlo cleavage simulations
    for _ in range(100):
        last = 0
        baseline = 0.25  # geometric(0.1) has a mean of 10
        for i, (a, c) in enumerate(zip(log['vaccine'], cleavages)):
            if i <= last + 3:
                continue

            # c = log( p(cut) / p(baseline) )
            pcut = np.exp(c) * baseline
            if np.random.random() < pcut:
                offset = 0.1 * (2 * np.random.random() - 1)
                ax.plot([i - 0.5 + offset, i - 0.5 + offset], ylim, 'k', alpha=0.03)
                last = i

    if fig is not None:
        fig.tight_layout()
        if savefig:
            fig.savefig(fname.replace('.csv', '.png'))


def plot_vaccine_interact_style(fname, hline=None, gray_first=(0,-1), ylim=(-2.1, 2.1),
                                savefig=False, style='bar', ax=None):
    '''
    plots a vaccine to be used in the presentation for <interact>
    hides the cleavage score, emphasizes Monte Carlo cleavage probability,
    and correct/wrong cleavage positions
    '''
    c2 = mpl.rcParams['axes.prop_cycle'].by_key()['color'][2]
    spacer_color = rgba2rgb(c2, '#ffffff', alpha=0.4)

    c0 = mpl.rcParams['axes.prop_cycle'].by_key()['color'][3]
    epi_color = rgba2rgb(c0, '#ffffff', alpha=0.4)

    with open(fname) as f:
        log = next(csv.DictReader(f))

    if ax is None:
        fig = plt.figure(figsize=(15, 3), dpi=96)
        ax = fig.subplots()
    else:
        fig = None

    # highlight spacers
    cursor = 9
    spacers = []
    cols = [0] * cursor
    for i, spa in enumerate(log['spacers'].split(';')):
        s, e = cursor, cursor + len(spa)
        spacers.append((s, e))
        cursor += 9 + len(spa)

        if style == 'bar':
            cols.extend([1] + [0] * (len(spa) - 1))
            cols.extend([1] + [0] * 8)

        if style == 'line':
            ax.fill_between([s - 0.5, e - 0.5], [ylim[0], ylim[0]], [ylim[1], ylim[1]],
                            color='C2', alpha=0.3, label='Spacers' if i == 0 else None)
            if gray_first is not None:
                ax.fill_between([e - 0.5 + gray_first[0], e + gray_first[1] - 0.5],
                                [ylim[0], ylim[0]], [ylim[1], ylim[1]], color='#dddddd')

    # highlight cleaved positions on the x ticks
    bgc = rgba2rgb(c2, '#ffffff', alpha=0.25)
    ax.set_xticks(range(len(log['vaccine'])))
    ax.set_xticklabels(list(log['vaccine']))
    for i, t in enumerate(ax.get_xticklabels()):
#         if cleavages[i] > 0:
#             t.set_color('red')
#             t.set_fontweight('bold')

        # highlight spacers x ticks
        if style == 'bar':
            for ss, se in spacers:
                if ss <= i < se:
                    #t.set_backgroundcolor(bgc)
                    t.set_fontweight('bold')
                elif se + gray_first[0] <= i <= se + gray_first[1]:
                    t.set_backgroundcolor('#dddddd')

    ax.set_xlabel('Vaccine Sequence')
    ax.set_ylabel('Cleavage Probability')
    ax.set_title('Immunogenicity: %.3f' % float(log['immunogen']))
    ax.set_ylim(ylim)
    ax.set_xlim(-0.5, len(log['vaccine']) - 0.5)
    ax.grid(False, axis='x')

    # monte carlo cleavage simulations
    cuts = [0] * len(log['vaccine'])
    cleavages = list(map(float, log['cleavage'].split(';')))
    for _ in range(100):
        last = 0
        baseline = 0.25  # geometric(0.1) has a mean of 10
        for i, (a, c) in enumerate(zip(log['vaccine'], cleavages)):
            if i <= last + 3:
                continue

            # c = log( p(cut) / p(baseline) )
            pcut = np.exp(c) * baseline
            if np.random.random() < pcut:
                offset = 0.1 * (2 * np.random.random() - 1)
                #ax.plot([i - 0.5 + offset, i - 0.5 + offset], ylim, 'k', alpha=0.03)
                cuts[i] += 1
                last = i

    ax.bar(range(len(cuts)), [c / 100 for c in cuts], color=[
        [epi_color, spacer_color, '#dddddd'][c] for c in cols
    ])
    
    ax.legend([
        mpl.patches.Patch(facecolor=epi_color),
        mpl.patches.Patch(facecolor=spacer_color),
    ], ['Wrong', 'Correct'], loc='upper left')

    if fig is not None:
        fig.tight_layout()
        if savefig:
            fig.savefig(fname.replace('.csv', '.png'))


def plot_prefix_2(prefix, xlabel, ylabel, title, ax=None, xgrid=None,
                  ygrid=None, imshow_kwargs=None, swapxy=False, key='immunogen'):
    '''
    plots the results of an experiment with two parameters as a heatmap
    '''
    imshow_kwargs = imshow_kwargs or {}

    log = read_log(prefix)

    xgrid = xgrid or set(x for x, y in log.keys())
    all_xs = sorted(xgrid)

    ygrid = ygrid or set(y for x, y in log.keys())
    all_ys = sorted(ygrid)

    mat = -1 * np.ones((len(all_xs), len(all_ys)))
    for i, x in enumerate(all_xs):
        for j, y in enumerate(all_ys):
            mat[i, j] = float(log.get((x, y), {}).get(key, np.nan))

    if swapxy:
        all_xs, all_ys = all_ys, all_xs
        xlabel, ylabel = ylabel, xlabel
        mat = mat.T

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()
    else:
        fig = None

    ax.set_title(title)
    mm = ax.imshow(mat.T, aspect='auto', **imshow_kwargs)
    ax.set_xticks(range(len(all_xs)))
    ax.set_xlim(-0.5, len(all_xs) - 0.5)
    ax.set_xticklabels(all_xs)
    ax.set_xlabel(xlabel)
    ax.set_yticks(range(len(all_ys)))
    ax.set_yticklabels(all_ys)
    ax.set_ylim(-0.5, len(all_ys) - 0.5)
    ax.set_ylabel(ylabel)
    if fig is not None:
        fig.colorbar(mm, ax=ax)
    return mm


def plot_by_baseline(ax, data, col_name, xlabel=None, ylabel=None,
                     title=None, experiment_names=None):
    experiment_names = experiment_names or {}
    experiment = data.experiment.iloc[0]
    mean = data['mean_' + col_name].values
    q25 = data[col_name + '_q25'].values
    q75 = data[col_name + '_q75'].values
    errors_hi = q75 - mean
    errors_lo = mean - q25

    ax.errorbar(
        data.baseline / 1000,
        mean,
        yerr=(errors_lo, errors_hi), # comparison[mask]['std_' + values],
        fmt='.-',
        capsize=4,
        label=experiment_names.get(experiment, experiment),
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')


def plot_many_by_baseline(ax, comparison_df, masks, col_name, xlabel=None,
                          ylabel=None, title=None, experiment_names=None):
    for mm in masks:
        plot_by_baseline(ax, comparison_df[mm], col_name, xlabel, ylabel,
                         title, experiment_names)


def find_parameter_trace(comparison):
    done = set()
    xs, ys, ps = [], [], []
    xgrid = sorted(comparison.param_1.unique())
    ygrid = sorted(comparison.param_2.unique())

    for i, row in comparison.iterrows():
        i1, i2 = xgrid.index(row.param_1), ygrid.index(row.param_2)
        if (i1, i2) in done:
            continue

        done.add((i1, i2))
        xs.append(row.param_1)
        ys.append(row.param_2)
        ps.append(row.baseline / 1000)

    return xs, ys, ps


def annotate_axis(ax, notes, xs, ys, alignments, offset_base=4):
    for i in range(len(notes)):
        va, ha = alignments[i].split()
        offset = (
            -offset_base if ha == 'right' else offset_base if ha == 'left' else 0,
            offset_base if va == 'bottom' else -offset_base if va == 'top' else 0,
        )
        
        ax.annotate(
            notes[i],
            xy=(xs[i], ys[i]),
            xytext=offset,
            textcoords='offset points',
            ha=ha, va=va
        )


def plot_parameter_evolution(comparison, comp_cov, ax):
    xgrid = [1.4, 1.74, 1.8, 1.95, 2.25, 2.5, 2.75]
    epigrid = [-1, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]

    xs_eig, ys_eig, ps_eig = find_parameter_trace(
        comparison[comparison.experiment == 'res-comb-nc-']
    )

    xs_cov, ys_cov, ps_cov = find_parameter_trace(
        comp_cov[comp_cov.experiment == 'res-cov-']
    )

    ax.plot(ys_eig, xs_eig, 'o-', label='Eff. imm.')
    ax.plot(ys_cov, xs_cov, 'o-', label='Eff. cov.')
    
    annotate_axis(
        ax, ['%.3f' % p for p in ps_eig],
        ys_eig, xs_eig, alignments=[
            'center left', 'center right', 'center right',
            'top left', 'center left'
        ]
    )

    annotate_axis(
        ax, ['%.3f' % p for p in ps_cov],
        ys_cov, xs_cov, alignments=[
            'top center', 'bottom center', 'bottom center'
        ]
    )

    ax.set_ylabel('Minimum termini cleavage')
    ax.set_xlabel('Maximum inner epitope cleavage')
    ax.set_title('(a)')
    ax.legend(loc='upper left')
    ax.set_xticks([-1, -0.5, 0., 0.5, 1., 1.5, 2.])


def plot_gridsearch(fig ,ax):
    mm = plot_prefix_2(
        'res-comb-nc-',
        xlabel='Minimum termini cleavage',
        ylabel='Maximum inner epitope cleavage',
        title='(b)',
        #swapxy=True,
        ax=ax,
        imshow_kwargs={'cmap': 'viridis', 'vmin': 0.4, 'vmax': 1.3}
    )
    ax.grid(False)
    fig.colorbar(mm, ax=ax)


def plot_eig_by_settings(df, fig, axes):
    baselines = sorted(df.baseline.unique().tolist())
    termini_cleavage = sorted(set(df[df.experiment == 'res-comb-nc-'].param_1))
    vmin, vmax = min(termini_cleavage), max(termini_cleavage)
    cmap = plt.get_cmap('viridis')
    colors = [cmap.colors[int(200 * (b - vmin) / (vmax - vmin))] for b in termini_cleavage]

    def app(g):
        base = g.baseline.values[0]
        param1 = g.param_1.values[0]
        axidx = baselines.index(base)
        if axidx >= len(axes):
            return
        
        axes[axidx].plot(
            g.param_2, g.mean_eig,
            c=colors[termini_cleavage.index(param1)],
            label=f'tc: {param1:.1f}'
        )
    
    groups = df[(
        df.experiment == 'res-comb-nc-'
    )].groupby([
        'baseline', 'experiment', 'param_1', 'param_2', 'param_3', 'param_4'
    ]).apply(
        summarize_experiment
    ).reset_index().groupby([
        'baseline', 'experiment', 'param_1',
    ]).apply(lambda g: app(g))

    for i, (ax) in enumerate(axes):
        if i == 1:
            ax.set_title(f'(c)')

        ax.annotate(
            'pc: ' + f'{baselines[i] / 1000:.3f}'[1:],
            xy=(0, 0.95), xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='top'
        )
        ax.set_ylim(0, 1.0)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([0, 0.3, 0.6, 0.9])

        if i < 6:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(['', '-0.5', '', '0.5', ''])

        if i not in [0, 3, 6]:
            ax.set_yticklabels([])

        if i == 7:
            ax.set_xlabel('Inner epitope cleavage')
        if i == 3:
            ax.set_ylabel('Effective immunogenicity')

            
def plot_ranked_parameters(df, summaries, column, ax, xlim, xticks, ylabel):
    cmap = plt.get_cmap('plasma')
    baselines = sorted(df.baseline.unique())
    vmin, vmax = 0, len(baselines)
    cmap2 = plt.get_cmap('tab20c')

    colors = [4, 4, 4, 12, 12, 12, 12, 20, 20, 20]
    for i, (baseline, group) in enumerate(summaries.groupby('baseline')):
        ax.loglog(
            range(1, len(group) + 1),
            group[column].sort_values(ascending=False) / group[column].max(),
            #c=cmap(int(255 * ((baselines.index(baseline)) - vmin) / (vmax - vmin))),
            #c=cmap2(4 + 4 * (i // 3)),
            c=cmap2(colors[i] - 4),
            #label=f'pc: {baseline / 1000:.3f}',
        )

    ax.grid(True, axis='x', which='minor')
    ax.legend([
         mpl.patches.Patch(color=cmap2(i - 4)) for i in sorted(set(colors))
    ], ['0<=pc<0.1', '0.1<=pc<0.5', '0.5<=pc<=1'], loc='lower left', ncol=1)
    ax.set_xlabel('Parameters rank')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.45, 1.04)
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.grid(True, axis='y', which='minor')
