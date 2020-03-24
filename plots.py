# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # plots 

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.tools import add_constant
import statsmodels.formula.api as smf
import matplotlib.patches as mpatches
from collections import defaultdict, Counter
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import plot_utils as utl

# %%
#\usepackage{layouts}
#\printinunitsof{in}\prntlen{\textwidth} \prntlen{\linewidth} 

OABtextwidth = 6.7261 #in
OABlinewidth = 3.2385 #in
OABdpi = 350
OABfigfmt = 'pdf'

# %%
sns.set(context='paper', style='whitegrid')
plt.rc('grid', linewidth=0.3)
sns.set_palette('colorblind')

# use LaTeX fonts in the plot
# https://ercanozturk.org/2017/12/16/python-matplotlib-plots-in-latex/
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# %%
df_fname = './dev/experiments-monte-carlo.csv.gz'
if not os.path.exists(df_fname):
    df = utl.compute_mc_experiments(baselines=[
        0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.500, 0.650, 1.0
    ])
    # convert to int so we can reliaby filter on them
    df.baseline = 1000 * df.baseline
    df.baseline = df.baseline.astype(np.int)
    df = df.fillna(-100)  # stupid pandas discards na's in groupby's
    df.to_csv(df_fname, index=False)
    print('Re-created Monte Carlo experiments')
else:
    df = pd.read_csv(df_fname)
    print('Used existing Monte Carlo experiments')
    
df

# %% [markdown]
# # comparison

# %%
# comparison on effective immunogenicity and protein converage
comparison = utl.compare_experiments(df, 'mean_eig')
comp_cov = utl.compare_experiments(df, 'mean_prot')

mask_seq = (comparison.experiment == 'sequential')
mask_us = (comparison.experiment == 'res-comb-nc-')

mask_comp_us = (comp_cov.experiment == 'res-cov-')
mask_comp_seq = (comp_cov.experiment == 'sequential-cov')

# expected improvement
improv = utl.compute_expected_improvement(
    comparison, 'mean_eig', 'sequential', 'res-comb-nc-'
)
improv_cov = utl.compute_expected_improvement(
    comp_cov, 'mean_prot', 'sequential-cov', 'res-cov-'
)
improv_alle = utl.compute_expected_improvement(
    comp_cov, 'mean_alle', 'sequential-cov', 'res-cov-'
)

# probability of improvement
poi = utl.compute_probability_of_improvement(
    df, comparison, 'effective_immunogen', 'sequential', 'res-comb-nc-', 5000
)
poi_cov = utl.compute_probability_of_improvement(
    df, comp_cov, 'proteins', 'sequential-cov', 'res-cov-', 100
)
poi_alle = utl.compute_probability_of_improvement(
    df, comp_cov, 'alleles', 'sequential-cov', 'res-cov-', 100
)

# %%
experiment_replacement = {
    'res-comb-nc-': 'Sim.',
    'res-cov-': 'Sim.',
    'sequential': 'Seq.',
    'sequential-cov': 'Seq.',
}

utl.set_font_size(6)

fig = plt.figure(figsize=(OABtextwidth, OABtextwidth / 2), dpi=OABdpi)
((ax2, ax1, ax3), (ax4, ax5, ax6)) = fig.subplots(2, 3)

utl.plot_many_by_baseline(
    ax1, comparison, [mask_us, mask_seq], 'rec',
    title='(b) Recovered epitopes',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax2, comparison, [mask_us, mask_seq], 'eig',
    title='(a) Effective Immunogenicity',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax3, comp_cov, [mask_comp_us, mask_comp_seq], 'prot',
    xlabel='Prior cleavage probability',
    title='(c) Pathogen Coverage',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax4, comp_cov, [mask_comp_us, mask_comp_seq], 'alle',
    xlabel='Prior cleavage probability',
    title='(d) HLA Coverage',
    experiment_names=experiment_replacement
)

ax5.semilogx(poi_cov.index, 1 - poi_cov['ge'], 'C0.-', 
             label='Eff. pathogen coverage')
ax5.semilogx(poi.index, 1 - poi['ge'], 'C1.-',
             label='Eff. immunogenicity')
ax5.semilogx(poi_alle.index, 1 - poi_alle['ge'], 'C2.-',
             label='Eff. allele coverage')

ax6.loglog(improv.index, improv['res-comb-nc-'] / improv['sequential'], 'C0.-',
           label='Eff. pathogen coverage')
ax6.loglog(improv_cov.index, improv_cov['res-cov-'] / improv_cov['sequential-cov'],
           'C1.-', label='Eff. immunogenicity')
ax6.loglog(improv_alle.index, improv_alle['res-cov-'] / improv_alle['sequential-cov'],
           'C2.-', label='Eff. allele coverage')

ax5.set_ylim(-0.05, 0.55)
ax6.set_ylim(0.8, 1.2e3)
ax5.set_xlim(0.02, 1.1)
ax6.set_xlim(0.02, 1.1)

ax5.set_title('(e) Worsening probability')
ax6.set_title('(f) Expected improvement')

ax6.set_xlabel('Prior cleavage probability $p_c$')

ax6.grid(True, axis='y', which='minor')

ax4.set_ylim(-1, 31)
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax5.legend(loc='upper left')
ax6.legend(loc='upper left')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.grid(True, 'minor', axis='x')
    ax.set_xticks([0.02, 0.1, 0.4, 1])
    ax.set_xticklabels(['0.02', '0.1', '0.4', '1.0'])

fig.tight_layout()
fig.savefig(f'dev/fig3.{OABfigfmt}', bbox_inches='tight')

# %% [markdown]
# # study of parameters

# %%
summaries = df[
    df.experiment == 'res-comb-nc-'
].groupby([
    'baseline', 'param_1', 'param_2'
]).apply(
    utl.summarize_experiment
).reset_index()

# %%
summ_cov = df[
    df.experiment == 'res-cov-'
].groupby([
    'baseline', 'param_1', 'param_2'
]).apply(
    utl.summarize_experiment
).reset_index()

# %%
utl.set_font_size(6)
fig = plt.figure(figsize=(OABtextwidth, OABtextwidth / 2), dpi=OABdpi)

root_gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1])

left_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=root_gs[0], hspace=0.35
)

botleft_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=left_gs[1], wspace=0.35, hspace=0.35
)

right_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    5, 2, subplot_spec=root_gs[1], wspace=0.075, hspace=0.125
)

utl.plot_ranked_parameters(
    df, summaries, 'mean_eig', fig.add_subplot(left_gs[0]),
    xlim=(0.9, 50), xticks=[1, 2, 3, 4, 5, 10, 20, 30, 40], ylabel='Relative eff. immunog.'
)
utl.plot_parameter_evolution(comparison, comp_cov, fig.add_subplot(botleft_gs[0]))
utl.plot_gridsearch(fig, fig.add_subplot(botleft_gs[1]))
utl.plot_eig_by_settings(df, fig, [
    fig.add_subplot(right_gs[i, j])
    for i in range(5) for j in range(2)
])
fig.tight_layout()
fig.savefig(f'dev/fig4.{OABfigfmt}', bbox_inches='tight')

# %% [markdown]
# # vaccine sequences

# %% [markdown]
# ## interact

# %%
utl.set_font_size(12)
fig = plt.figure(figsize=(12, 3), dpi=300)
ax = fig.subplots()
utl.plot_vaccine_interact_style('dev/sequential-full.csv', ylim=(0, 1.1), ax=ax)
ax.set_title('')
sns.despine(fig, top=True, left=True, right=True)
ax.grid(True, axis='y', zorder=0)
fig.tight_layout()
fig.savefig('./dev/interact_sequential.pdf')

fig = plt.figure(figsize=(12, 3), dpi=300)
ax = fig.subplots()
utl.plot_vaccine_interact_style('dev/showoff.csv', ylim=(0, 1.1), ax=ax)
ax.set_title('')
sns.despine(fig, top=True, left=True, right=True)
ax.grid(True, axis='y', zorder=0)
fig.tight_layout()
fig.savefig('./dev/interact_ours.pdf')

# %% [markdown]
# ## paper

# %%
seq_fname = 'dev/sequential-full.csv'
our_fname = 'dev/showoff.csv'

seq_log = utl.read_results(seq_fname)
seq_eig = np.mean(list(utl.effective_immunogen(seq_log)))
our_log = utl.read_results(our_fname)
our_eig = np.mean(list(utl.effective_immunogen(our_log)))

data = utl.read_bootstraps('cterm')

# %%
utl.set_font_size(5)

fig = plt.figure(figsize=(OABtextwidth, OABtextwidth * 37 / 100), dpi=OABdpi)

root_gs = mpl.gridspec.GridSpec(
    2, 2, figure=fig, width_ratios=[1, 3.5]
)

ax1 = fig.add_subplot(root_gs[0, 0])
ax2 = fig.add_subplot(root_gs[1, 0])
p4 = fig.add_subplot(root_gs[0, 1])
p1 = fig.add_subplot(root_gs[1, 1])

utl.plot_vaccine('dev/sequential-full.csv', ylim=(-2.5, 2.5), ax=p1)
utl.plot_vaccine('dev/showoff.csv', ylim=(-2.5, 2.5), ax=p4)

p1.set_title(f'(d) Sequential - Immunogenicity: {float(seq_log["immunogen"]):.3f} (Effective: {seq_eig:.3f})')
p4.set_title(f'(c) Simultaneous - Immunogenicity: {float(our_log["immunogen"]):.3f} (Effective: {our_eig:.3f})')
p1.set_xlabel(None)
p1.tick_params(axis='y', rotation=90)
p4.tick_params(axis='y', rotation=90)

# actual significance computed below
utl.plot_cleavages_by_location(ax1, data, 'scores', significance=[3, 3, 3])
ax1.set_title('(a) Cleavage score by location')
ax1.set_ylabel('Score')
ax1.tick_params(axis='y', rotation=90)

utl.plot_cleavages_by_location(ax2, data, 'netchop', significance=[3, 3, 0])
ax2.set_title('(b) Cleavage sites by location')
ax2.set_ylabel('Count')
ax2.tick_params(axis='y', rotation=90)

fig.tight_layout()
fig.savefig(f'dev/fig2.{OABfigfmt}', bbox_inches='tight', format=OABfigfmt)

# %% [markdown]
# ## analysis

# %%
print(f'Sequential design - Immunogenicity: {float(seq_log["immunogen"]):.3f} (Effective: {seq_eig:.3f})')
print(f'Our solution - Immunogenicity: {float(our_log["immunogen"]):.3f} (Effective: {our_eig:.3f})')

print('sequential recoveries', pd.Series(list(utl.sample_recovery(seq_log))).describe())
print('simultaneous recoveries', pd.Series(list(utl.sample_recovery(our_log))).describe())

# %%
ssf_log = utl.read_results('dev/showoff_strict.csv')
ssf_eig = np.mean(list(utl.effective_immunogen(ssf_log)))

print('strict immunogen:', ssf_log['immunogen'])
print('strict eff immunogen:', ssf_eig)

# %%
print('immunogenicity increase from strict to non strict:', 100 * (
    float(our_log['immunogen']) - float(ssf_log['immunogen'])
) / float(ssf_log['immunogen']))

# %%
print('effective immunogenicity increase from strict to non strict:',
      100 * (our_eig - ssf_eig) / ssf_eig)

# %% [markdown]
# ### netchop results

# %%
df = pd.DataFrame.from_dict({
    'terminals': [d['netchop_terminals'] for d in data],
    'epitopes': [d['netchop_epitopes'] for d in data],
    'spacers': [d['netchop_spacers'] for d in data],
    'method': [d['method'] for d in data],
})
df.groupby('method').apply(lambda g: g.describe().T)

# %%
print('effect size for terminals')
print('\t', (
    df[df.method == 'simultaneous'].terminals.mean() 
    - df[df.method == 'sequential'].terminals.mean()
) / df[df.method == 'simultaneous'].terminals.std())

print('effect size for epitopes')
print('\t', (
    df[df.method == 'simultaneous'].epitopes.mean() 
    - df[df.method == 'sequential'].epitopes.mean()
) / df[df.method == 'simultaneous'].epitopes.std())

print('effect size for spacers')
print('\t', (
    df[df.method == 'simultaneous'].spacers.mean() 
    - df[df.method == 'sequential'].spacers.mean()
) / df[df.method == 'simultaneous'].spacers.std())

# %%
ddf = pd.get_dummies(df)
def test_netchop_improvement(key):
    res = Poisson(
        ddf[key].values,
        add_constant(ddf.method_simultaneous)
    ).fit()
    print(res.summary())
    return res


# %%
rr = test_netchop_improvement('spacers')

# %%
rr = test_netchop_improvement('epitopes')

# %%
rr = test_netchop_improvement('terminals')

# %% [markdown]
# ### effective immunogenicity

# %%
eigdf = pd.DataFrame.from_dict({
    'eig': [d['eff_ig'] for d in data],
    'simultaneous': [int(d['method'] == 'simultaneous') for d in data],
})

print(eigdf.groupby('simultaneous').apply(lambda g: g.describe().T))

mean_diff = (
    eigdf[eigdf.simultaneous == 1].eig.mean() - 
    eigdf[eigdf.simultaneous == 0].eig.mean()
)
print('increase in eff. imm.:', mean_diff)
print('effect size of increase in eff. imm.:', 
      mean_diff / eigdf[eigdf.simultaneous == 0].eig.std())

# %%
rr = smf.ols(
    f'eig ~ 1 + simultaneous', data=eigdf
).fit()
print(rr.summary())
print('exact p-values', rr.pvalues)

# %% [markdown]
# ### cleavage scores

# %%
res_ours, res_seq = defaultdict(list), defaultdict(list)
for d in data:
    r = res_ours if d['method'] == 'simultaneous' else res_seq
    
    for k in ['terminals', 'epitopes', 'spacers']:
        r[k].extend(d[f'scores_{k}'])


# %%
def test_score_improvement(key):
    df = pd.DataFrame.from_dict({
        key: res_ours[key] + res_seq[key],
        'simultaneous': np.concatenate([
            np.ones(len(res_ours[key])),
            np.zeros(len(res_seq[key]))
        ])
    })
    
    print(df.groupby('simultaneous').apply(lambda g: g[key].describe().T))
    print('effect size: ', (
        df[df.simultaneous == 1][key].mean() - df[df.simultaneous == 0][key].mean()
    ) / df[df.simultaneous == 0][key].std())
    
    res = smf.ols(
        f'{key} ~ 1 + simultaneous', data=df
    ).fit()
    print(res.summary())
    print('exact pvalues', res.pvalues)
    return res


# %%
rr = test_score_improvement('epitopes')

# %%
rr = test_score_improvement('terminals')

# %%
rr = test_score_improvement('spacers')
