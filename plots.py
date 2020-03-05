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
import matplotlib.patches as mpatches
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import plot_utils as utl

# use LaTeX fonts in the plot
# https://ercanozturk.org/2017/12/16/python-matplotlib-plots-in-latex/
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

# %%
sns.set()

# %%
df = utl.compute_mc_experiments(baselines=[
    0.025, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300, 0.500, 0.650, 0.1000
])
# convert to int so we can reliaby filter on them
df.baseline = 1000 * df.baseline
df.baseline = df.baseline.astype(np.int)
df = df.fillna(-100)  # stupid pandas discards na's in groupby's

df

# %% [markdown]
# # comparison

# %%
comparison = utl.compare_experiments(df, 'mean_eig')  # best effective immunogen
comp_cov = utl.compare_experiments(df, 'mean_prot')   # best coverage

# %%
fig = plt.figure(figsize=(12, 4), dpi=96)
(ax1, ax2, ax3) = fig.subplots(1, 3)

experiment_replacement = {
    'res-comb-nc-': 'Our results',
    'sequential': 'Sequential',
}

mask_seq = (comparison.experiment == 'sequential') & (comparison.baseline.isin(bases))
mask_us = (comparison.experiment == 'res-comb-nc-') & (comparison.baseline.isin(bases))

utl.plot_many_by_baseline(
    ax1, comparison, [mask_us, mask_seq], 'eig',
    title='(a) Effective Immunogenicity',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax2, comparison, [mask_us, mask_seq], 'rec',
    title='(b) Recovered epitopes',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax3, comparison, [mask_us, mask_seq], 'len',
    title='(c) Average fragment length',
    experiment_names=experiment_replacement
)

ax3.legend()

fig.tight_layout()
fig.savefig('./dev/effective.pdf')

# %% [markdown]
# # comparison on coverage

# %%
experiment_replacement['sequential-cov'] = 'Sequential'
experiment_replacement['res-cov-'] = 'Our Results'

fig = plt.figure(figsize=(10, 4), dpi=300)
ax1, ax2 = fig.subplots(1, 2)

mask_comp_us = (comp_cov.experiment == 'res-cov-') & (comp_cov.baseline.isin(bases))
mask_comp_seq = (comp_cov.experiment == 'sequential-cov') & (comp_cov.baseline.isin(bases))

utl.plot_many_by_baseline(
    ax1, comp_cov, [mask_comp_us, mask_comp_seq], 'prot',
    xlabel='Prior cleavage probability',
    title='(a) Pathogen Coverage',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax2, comp_cov, [mask_comp_us, mask_comp_seq], 'alle',
    xlabel='Prior cleavage probability',
    title='(a) HLA Coverage',
    experiment_names=experiment_replacement
)

ax2.set_ylim(-1, 31)
ax1.legend(loc='upper right')

fig.tight_layout()

fig.savefig('dev/coverage.pdf')

# %% [markdown]
# # comparison together

# %%
fig = plt.figure(figsize=(10, 6), dpi=300)
((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)

utl.plot_many_by_baseline(
    ax1, comparison, [mask_us, mask_seq], 'rec',
    title='(a) Recovered epitopes',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax2, comparison, [mask_us, mask_seq], 'eig',
    title='(b) Effective Immunogenicity',
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
    title='(c) HLA Coverage',
    experiment_names=experiment_replacement
)

ax4.set_ylim(-1, 31)
ax1.legend(loc='upper right')

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True, 'minor', axis='x')

fig.tight_layout()
fig.savefig('dev/comparison_all.pdf')

# %% [markdown]
# # probability of improvement and expected improvement

# %%
improv = utl.compute_expected_improvement(
    comparison, 'mean_eig', 'sequential', 'res-comb-nc-'
)
poi = utl.compute_probability_of_improvement(
    df, comparison, 'effective_immunogen', 'sequential', 'res-comb-nc-', 5000
)

# %%
fig = plt.figure(figsize=(10, 4), dpi=300)
ax1, ax2 = fig.subplots(1, 2)
ax1.plot(poi['ge'].iloc[1:], 'o-', label='Not worse')
ax1.plot(poi['eq'].iloc[1:], 'o-', label='Both zero')
ax1.legend()

ax1.set_xlabel('Prior cleavage probability')
ax1.set_ylabel('Probability')
ax1.set_title('(a) Probability of improvement')

ax2.semilogy(improv['res-comb-nc-'] / improv['sequential'], 'o-')
ax2.grid(True, 'minor')
#ax2.plot([0, 0.7], [1, 1], 'C2--')
ax2.set_title('(b) Expected improvement')
ax2.set_xlabel('Prior cleavage probability')

fig.tight_layout()
fig.savefig('dev/improvement.pdf')

# %% [markdown]
# # probability of improvement/expected improvement of all metrics

# %%
improv_cov = utl.compute_expected_improvement(
    comp_cov, 'mean_prot', 'sequential-cov', 'res-cov-'
)
improv_alle = utl.compute_expected_improvement(
    comp_cov, 'mean_alle', 'sequential-cov', 'res-cov-'
)

poi_cov = utl.compute_probability_of_improvement(
    df, comp_cov, 'proteins', 'sequential-cov', 'res-cov-', 100
)
poi_alle = utl.compute_probability_of_improvement(
    df, comp_cov, 'alleles', 'sequential-cov', 'res-cov-', 100
)

# %%
fig = plt.figure(figsize=(10, 4), dpi=96)
ax1, ax2 = fig.subplots(1, 2)

ax1.semilogx(poi_cov.index, poi_cov['ge'], 'C0o-', label='Effective pathogen coverage')
ax1.semilogx(poi.index, poi['ge'], 'C1o-', label='Effective immunogenicity')
ax1.semilogx(poi_alle.index, poi_alle['ge'], 'C2o-', label='Effective allele coverage')

ax1.set_ylim(-0.05, 1.05)
ax1.legend(loc='lower left')
ax1.set_title('(a) Probability of no-reduction')
ax1.set_xlabel('Prior cleavage probability')

ax2.loglog(improv.index, improv['res-comb-nc-'] / improv['sequential'], 'C0o-')
ax2.loglog(improv_cov.index, improv_cov['res-cov-'] / improv_cov['sequential-cov'], 'C1o-')
ax2.loglog(improv_alle.index, improv_alle['res-cov-'] / improv_alle['sequential-cov'], 'C2o-')

ax2.set_ylim(0.8, 1.2e3)
ax1.set_xlim(0.02, 0.8)
ax2.set_xlim(0.02, 0.8)
ax2.set_title('(b) Expected improvement')
ax2.set_xlabel('Prior cleavage probability')

ax1.grid(True, axis='x', which='minor')
ax2.grid(True, axis='y', which='minor')
ax2.grid(True, axis='x', which='minor')

fig.tight_layout()
fig.savefig('dev/improvement_ig_cov_alle.pdf')

# %% [markdown]
# # evolution of best parameter

# %%
xgrid = [1.4, 1.74, 1.8, 1.95, 2.25, 2.5, 2.75]
epigrid = [-1, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]

done = set()
xs, ys, ps, cs = [], [], [], []
for i, row in comparison[comparison.experiment == 'res-comb-nc-'].iterrows():
    if i == 0:
        continue
        
    i1, i2 = xgrid.index(row.param_1), epigrid.index(row.param_2)
    if (i1, i2) in done:
        continue
        
    done.add((i1, i2))
    xs.append(i1)
    ys.append(i2)
    ps.append(row.baseline / 1000)
    
    cmap = plt.get_cmap('viridis')
    cs.append(cmap.colors[int(255 * row.mean_eig / 0.8)])

fig = plt.figure()
ax = fig.subplots()

ax.plot(ys, xs, 'o-')
#ax.scatter(ys, xs, c=cs)

offsets = [
    (-3, 0),
    (-5, 3),
    (40, -7),
    (-5, -7),
    (-5, -7),
    (-5, 3),
    (25, -20),
    (30, 3),
]
for i in range(len(ps)):
    ax.annotate('%.3f' % ps[i],
            xy=(ys[i], xs[i]), xytext=offsets[i],
            #color='red',
            textcoords="offset points",
            ha='right', va='bottom')

ax.set_xticks(range(len(epigrid)))
ax.set_xticklabels(['%.1f' % x for x in sorted(set(epigrid))])
ax.set_yticks(range(len(xgrid)))
ax.set_yticklabels(['%.2f' % x for x in sorted(set(xgrid))])
ax.set_ylim(0.5, 5.5)
ax.set_ylabel('Termini cleavage')
ax.set_xlabel('Internal epitope cleavage')
fig.tight_layout()

# %% [markdown]
# # vaccine sequences

# %% [markdown]
# ## interact

# %%
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

print(f'Sequential design - Immunogenicity: {float(seq_log["immunogen"]):.3f} (Effective: {seq_eig:.3f})')
print(f'Our solution - Immunogenicity: {float(our_log["immunogen"]):.3f} (Effective: {our_eig:.3f})')

print(pd.Series(list(utl.sample_recovery(seq_log))).describe())
print(pd.Series(list(utl.sample_recovery(our_log))).describe())

# %%
fig = plt.figure(figsize=(12, 6), dpi=300)
p1, p4 = fig.subplots(2, 1, gridspec_kw={
    'hspace': 0.7,
    #'height_ratios': [1, 0.1, 1, 1],
    'left': 0.05, 'right': 0.99, 'top': 0.925, 'bottom': 0.15
})

utl.plot_vaccine('dev/sequential-full.csv', ylim=(-2.5, 2.5), ax=p1)
utl.plot_vaccine('dev/showoff.csv', ylim=(-2.5, 2.5), ax=p4)

p1.set_title(f'(a) Sequential design - Immunogenicity: {float(seq_log["immunogen"]):.3f} (Effective: {seq_eig:.3f})')
p4.set_title(f'(b) Our solution - Immunogenicity: {float(our_log["immunogen"]):.3f} (Effective: {our_eig:.3f})')

fig.savefig('./dev/comparison.pdf')
