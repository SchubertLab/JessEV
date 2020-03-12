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
#\usepackage{layouts}
#\printinunitsof{in}\prntlen{\textwidth} \prntlen{\linewidth} 

OABtextwidth = 6.7261 #in
OABlinewidth = 3.2385 #in
OABdpi = 350

# %%
sns.set(context='paper', style='whitegrid')

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
    'res-comb-nc-': 'Ours',
    'res-cov-': 'Ours',
    'sequential': 'Seq.',
    'sequential-cov': 'Seq.',
}

utl.set_font_size(6)
fig = plt.figure(figsize=(OABtextwidth, OABtextwidth / 2), dpi=OABdpi)
((ax1, ax2, ax5), (ax3, ax4, ax6)) = fig.subplots(2, 3)

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
    title='(d) Pathogen Coverage',
    experiment_names=experiment_replacement
)

utl.plot_many_by_baseline(
    ax4, comp_cov, [mask_comp_us, mask_comp_seq], 'alle',
    xlabel='Prior cleavage probability',
    title='(e) HLA Coverage',
    experiment_names=experiment_replacement
)

ax5.semilogx(poi_cov.index, poi_cov['ge'], 'C0.-', 
             label='Effective pathogen coverage')
ax5.semilogx(poi.index, poi['ge'], 'C1.-',
             label='Effective immunogenicity')
ax5.semilogx(poi_alle.index, poi_alle['ge'], 'C2.-',
             label='Effective allele coverage')

ax6.loglog(improv.index, improv['res-comb-nc-'] / improv['sequential'], 'C0.-',
           label='Effective pathogen coverage')
ax6.loglog(improv_cov.index, improv_cov['res-cov-'] / improv_cov['sequential-cov'],
           'C1.-', label='Effective immunogenicity')
ax6.loglog(improv_alle.index, improv_alle['res-cov-'] / improv_alle['sequential-cov'],
           'C2.-', label='Effective allele coverage')

ax5.set_ylim(-0.05, 1.05)
ax6.set_ylim(0.8, 1.2e3)

ax5.set_xlim(0.02, 1.1)
ax6.set_xlim(0.02, 1.1)

ax5.set_title('(c) Probability of no-reduction')
ax6.set_title('(f) Expected improvement')

ax6.set_xlabel('Prior cleavage probability')

ax6.grid(True, axis='y', which='minor')

ax4.set_ylim(-1, 31)
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax5.legend(loc='lower left')
ax6.legend(loc='upper left')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.grid(True, 'minor', axis='x')
    ax.set_xticks([0.02, 0.1, 0.4, 1])
    ax.set_xticklabels(['0.02', '0.1', '0.4', '1.0'])

fig.tight_layout()
fig.savefig('dev/comparison_all_together.pdf')

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
utl.set_font_size(6)
fig = plt.figure(figsize=(OABtextwidth, OABtextwidth / 3), dpi=OABdpi)

root_gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

left_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=root_gs[0], wspace=0.3
)

right_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    3, 3, subplot_spec=root_gs[1], wspace=0.1, hspace=0.1
)

utl.plot_parameter_evolution(comparison, comp_cov, fig.add_subplot(left_gs[0]))
utl.plot_gridsearch(fig, fig.add_subplot(left_gs[1]))
utl.plot_eig_by_settings(df, fig, [
    fig.add_subplot(right_gs[i, j])
    for i in range(3) for j in range(3)
])

fig.tight_layout()
fig.savefig('dev/parameters.pdf')

# %%
summ_cov = df[
    df.experiment == 'res-cov-'
].groupby([
    'baseline', 'param_1', 'param_2'
]).apply(
    utl.summarize_experiment
).reset_index()

# %%
sorted(df.baseline.unique())

# %%
utl.set_font_size(6)
fig = plt.figure(figsize=(OABtextwidth, OABtextwidth / 2), dpi=OABdpi)

root_gs = mpl.gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1])

left_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=root_gs[0], wspace=0.4, hspace=0.5
)

right_gs = mpl.gridspec.GridSpecFromSubplotSpec(
    5, 2, subplot_spec=root_gs[1], wspace=0.075, hspace=0.075
)

utl.plot_ranked_parameters(
    df, summaries, 'mean_eig', fig.add_subplot(left_gs[0, 0]),
    xlim=(0.9, 25), xticks=[1, 2, 3, 4, 5, 10, 20], ylabel='Relative eff. immunog.'
)
utl.plot_ranked_parameters(
    df, summ_cov, 'mean_prot', fig.add_subplot(left_gs[0, 1]),
    xlim=(0.9, 7), xticks=[1, 2, 3, 4, 5, 6], ylabel='Relative pathogen coverage'
)
utl.plot_parameter_evolution(comparison, comp_cov, fig.add_subplot(left_gs[1, 0]))
utl.plot_gridsearch(fig, fig.add_subplot(left_gs[1, 1]))
utl.plot_eig_by_settings(df, fig, [
    fig.add_subplot(right_gs[i, j])
    for i in range(5) for j in range(2)
])

fig.tight_layout()

# %%
smm = df[df.experiment == 'res-cov-'].groupby(['baseline', 'param_1', 'param_2', 'param_3', 'param_4']).apply(utl.summarize_experiment).reset_index()
smm.head()

# %%
cmap2 = plt.get_cmap('tab20c')
    
for i, (b, g) in enumerate(smm[(smm.param_1==1.95) & (smm.param_2 == 1)].groupby('param_4')):
    plt.semilogx(g.baseline, g.mean_prot, label=str(b), c=cmap2(2 - i))
    
for i, (b, g) in enumerate(smm[(smm.param_1==1.95) & (smm.param_2 == 1.5)].groupby('param_4')):
    plt.semilogx(g.baseline, g.mean_prot, label=str(b), c=cmap2(7 - i))
    
plt.legend()

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

print(f'Sequential design - Immunogenicity: {float(seq_log["immunogen"]):.3f} (Effective: {seq_eig:.3f})')
print(f'Our solution - Immunogenicity: {float(our_log["immunogen"]):.3f} (Effective: {our_eig:.3f})')

print(pd.Series(list(utl.sample_recovery(seq_log))).describe())
print(pd.Series(list(utl.sample_recovery(our_log))).describe())

# %%
utl.set_font_size(6)

fig = plt.figure(figsize=(OABtextwidth, OABtextwidth / 3), dpi=OABdpi)
p1, p4 = fig.subplots(2, 1, gridspec_kw={
    'hspace': 0.9,
    'left': 0.08, 'right': 0.99, 'top': 0.925, 'bottom': 0.15
})

utl.plot_vaccine('dev/sequential-full.csv', ylim=(-2.5, 2.5), ax=p1)
utl.plot_vaccine('dev/showoff.csv', ylim=(-2.5, 2.5), ax=p4)

p1.set_title(f'(a) Sequential design - Immunogenicity: {float(seq_log["immunogen"]):.3f} (Effective: {seq_eig:.3f})')
p4.set_title(f'(b) Our solution - Immunogenicity: {float(our_log["immunogen"]):.3f} (Effective: {our_eig:.3f})')
p1.set_xlabel(None)

fig.savefig('./dev/comparison.pdf')

# %%