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
    'res-comb-nc-': 'Our results',
    'res-cov-': 'Our results',
    'sequential': 'Sequential',
    'sequential-cov': 'Sequential',
}

fig = plt.figure(figsize=(15, 6), dpi=300)
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

ax5.semilogx(poi_cov.index, poi_cov['ge'], 'C0o-', label='Effective pathogen coverage')
ax5.semilogx(poi.index, poi['ge'], 'C1o-', label='Effective immunogenicity')
ax5.semilogx(poi_alle.index, poi_alle['ge'], 'C2o-', label='Effective allele coverage')

ax6.loglog(improv.index, improv['res-comb-nc-'] / improv['sequential'], 'C0o-', label='Effective pathogen coverage')
ax6.loglog(improv_cov.index, improv_cov['res-cov-'] / improv_cov['sequential-cov'], 'C1o-', label='Effective immunogenicity')
ax6.loglog(improv_alle.index, improv_alle['res-cov-'] / improv_alle['sequential-cov'], 'C2o-', label='Effective allele coverage')

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
def plot_parameter_evolution(ax):
    xgrid = [1.4, 1.74, 1.8, 1.95, 2.25, 2.5, 2.75]
    epigrid = [-1, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]

    xs_eig, ys_eig, ps_eig = utl.find_parameter_trace(
        comparison[comparison.experiment == 'res-comb-nc-']
    )

    xs_cov, ys_cov, ps_cov = utl.find_parameter_trace(
        comp_cov[comp_cov.experiment == 'res-cov-']
    )

    ax.plot(ys_eig, xs_eig, 'o-', label='Effective immunogenicity')
    ax.plot(ys_cov, xs_cov, 'o-', label='Effective coverage')
    
    utl.annotate_axis(
        ax1, ['%.3f' % p for p in ps_eig],
        ys_eig, xs_eig, offsets=[(40, -18), (-5, 3), (-5, 3), (40, -18), (40, -10)]
    )

    utl.annotate_axis(
        ax, ['%.3f' % p for p in ps_cov],
        ys_cov, xs_cov, offsets=[(-5, -10), (-5, 3), (-5, 3)]
    )

    ax.set_ylabel('Minimum termini cleavage')
    ax.set_xlabel('Maximum inner epitope cleavage')
    ax.set_title('(a) Best parameters for different thresholds')
    ax.legend()


def plot_gridsearch(fig ,ax):
    mm = utl.plot_prefix_2(
        'res-comb-nc-',
        xlabel='Minimum termini cleavage',
        ylabel='Maximum inner epitope cleavage',
        title='(b) Immunogenicity for each parameter setting',
        #swapxy=True,
        ax=ax,
        imshow_kwargs={'cmap': 'viridis', 'vmin': 0.4, 'vmax': 1.3}
    )
    ax.grid(False)
    fig.colorbar(mm, ax=ax)


def plot_eig_by_settings(fig, axes):
    baselines = df.baseline.unique().tolist()
    termini_cleavage = sorted(set(df[df.experiment == 'res-comb-nc-'].param_1))
    vmin, vmax = min(termini_cleavage), max(termini_cleavage)
    cmap = plt.get_cmap('viridis')
    colors = [cmap.colors[int(200 * (b - vmin) / (vmax - vmin))] for b in termini_cleavage]

    df[(
        df.experiment == 'res-comb-nc-'
    ) & (
        df.baseline != 1000
    )].groupby([
        'baseline', 'experiment', 'param_1', 'param_2', 'param_3', 'param_4'
    ]).apply(
        utl.summarize_experiment
    ).reset_index().groupby([
        'baseline', 'experiment', 'param_1',
    ]).apply(lambda g: axes[
            bb.index(g.baseline[0])
        ].plot(
            g.param_2, g.mean_eig,
            c=colors[termini_cleavage.index(g.param_1[0])],
            label=f'tc: {g.param_1[0]:.1f}'
        ))

    for i, ax in enumerate(axes):
        if i == 1:
            ax.set_title(f'(c) Effective immunogenicity breakdown')

        ax.annotate(
            f'pc: {bb[i] / 1000:.3f}',
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


fig = plt.figure(figsize=(18, 5), dpi=300)

gs0 = mpl.gridspec.GridSpec(1, 3, figure=fig)
gs00 = mpl.gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs0[2])

ax1 = fig.add_subplot(gs0[0])
ax2 = fig.add_subplot(gs0[1])
axes = [
    fig.add_subplot(gs00[i, j])
    for i in range(3) for j in range(3)
]

plot_parameter_evolution(ax1)
plot_gridsearch(fig, ax2)
plot_eig_by_settings(fig, axes)

fig.tight_layout()
fig.savefig('dev/parameters.pdf')

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
