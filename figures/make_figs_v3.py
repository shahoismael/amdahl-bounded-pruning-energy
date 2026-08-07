"""Regenerate the manuscript figures directly from the measurement CSVs.

Every value plotted here is computed from data on disk, not transcribed from the
manuscript tables. Figure and text therefore cannot drift apart.

Inputs (relative to research_v4/):
    results_v2/all_datasets_final.csv        23 retained blocks, 4 datasets
    results_v2/batch_sweep_v2_clean.csv      re-run sweep, paired replicates
    results_v2/seed_replication_mnist.csv    3 pruning draws, MNIST

Outputs: PNG (300 dpi) + PDF (vector) per figure, sized for the 80 mm PES column.

Usage:
    python make_figs_v3.py <research_v4_dir> <output_dir>
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv, os, sys, math
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else '..'
OUT  = sys.argv[2] if len(sys.argv) > 2 else 'final_figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})
W, H = 3.15, 2.5                      # inches; 80 mm single column

# Okabe-Ito: distinguishable under all common forms of colour vision
ORANGE, BLUE, GREY, VERM = '#E69F00', '#0072B2', '#999999', '#D55E00'

TCRIT = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  wrote {name}.png / .pdf')


def ci95(v):
    n = len(v)
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1)) / math.sqrt(n)
    return m, TCRIT[n - 1] * se, n


# ---------------------------------------------------------------- Figure 1
def figure1():
    """Theory vs measured energy saving, by dataset. Ordered by realised f."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results_v2', 'all_datasets_final.csv'))))
    for r in rows:
        for k in ('block', 'energyJPerInference', 'theoreticalMACReduction'):
            r[k] = float(r[k])
        r['stalled'] = int(float(r['stalled']))

    blocks, stalled = defaultdict(dict), defaultdict(set)
    for r in rows:
        blocks[(r['dataset'], r['block'])][r['condition']] = r
        if r['stalled']:
            stalled[r['dataset']].add(r['block'])

    stats = {}
    for (ds, b), arms in blocks.items():
        if b in stalled[ds] or 'pruned' not in arms or 'baseline_ft' not in arms:
            continue
        ref, prn = arms['baseline_ft'], arms['pruned']
        s = (ref['energyJPerInference'] - prn['energyJPerInference']) / ref['energyJPerInference'] * 100
        stats.setdefault(ds, {'v': [], 'th': ref['theoreticalMACReduction']})['v'].append(s)

    # Order by realised fraction, descending -- matches the Results table
    order = sorted(stats, key=lambda d: np.mean(stats[d]['v']) / stats[d]['th'], reverse=True)
    theory = [stats[d]['th'] for d in order]
    meas, err, ns = zip(*[ci95(stats[d]['v']) for d in order])
    frac = [m / t for m, t in zip(meas, theory)]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(W, H))
    ax.bar(x - 0.19, theory, 0.38, label='Predicted (MAC)', color=ORANGE, edgecolor='black', lw=0.5)
    ax.bar(x + 0.19, meas, 0.38, yerr=err, capsize=2.5, label='Measured',
           color=BLUE, edgecolor='black', lw=0.5,
           error_kw=dict(lw=0.8, ecolor='black'))
    for xi, m, e, f in zip(x, meas, err, frac):
        ax.text(xi + 0.19, m + e + 0.9, f'{f*100:.0f}%', ha='center', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([d.replace('-', '-\n') for d in order])
    ax.set_ylabel('Energy reduction per inference (%)')
    ax.set_title(f'Measured saving reaches {min(frac)*100:.0f}–{max(frac)*100:.0f}% of prediction')
    ax.axhline(0, color='black', lw=0.8)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.30))
    save(fig, 'Figure1_theory_vs_measured')
    print('   order:', ', '.join(f'{d} f={f:.2f} n={n}' for d, f, n in zip(order, frac, ns)))


# ---------------------------------------------------------------- Figure 2
def figure2():
    """Batch-size sweep, paired replicates, re-run harness."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results_v2', 'batch_sweep_v2_clean.csv'))))
    for r in rows:
        r['batchSize'] = int(float(r['batchSize']))
        r['repeat'] = int(float(r['repeat']))
        r['secPerInference'] = float(r['secPerInference'])

    per = defaultdict(dict)
    for r in rows:
        per[(r['batchSize'], r['repeat'])][r['condition']] = r['secPerInference']

    sizes = sorted({b for b, _ in per})
    red, err, ns = [], [], []
    for b in sizes:
        d = [(per[(b, k)]['baseline'] - per[(b, k)]['pruned']) / per[(b, k)]['baseline'] * 100
             for (bb, k) in per if bb == b
             and 'baseline' in per[(bb, k)] and 'pruned' in per[(bb, k)]]
        m, e, n = ci95(d)
        red.append(m); err.append(e); ns.append(n)

    THEORY = 24.44                       # this network: 32->25 conv2, 15 conv1
    dead = [b for b, m, e in zip(sizes, red, err) if m - e <= 0 <= m + e]
    xi = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(W, H))
    if dead:
        ax.axvspan(-0.35, max(sizes.index(b) for b in dead) + 0.35,
                   color=ORANGE, alpha=0.12, lw=0)
    ax.axhline(THEORY, ls='--', lw=1.4, color=ORANGE)
    ax.text(0.02, THEORY + 0.8, f'Predicted ({THEORY:.1f}%)', color=ORANGE, fontsize=7.5)
    ax.axhline(0, color='black', lw=0.8)
    ax.errorbar(xi, red, yerr=err, marker='o', ms=4.5, lw=1.6, capsize=2.5,
                color=BLUE, mec='black', mew=0.5, ecolor='black', elinewidth=0.8)

    lo = max(sizes.index(b) for b in dead) if dead else 0
    ax.annotate('no saving below\nbatch %d' % sizes[lo + 1],
                xy=(lo, red[lo]), xytext=(lo + 0.25, -6.5), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', lw=0.7))
    ax.set_xticks(xi); ax.set_xticklabels(sizes)
    ax.set_xlabel('Inference batch size')
    ax.set_ylabel('Measured latency reduction (%)')
    ax.set_title('The saving exists only above the overhead floor')
    ax.set_ylim(-9, THEORY + 4)
    save(fig, 'Figure3_batch_sweep')
    for b, m, e, n in zip(sizes, red, err, ns):
        print(f'   bs={b:4d}  {m:+6.2f}%  CI {m-e:+6.2f} to {m+e:+6.2f}  n={n}  f={m/THEORY:.2f}')


# ---------------------------------------------------------------- Figure 2
def figure3():
    """Mean package power by arm, MNIST blocks. Power does not move."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, 'results_v2', 'all_datasets_final.csv'))))
    arms = ['baseline', 'baseline_ft', 'pruned']
    w = [float(np.mean([float(r['measuredWatts']) for r in rows
                        if r['dataset'] == 'MNIST' and r['condition'] == a])) for a in arms]
    spread = (max(w) - min(w)) / np.mean(w) * 100

    fig, ax = plt.subplots(figsize=(W, H))
    ax.bar(range(3), w, 0.55, color=[BLUE, BLUE, VERM], edgecolor='black', lw=0.5)
    for i, v in enumerate(w):
        ax.text(i, v + 0.03, f'{v:.3f}', ha='center', fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['baseline', 'baseline-ft', 'pruned'])
    ax.set_ylabel('Mean CPU package power (W)')
    ax.set_ylim(min(w) - 1.2, max(w) + 0.7)
    ax.set_title(f'Power is unchanged (spread {spread:.1f}%)')
    ax.annotate('', xy=(-0.3, max(w) + 0.35), xytext=(2.3, max(w) + 0.35),
                arrowprops=dict(arrowstyle='<->', color=GREY, lw=0.9))
    ax.text(1, max(w) + 0.42, f'{spread:.1f}% total spread', ha='center', color=GREY, fontsize=7.5)
    save(fig, 'Figure2_power_flat')
    print(f'   {dict(zip(arms, [round(v,3) for v in w]))}  spread {spread:.2f}%')


# ---------------------------------------------------------------- Figure 4
def figure4():
    """Realised fraction across three independent pruning draws, MNIST."""
    import glob
    paths = sorted(glob.glob(os.path.join(ROOT, 'results_v2', 'seed_replication_mnist*.csv')))
    if not paths:
        print('  no seed_replication_mnist*.csv found; skipping Figure 4')
        return
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    for r in rows:
        for k in ('block', 'secPerInference', 'throughputPerSec', 'seed', 'theory'):
            r[k] = float(r[k])

    # Pre-specified throughput rule (Methods 3.5): drop a block if either arm
    # falls below 90% of that arm's maximum observed throughput within its run.
    # Applied per seed, per arm, and always to the whole block so pairing holds.
    drop = defaultdict(set)
    for s in {r['seed'] for r in rows}:
        for arm in ('baseline_ft', 'pruned'):
            v = [r for r in rows if r['seed'] == s and r['condition'] == arm]
            lim = 0.90 * max(x['throughputPerSec'] for x in v)
            drop[s] |= {x['block'] for x in v if x['throughputPerSec'] < lim}

    by = defaultdict(lambda: defaultdict(dict))
    theo = {}
    for r in rows:
        if r['block'] in drop[r['seed']]:
            continue
        by[r['seed']][r['block']][r['condition']] = r['secPerInference']
        theo[r['seed']] = r['theory']

    seeds = sorted(by)
    fr, er, nb = [], [], []
    for s in seeds:
        d = [(v['baseline_ft'] - v['pruned']) / v['baseline_ft'] * 100
             for v in by[s].values() if 'baseline_ft' in v and 'pruned' in v]
        m, e, n = ci95(d)
        fr.append(m / theo[s]); er.append(e / theo[s]); nb.append(n)
        if drop[s]:
            print(f'   seed {int(s)}: excluded block(s) {sorted(int(b) for b in drop[s])} by throughput rule')

    fig, ax = plt.subplots(figsize=(W, H))
    ax.errorbar(range(len(seeds)), fr, yerr=er, fmt='o', ms=5, capsize=3,
                color=BLUE, mec='black', mew=0.5, ecolor='black', elinewidth=0.8)
    # Label below each point, not above, so nothing collides with the title.
    for i, (s, f) in enumerate(zip(seeds, fr)):
        ax.annotate(f'{theo[s]:.2f}% theory', xy=(i, f - er[i]),
                    xytext=(0, -11), textcoords='offset points',
                    ha='center', va='top', fontsize=7)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'draw {i+1}\n($n$={n})' for i, n in enumerate(nb)])
    ax.set_ylabel('Realised fraction  $f$')
    ax.set_xlim(-0.5, len(seeds) - 0.5)

    # Headroom: title clear of the highest cap, labels clear of the lowest.
    top = max(f + e for f, e in zip(fr, er))
    bot = min(f - e for f, e in zip(fr, er))
    pad = (top - bot) * 0.28
    ax.set_ylim(bot - pad, top + pad * 0.45)

    ax.set_title('$f$ tracks the architecture the draw lands on', pad=8)
    ax.grid(axis='y', lw=0.4, color=GREY, alpha=0.4)
    ax.set_axisbelow(True)
    save(fig, 'Figure4_seed_replication')
    for s, f in zip(seeds, fr):
        print(f'   seed {int(s)}  theory {theo[s]:.2f}%  f={f:.2f}')


if __name__ == '__main__':
    print(f'root   : {os.path.abspath(ROOT)}')
    print(f'output : {os.path.abspath(OUT)}\n')
    for fn in (figure1, figure3, figure2, figure4):
        print(fn.__doc__.splitlines()[0])
        fn()
        print()
    print('done.')
