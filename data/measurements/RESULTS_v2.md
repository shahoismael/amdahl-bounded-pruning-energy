# v2 Results — corrected harness, four datasets

All numbers from `code/v2/`. v1 results are superseded.

---

## 1. Headline

A theoretical reduction of ~23% in multiply-accumulate operations produced a
measured energy saving of **9.4%** — about **40% of what the metric predicted**.
The shortfall is significant on every dataset independently.

| Dataset | Theory (MAC) | Measured saving | 95% CI | Realised | p (non-inferiority) | Holm |
|---|---|---|---|---|---|---|
| MNIST | 22.26% | **+10.56%** | 8.30 – 12.82 | 47% | 1.3×10⁻⁵ | 5.0×10⁻⁵ ✔ |
| Fashion-MNIST | 24.44% | **+13.44%** | 8.93 – 17.95 | 55% | 1.7×10⁻³ | 3.3×10⁻³ ✔ |
| MIT-BIH | 25.51% | **+6.27%** | 2.77 – 9.78 | 25% | 2.7×10⁻⁵ | 8.1×10⁻⁵ ✔ |
| ChestX-ray14 | 22.23% | **+8.04%** | −0.19 – 16.28 | 36% | 2.3×10⁻³ | 3.3×10⁻³ ✔ |
| **Pooled (n=23)** | 23.60% | **+9.41%** | 7.15 – 11.67 | **40%** | **4.0×10⁻¹²** | — |

**All four reject after Holm correction.**

Clustered estimate (dataset as random intercept, 4 clusters):
grand mean **+9.58%**, 95% CI **+4.62 to +14.54**, t(3) = −9.00, **p = 1.4×10⁻³**.
ICC = 0.184; between-dataset SD 2.28, within-dataset SD 4.81. The cluster-aware
SE (1.558) is 43% larger than the naive SE (1.088) — the pooled row above is
anticonservative and the clustered row is the one to report.

Order effect across all 23 blocks: r = +0.284, p = 0.190. The interleaved,
order-randomised protocol removed the confound that dominated v1.

---

## 2. The saving is time, not watts

MNIST, mean CPU package power by arm:

| arm | power |
|---|---|
| baseline | 23.352 W |
| baseline_ft | 23.152 W |
| pruned | 23.083 W |

Spread 1.2%. Latency fell 10.23%, energy fell 10.56%, power moved 0.3%.

**Pruning does not reduce power draw. It reduces energy only by finishing
sooner.** The processor works just as hard, for less time.

---

## 3. Batch size decides whether pruning does anything at all

| batch | baseline µs/inf | pruned µs/inf | reduction |
|---|---|---|---|
| **1** | 1297.3 | 1299.5 | **−0.12%** (CI −3.71 to +3.47, p = 0.93) |
| 8 | 307.79 | 289.91 | 5.74% |
| 32 | 239.24 | 218.33 | 8.70% |
| 128 | 210.65 | 182.50 | 13.39% |
| 512 | 200.40 | 177.09 | 11.65% |

Compute-bound regime (bs ≥ 32), n = 9: **11.25%** (CI 8.91–13.58),
vs zero p = 6.6×10⁻⁶, 46% of theory.

At batch size 1 framework dispatch dominates and pruning delivers **nothing**.
v1 measured at batch size 1. That is the entire explanation for the v1 null.

bs = 1 replicates 1–3 were discarded as warm-up contamination (pruned network
reported 4.5 ms/inference against a 1.2 ms steady state). The sweep now runs
burn-in replicates and scales warm-up by batch size.

---

## 4. The compute-matched control separates two effects v1 conflated

Balanced accuracy (mean per-class recall).

| Dataset | baseline | baseline_ft | pruned | Δ training | Δ pruning |
|---|---|---|---|---|---|
| MNIST | 98.86% | 98.76% | 97.31% | −0.10 pp | −1.45 pp |
| Fashion-MNIST | 90.19% | 89.66% | 89.30% | −0.53 pp | −0.36 pp |
| MIT-BIH | 79.23% | 62.97% | 44.89% | −16.27 pp | −18.08 pp |
| ChestX-ray14 | 58.83% | 55.04% | 51.18% | −3.79 pp | −3.86 pp |

v1 reported "accuracy retained within 1–2 pp" while comparing a pruned network
that had also received two extra epochs against a baseline that had not. The
control separates them.

---

## 5. Two data-splitting defects found and fixed

**MIT-BIH.** v1 shuffled at beat level, so beats from the same patient appeared
in train and test. Consecutive beats from one ECG share electrode placement,
morphology and noise, so a classifier scores well by recognising the patient.
Fixed to record-level splitting, stratified by abnormal fraction.

A first attempt with an unstratified record split put records 101 and 117 in
test, giving 3392 Normal against 17 Abnormal (99.5%). A constant "Normal"
predictor scored 99.09%, and the pruned model appeared to *gain* 34.79 pp.
Records are now dealt round-robin by abnormal fraction.

Under the honest split, training accuracy reached 99% while validation sat at
40–50%. **This architecture does not solve the inter-patient task**; pruned
balanced accuracy (44.89%) is below chance. This must be stated plainly. The
energy audit is unaffected — energy depends on network shape, not on labels.

**ChestX-ray14.** v1 used `splitEachLabel(..., 'randomized')`, which splits at
image level; one PatientID has many images. Fixed to patient-level splitting.
Images are also now decoded once into memory rather than re-read from disk
inside the timed window, where PNG decode was being charged to "inference
energy".

---

## 6. Recovery of the v1 energy figures

`measuredWatts` was `NaN` in all 14 v1 result files. Using the stored window
timestamps plus `Log_to_CSV.csv`, 12 of 14 runs were recovered
(`results/recovered_power.csv`). Two predate the log.

Split by whether the CPU was throttled during the window:

| | n | mean saving | savings seen in |
|---|---|---|---|
| Clean runs | 8 | **−6.00%** | 1 / 8 |
| Throttled runs | 4 | **+21.68%** | **4 / 4** |

Correlation between measured slowdown and apparent saving: **r = −0.68, p = 0.016**.

Every apparent saving in v1 came from a run where the baseline window caught the
CPU in a slow state. v1 always measured baseline first, so ordering was
perfectly confounded with condition.

---

## 7. A stalled measurement nearly destroyed the Fashion-MNIST result

Fashion-MNIST block 4, pruned arm, ran **324 s against a 60 s budget** at one
tenth normal throughput — something else took the CPU. Its energy figure was 6x
too high, and on its own it swung the six-block mean from **+13.44% to −69.81%**
with an SD of 204.

The harness now records `overrunRatio`, flags any measurement exceeding 1.20x
its budget, and `analyze_energy_v2` drops the whole block by rule.

---

## 8. What the paper should claim

Not "pruning does not save energy" — it does, and the v1 data could not have
shown otherwise.

> A theoretical reduction of 22–26% in multiply-accumulate operations produced a
> measured energy saving of 9.4% across four datasets and 23 interleaved
> measurement blocks (clustered estimate 9.6%, 95% CI 4.6–14.5) — roughly 40% of
> the predicted value, and significantly below it on every dataset after Holm
> correction (pooled t = −9.00, p = 1.4×10⁻³). Mean processor power was unchanged
> across all arms (spread 1.2%); the saving arises entirely from reduced
> execution time. At batch size 1, where framework overhead dominates, the same
> compression delivered no measurable saving at all (−0.1%, p = 0.93).
> Theoretical compression metrics therefore overstate deployed energy benefit by
> roughly a factor of two and a half, and predict a benefit that does not exist
> in the low-batch regime typical of latency-sensitive edge inference.

---

## Files

| File | Contents |
|---|---|
| `results_v2/measurements_*.csv` | Raw measurements, 6 blocks × 3 arms per dataset |
| `results_v2/run_*.mat` | Full run state: nets, FLOP reports, diagnostics |
| `results/recovered_power.csv` | 12 recovered v1 runs |
| `results/batch_sweep.csv` | Batch-size sweep |
| `figs_v2/Figure1..4` | PNG + vector PDF |

## Outstanding

- `fitlme` unavailable (no Statistics Toolbox). Replaced by a closed-form
  random-intercept estimate in `analyze_energy_v2`, verified against
  statsmodels REML to four decimals on every component.
- HWiNFO polling remained at 2000 ms for MNIST and Fashion-MNIST (30 samples per
  60 s window). MIT-BIH and ChestX-ray14 onward should be checked; report the
  interval actually used per dataset.
