# Methods Revision Memo — "The Green Illusion" (v2)

**Status:** seven defects identified in the v1 code and analysis. Five are fixable
from data already on disk. Two require re-measurement. New code is in `code/v2/`.

**The important thing first:** the re-analysis does not weaken the paper. It makes
the central claim *stronger*, because it replaces an argument from absence of
evidence with a decisive positive test — and it surfaces a finding sitting
unreported in the saved run files.

---

## Part 1 — What the existing data already shows

### 1.1 The pruned network is slower, not faster

`results/*.mat` records `numRuns` and `avgInferenceTimeSec` for both arms of every
saved run. Those survived even though `measuredWatts` did not. Recovering
throughput from them gives this:

| | n | mean latency change | median | pruned slower in |
|---|---|---|---|---|
| **Clean runs** (both arms ≥80% of nominal throughput) | 9 | **+3.23%** | +2.20% | 6 / 9 |
| **Throttled runs** (either arm <80% of nominal) | 5 | **−19.57%** | −20.46% | **0 / 5** |

Positive = pruned slower. The separation is total: **not one throttled run showed
the pruned arm slower, and it holds at every threshold from 0.75 to 0.95.**

This is the signature of an ordering artefact. v1 always measured baseline first and
pruned second. When the CPU was throttled during the baseline window and recovered
before the pruned window, the pruned arm inherited a spurious advantage. Every
"pruning saved energy" run in the dataset is one of those runs.

In the clean runs, a network with 24.4% fewer MACs runs **2–4% slower**. That is the
paper's most interesting result and it is currently unreported.

### 1.2 The energy figures are not reproducible from the repository

`measuredWatts` is `NaN` in **all 14** saved result files. The wattages behind
Tables 6 and 7 exist only in whatever spreadsheet they were typed into. Section 3.5
already discloses the MIT-BIH run-3 log-matching incident; that is the same failure
mode, caught once.

### 1.3 Six of the twenty claimed runs have no saved file

| Dataset | Saved `.mat` files | Claimed | Missing |
|---|---|---|---|
| MNIST | 1 | 5 | **4** |
| Fashion-MNIST | 3 | 5 | **2** |
| MIT-BIH | 5 | 5 | 0 |
| ChestX-ray14 | 5 | 5 | 0 |
| **Total** | **14** | **20** | **6** |

Either locate them before submission, or reduce the reported n. A repository that
does not contain the runs the paper counts is a retraction risk.

Reproduce all three findings with `audit_v1_runs('../../results')`.

---

## Part 2 — The statistical correction (no re-measurement needed)

### 2.1 The v1 test asks the wrong question

v1 ran a one-sample t-test against zero (p = 0.76) and concluded there is no energy
saving. Failing to reject a null is not evidence for it. The v1 CI, [−8.55, +6.37],
comfortably contains a genuine 6% saving. As analysed, the data cannot rule one out —
so the abstract's "the two are separate claims, not one" is not supported by the test
that was run.

### 2.2 The right question has a proper test

The paper does not actually care whether the saving is zero. It cares whether the
saving is **as large as the theoretical metric predicts**. That is a non-inferiority
question:

> H₀: true saving ≥ theoretical predicted saving
> H₁: true saving < theoretical predicted saving

Rejecting H₀ is a *positive finding*. Computing the theoretical prediction properly —
by MACs, which is what the field means by FLOPs — gives **24.44%** for the 28×28
datasets (24.47% for ChestX-ray14). Testing the v1 numbers against it:

| Group | n | measured saving | SD | t | df | p (one-sided) | Holm |
|---|---|---|---|---|---|---|---|
| **All pooled** | 20 | +1.09% | 15.94 | **−6.551** | 19 | **1.4×10⁻⁶** | — |
| ChestX-ray14 | 5 | −0.06% | 6.14 | −8.922 | 4 | 4.4×10⁻⁴ | **1.7×10⁻³ ✔** |
| MNIST | 5 | −9.48% | 8.93 | −8.494 | 4 | 5.3×10⁻⁴ | **1.6×10⁻³ ✔** |
| Fashion-MNIST | 5 | +3.83% | 19.49 | −2.365 | 4 | 3.9×10⁻² | 7.7×10⁻² |
| MIT-BIH | 5 | +10.08% | 21.38 | −1.502 | 4 | 1.0×10⁻¹ | 1.0×10⁻¹ |

**The pooled data rejects the theoretical prediction at p = 1.4×10⁻⁶.** Two of four
datasets survive Holm correction individually; the other two are underpowered at n = 5,
which should be stated rather than hidden.

This single change converts the paper from "we found nothing" to "we reject the
field's standard efficiency metric as a predictor of measured energy, decisively."

### 2.3 Three secondary statistical fixes

- **Clustering.** Pooling 20 runs with df = 19 treats runs nested within a dataset as
  independent. They share a training pipeline, a session and a thermal environment.
  `analyze_energy_v2` fits `energySavingPct ~ 1 + (1|dataset)` when the Statistics
  Toolbox is present.
- **Multiplicity.** Four dataset-level tests with no correction, and MNIST highlighted
  at p = 0.077. Holm–Bonferroni is now applied and reported.
- **Detectable effect.** The analysis reports the minimum detectable effect at 80%
  power, so the paper can state the smallest effect the design *could* have found
  rather than implying it found none.

---

## Part 3 — Code defects and their fixes

All new files are in `code/v2/`. v1 is left untouched for provenance.

### 3.1 The two arms ran through different framework paths — *critical*

`train_baseline` returns a **SeriesNetwork**; `prune_model` returns a **dlnetwork**.
The v1 timing loop branched on this:

```matlab
isDl = isa(net, 'dlnetwork');
...
if isDl
    Xsample = dlarray(single(Xsample), 'SSCB');   % pruned arm ONLY
end
predict(net, Xsample);
```

So the pruned arm paid a `dlarray` construction on **every single inference** that the
baseline never paid, and the two arms dispatched `predict` through entirely different
code. The measured difference confounds "effect of pruning" with "effect of changing
framework". This plausibly accounts for the whole MNIST +9.5% result — that run shows
the pruned arm 23.8% *slower*.

**Fix:** `to_dlnetwork.m` normalises every arm to `dlnetwork` before measurement.
`run_paired_measurement` hard-errors if any arm is not a `dlnetwork`.

### 3.2 Batch-size-1 inference measures framework overhead, not the network

At batch size 1 on a 20,490-parameter network, `predict()` is dominated by dispatch and
argument validation. Pruning cannot reduce dispatch. A batch-1 harness is structurally
incapable of resolving the effect under test — so part of the v1 null is an artefact of
the harness rather than a fact about pruning.

**Fix:** `measure_energy_v2.m` takes an explicit batch size (default 128).
Sweep it and report both regimes — "at batch 1 the effect is unmeasurable because
overhead dominates; at batch 256 it becomes resolvable" is a *better* paper than the
current one.

### 3.3 Allocation and RNG inside the timed loop

v1 charged `randi`, a strided copy out of the test array, and a type conversion to the
measured window.

**Fix:** `prepare_input_pool.m` builds all batches once, up front. The timed region
contains `predict` and nothing else. The same pool object is shared by every arm, so
cache behaviour is identical too.

### 3.4 No warm-up; output never consumed

First calls pay JIT and cold cache. And a discarded `predict` result invites elision or
lazy evaluation.

**Fix:** configurable warm-up (default 200 batches, discarded), and a checksum that
forces materialisation at identical cost in every arm.

### 3.5 The pruned arm received training the baseline never got — *critical*

`prune_model` runs a **full SGDM pass over the training set inside every pruning
iteration**. With 2 iterations that is 2 extra epochs the baseline never receives. So
v1's "pruned" is:

> baseline **+ 2 epochs extra training** + filter removal

The claim that the pruned model retains accuracy "within 1–2 percentage points"
therefore compares two things differing in two ways. The extra training plausibly
offsets some of the pruning damage.

**Fix:** `prune_model_v2.m` takes `pruneEnabled`. With `false` it runs the identical
loop — same seed, same shuffle, same gradient-step count, same `updateScore` calls —
but never calls `updatePrunables`. That yields a `baseline_ft` control arm, so:

- `acc(pruned) − acc(baseline_ft)` = effect of **pruning**
- `acc(baseline_ft) − acc(baseline)` = effect of **extra training**

`run_experiment_v2` measures all three arms.

### 3.6 Latent bug: stale SGDM momentum across pruning iterations

v1 carried `velocity` across iterations. `updatePrunables` changes learnable tensor
shapes, so the momentum accumulated against pre-prune shapes is stale on the next
iteration. **Fix:** velocity is reset whenever the structure changes.

### 3.7 Manual power transcription

**Fix:** `parse_hwinfo_log.m` extracts the window programmatically and returns the
full distribution — mean, SD, sample count, plus temperature and clock columns so
throttling can be *detected* rather than inferred after the fact.
`attach_power_to_table.m` fills every row and computes both total-draw and
idle-subtracted energy.

Also: HWiNFO was logging at ~2 s, giving ~15 samples per 30 s window. **Set it to
500 ms.**

### 3.8 FLOPs were never computed

The entire cited literature is FLOP-centric, and Section 2.4 pre-defends the omission
— which reads defensively. It is ten lines of arithmetic.

**Fix:** `count_flops.m` derives layer output sizes empirically from a forward pass
(robust to any architecture change) and reports MACs and FLOPs. Verified against the
manuscript: the script's parameter counts reproduce Table 4 exactly (20,490 → 15,810,
22.84%).

| Dataset | MACs baseline | MACs pruned | **MAC reduction** | Param reduction |
|---|---|---|---|---|
| MNIST / Fashion-MNIST | 1,031,744 | 779,590 | **24.44%** | 22.84% |
| ChestX-ray14 | 21,299,200 | 16,087,040 | **24.47%** | 22.16% |

---

## Part 4 — The mechanism you should name

Section 5 attributes all variance to "background system load or thermal state". That
reads as unexplained noise. There is a concrete, citable mechanism available, and the
data supports it.

Taylor pruning took conv2 from **32 → 25 filters**. 25 is not a multiple of 8. On AVX2
SIMD units, channel counts that do not align to the vector width break vectorisation:
the pruned layer can occupy the *same number of vector lanes* as the unpruned one while
doing less useful work — identical cost, less output. Combined with the framework
overhead in 3.1, this explains why the clean runs show the pruned model **slower**.

This converts the paper's argument from "we measured noise" to "we measured noise, and
here is the mechanism by which the signal was never there." Much stronger.

**Testable:** add an arm pruned to an 8-aligned count (conv2 32 → 24). If aligned
pruning yields a saving and unaligned does not, that is a genuine mechanistic finding
and arguably a second paper.

---

## Part 5 — Two paths forward

### Path A — Re-analysis only (days, no new measurement)

Salvages the paper with data on disk. Do all of this regardless.

1. Run `audit_v1_runs` — report the throughput finding as a primary result.
2. Re-frame the statistics as non-inferiority against 24.44% (Part 2).
3. Compute and report FLOPs.
4. Name the SIMD-alignment mechanism.
5. Fix the provenance gap: locate the 6 missing runs or reduce the reported n.
6. **Disclose the v1 harness defects as limitations.** The framework-path asymmetry
   (3.1) and the fine-tuning confound (3.5) must be stated. Reviewers who read the
   repository will find them.

**Revised claim:** *not* "pruning does not save energy" — the data cannot support
that. Instead: **"the measured saving falls decisively short of the theoretical
prediction (p = 1.4×10⁻⁶), and under unthrottled measurement the pruned network is
slower than the baseline despite 24.4% fewer MACs."** That is defensible, positive,
and more interesting.

### Path B — Full re-measurement (2–3 weeks)

Path A plus a clean dataset from the corrected harness.

```matlab
cd code/v2
T = run_experiment_v2('MNIST', XTrain, YTrain, XVal, YVal, XTest, YTest, ...
        numBlocks   = 6,   ...   % 6 interleaved A/B/C blocks, order randomised
        durationSec = 60,  ...   % 60 s per measurement, not 30
        settleSec   = 20,  ...   % symmetric settle before EVERY arm
        batchSize   = 128, ...
        hwinfoCsv   = 'C:\path\to\hwinfo.CSV', ...
        idleWatts   = 4.221);
results = analyze_energy_v2(T, theoreticalSaving = 24.44);
```

**Operator checklist:** HWiNFO at 500 ms with power, temperature and clock columns
enabled; mains power, battery not charging; power plan fixed and recorded; everything
else closed; five minutes idle before the first block; idle window recorded first.

**Also run the batch-size sweep** (1, 8, 32, 128, 512). Showing where the effect
becomes resolvable is the single most valuable addition available.

---

## Part 6 — Manuscript edits required

| Location | Change |
|---|---|
| Title | "Green Illusion" overclaims for one algorithm, one ratio, one CPU. Consider scoping it. |
| Abstract | Replace "close to zero / statistically indistinguishable" with the non-inferiority result. Absence of evidence framing must go. |
| §3.3 | Disclose the extra fine-tuning the pruned arm receives; add the `baseline_ft` control. |
| §3.4 | Disclose that v1 arms ran through different network classes and input paths. State batch size = 1. Report the 2 s logging interval as a defect, not a design choice. |
| §3.4 | Add MAC/FLOP counts (24.44%) alongside filter and parameter counts. |
| §4 | Add throughput as a primary result: pruned is slower in clean runs. Add the throttled/clean split. |
| §4 | Replace the t-test-against-zero with non-inferiority + Holm + MDE. |
| §5 | Add the SIMD-alignment mechanism. Soften "background load" from explanation to hypothesis. |
| §5 | Note that MNIST/Fashion-MNIST divergence is now partly attributable to the harness, not only to session conditions. |
| §6 | Promote interleaved paired measurement from implicit to a named recommendation. |
| Tables | `tables.md` is stale — it numbers 8 tables under a scheme that no longer matches the section files (now 1–5 in Methods, 6–7 in Results). Delete or regenerate. |
| References | `Dodge et al. (2022)` is listed but never cited in text. Verify `Argerich & Patiño-Martínez (2024)` page range "67890–67904" (looks like placeholder digits) and the `Peykani et al. (2026)` DOI year segment. |

---

## File index (`code/v2/`)

| File | Purpose |
|---|---|
| `to_dlnetwork.m` | Normalise every arm to one framework path (fixes 3.1) |
| `prepare_input_pool.m` | Pre-build batches; nothing allocated in the timed loop (3.3) |
| `measure_energy_v2.m` | Single-arm measurement: batched, warmed up, checksummed (3.2, 3.4) |
| `run_paired_measurement.m` | Interleaved, order-randomised, symmetric-settle protocol (1.1) |
| `prune_model_v2.m` | Pruning + compute-matched control arm; velocity reset (3.5, 3.6) |
| `parse_hwinfo_log.m` | Programmatic power extraction with temp/clock covariates (3.7) |
| `attach_power_to_table.m` | Fills power and energy for every row; total and net-of-idle |
| `count_flops.m` | MACs and FLOPs, empirically derived layer sizes (3.8) |
| `run_experiment_v2.m` | End-to-end orchestrator for one dataset |
| `analyze_energy_v2.m` | Paired non-inferiority / TOST / Holm / mixed model (Part 2) |
| `audit_v1_runs.m` | Forensic re-analysis of the v1 files (Part 1) |

The distribution helpers in `analyze_energy_v2` (`tcdf_local`, `tinv_local`,
`prctile_local`) are toolbox-free and were verified against SciPy: t-CDF agrees to
5×10⁻¹⁵, t-inverse to 4×10⁻⁸.
