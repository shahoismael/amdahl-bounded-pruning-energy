# Amdahl-bounded pruning energy

Code and measurement records for a hardware-measured audit of structured pruning's
inference energy saving on CPU.

A fixed Taylor channel-pruning configuration was applied to convolutional classifiers
on four datasets. Energy was measured from processor package power sensors rather than
estimated in software, using an interleaved, order-randomised protocol with a
compute-matched control arm.

**Headline:** a 22 to 26 percent reduction in multiply-accumulate operations produced a
measured energy saving of 9.4 percent, roughly 40 percent of the prediction. Processor
power did not change; the saving is entirely reduced execution time. Below batch size 32
the same compression delivered no measurable saving at all.

## Layout

```
code/v1/          Original harness. Superseded — retained for provenance.
code/v2/          Corrected harness. This is what produced every reported result.
figures/          Figure generation, reads the CSVs directly.
data/measurements/    Per-block records, 4 datasets x 6 blocks x 3 arms.
data/power_logs/      Raw HWiNFO sensor logs.
data/v1_recovery/     Forensic re-analysis of the superseded protocol.
```

## Reproducing the results

Requires MATLAB R2025b with the Deep Learning Toolbox and the Model Compression Library.
No Statistics Toolbox is needed; the distribution functions in `analyze_energy_v2.m` are
implemented from `betainc` and verified against SciPy.

```matlab
cd code/v2
addpath('../v1');

[XTrain, YTrain] = load_idx_dataset('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
[XTest,  YTest ] = load_idx_dataset('t10k-images.idx3-ubyte',  't10k-labels.idx1-ubyte');

T = run_experiment_v2('MNIST', XTrain, YTrain, XVal, YVal, XTest, YTest, ...
        numBlocks = 6, durationSec = 60, settleSec = 20, batchSize = 128, ...
        hwinfoCsv = 'path/to/hwinfo.CSV', idleWatts = 4.221);

results = analyze_energy_v2(T);
```

Figures regenerate from the CSVs with no MATLAB dependency:

```bash
pip install matplotlib numpy
python figures/make_figs_v3.py . output_dir
```

Every plotted value is computed at run time from `data/`. Nothing is transcribed.

## Operator checklist before measuring

Power measurement is fragile in ways that do not announce themselves. The v1 results in
`data/v1_recovery/` are what happens when these are skipped.

- HWiNFO in sensors-only mode, CSV logging on, interval 500 ms.
- Mains power. Battery not charging.
- Power plan fixed and recorded.
- Everything else closed. No browser, no antivirus scan, no sync client.
- Five minutes idle before the first block, so the machine starts from a steady
  thermal state.
- Record an idle window with `measure_idle_power` before you begin.

## What changed between v1 and v2

`docs_METHODS_REVISION_v2.md` documents seven defects found in the original harness and
analysis. The load-bearing ones:

- **Ordering artefact.** v1 always measured baseline first and pruned second, so machine
  drift was perfectly confounded with condition. Every apparent saving in the v1 data came
  from a run where the processor was throttled during the baseline window
  (*r* = −0.883 against run throughput). v2 interleaves and randomises order within blocks.
- **Framework-path asymmetry.** The two arms were different network object types and
  dispatched inference through different code. `to_dlnetwork.m` normalises every arm.
- **Fine-tuning confound.** Taylor pruning runs gradient updates as part of its scoring
  loop, so the pruned arm received training the baseline never got. `prune_model_v2.m`
  adds a compute-matched control that runs the identical loop without removing filters.
- **Batch size 1.** At that batch size framework dispatch dominates and pruning cannot
  reduce dispatch. The v1 null was partly an artefact of the harness.
- **Data leakage.** MIT-BIH split at beat level, putting the same patient in train and
  test. ChestX-ray14 split at image level with multiple images per patient. Both fixed to
  subject-level splits.

## Known limitations

Package power only; memory, storage and platform draw are not measured. Energy is
reported from total draw including a 4.221 W idle floor that pruning cannot reduce, which
makes the reported fractions conservative. Power logging ran at 2000 ms rather than the
intended 500 ms, so within-window variance is characterised coarsely. One pruning
algorithm, one compression ratio, one CPU.

## Data availability

The four datasets are public: MNIST and Fashion-MNIST via Zalando Research, MIT-BIH via
PhysioNet, NIH ChestX-ray14 via the NIH Clinical Center. They are not redistributed here.

## Licence

MIT. See `LICENSE`.
