# BVB Evaluation

The BVB protocol evaluates reconstructions on **two axes**.

| Axis | Measurement | Output summary |
| --- | --- | --- |
| **Dual VQA (DV)** | Retention of answers the judge gets right on the source video | `dual_vqa_summary.json` |
| **Latent Similarity (LS)** | Mean of layout and motion similarity computed from frozen V-JEPA 2.1 features | `vision_sim_summary.json` |

**Overall** is `((sqrt(DV) + sqrt(LS)) / 2) ** 2`, with both scores on a
0–100 scale. Both axes compare the rendered reconstruction with the source
video and run independently of the reconstruction agent.

Run all commands below from the **repository root**. The paper evaluates the
288 scenes and 5,130 questions in [`test.jsonl`](test.jsonl). Every
configuration uses the same reconstruction sandbox and a Stage-1 cost limit of
$3 per scene. Scores are never sent back to the reconstruction agent.

## 1. Render the animated scene camera

This step needs Blender and FFmpeg on the host and a Python 3.10+ environment.
The [Stage-1 environment](../sandbox/README.md) also works for this step.

```bash
python eval/batch_render_camera.py \
  --run sandbox/results/run_001 \
  --num-frames 64 \
  --resume
```

If Blender is not on your `PATH`, set `BLENDER_BIN` or pass
`--blender /path/to/blender`. The renderer samples frames across the full
camera timeline with EEVEE and caches the result as
`<run>/camera_renders/<scene>.mp4`. By default it deletes the intermediate PNG
frames after writing the MP4. Pass `--keep-pngs` to keep them for debugging.
This step does not call the judge API or the GPU encoder.

## 2. Score Dual VQA

Install the judge dependencies in an evaluation environment.

```bash
python -m pip install openai opencv-python-headless
export OPENAI_API_KEY="..."
```

The paper uses `gpt-5.4-mini` as the judge with 16 uniformly sampled frames per
video. The judge's answers on the source videos are computed once and shared
across runs in `sandbox/results/_dual_vqa_shared/original_answers.jsonl`. The
first command below builds this shared bank, and the second scores one run.

```bash
python eval/dual_vqa_metric.py \
  --ensure-originals-only \
  --model gpt-5.4-mini \
  --n-frames 16

python eval/dual_vqa_metric.py \
  --run sandbox/results/run_001 \
  --model gpt-5.4-mini \
  --n-frames 16
```

Both commands call the judge API only for answers that are not already cached,
and resume is on by default. Keep the judge model and frame sampling fixed when
you reuse a shared bank. Use a separate results directory if you change the
judging protocol.

Multiple-choice requests include the question and its separate `options` field,
preserving the option labels and order. This applies to source-video,
rendered-video, and text-only requests; questions without options are unchanged.
Answers cached by versions that omitted these options must be regenerated for
both source and rendered videos (and the text-only baseline, if used). Preserve
the old results and use a fresh results directory and run outputs for the corrected
protocol. `--force` only clears a run's Dual VQA outputs; it does not invalidate
the shared source-video or text-only answer banks.

Each run directory receives two outputs.

- `dual_vqa.jsonl` stores the answers, source-correct flags, and retention for
  each question.
- `dual_vqa_summary.json` stores retention, accuracy, per-task values, and
  coverage.

DV is `|C_source ∩ C_render| / |C_source|`, reported as a percentage, where
`C_source` and `C_render` are the sets of questions the judge answers correctly
on the source video and on the render. The denominator is the set of
**source-correct questions**, not all questions and not the number of scenes.
Per-scene DV is undefined for a scene with no source-correct questions, so such
a scene should not be counted as zero retention. The reported DV pools all
questions together and is not an unweighted mean of per-scene percentages.

The optional `--text-only` flag builds a chance-floor bank. The leaderboard
does not use it and reports retention directly. The old offline prediction
aggregator is deprecated and now exists only in Git history and local archives.
`import_dual_vqa_from_pilot.py` only migrates existing pilot logs and is not
needed for a fresh checkout.

## 3. Score Latent Similarity

Use a dedicated GPU environment that is separate from the Stage-1 agent
environment.

```bash
python -m pip install -r eval/requirements-vjepa.txt
python eval/vjepa_sim_metric.py \
  --run sandbox/results/run_001 \
  --vsi-bench VSI-Bench \
  --model apiantonio/vjepa2.1-vit-gigantic-384 \
  --num-frames 64 \
  --device cuda \
  --dtype bfloat16 \
  --resume
```

The frozen V-JEPA 2.1 ViT-G encoder compares the source and rendered clips in
two ways, and LS averages the two.

- **Layout** is the cosine similarity of temporally pooled spatial patch maps.
- **Motion** is the cosine similarity of global token-mean features.
- **LS** is the mean of Layout and Motion.

Each run directory receives two outputs.

- `vision_sim.jsonl` holds one score or failure record per selected scene.
- `vision_sim_summary.json` reports `vision_sim`, `layout_sim`, and
  `motion_sim` over all selected scenes, with failed rows counted as zero. Its
  `scored_only` field is a separate diagnostic over successful rows and is not
  the benchmark mean.

Raw summary scores are on a 0–1 scale, so multiply them by 100 for tables.
Render caching is on by default, so cached MP4s are reused instead of being
rendered again in Blender. `--no-keep-renders` turns off both cache reuse and
saving but leaves existing MP4s on disk. Newly rendered PNGs are temporary, and
features are never written to disk. Use `--limit`, `--sample`, or `--scene-ids`
only for subsets that are clearly labeled as subsets.

The [render and cluster guide](VJEPA_SIM_HANDOFF.md) explains how to move camera
renders, run independent GPU shards, and merge their results.

## 4. Check coverage and compute Overall

Use the same full test pool for every configuration. The paper assigns zero on
both axes to failed reconstructions, because dropping failures would change the
task. Record coverage alongside the scores, and fix any evaluation
infrastructure errors before reporting benchmark results.

The DV runner finds questions through the existing camera renders, and its raw
summary aggregates only the question records that were scored successfully. A
partial or failed run can therefore have a smaller denominator. **Do not treat
a partial-run summary as a complete 288-scene result.** Check it against
`test.jsonl` and the complete source-answer bank. Likewise, `num_scenes` in the
LS summary must cover the intended pool, including failure records.

Once both summaries cover the intended evaluation pool, compute Overall without
any further model calls.

```python
import json
import math
from pathlib import Path

run = Path("sandbox/results/run_001")
dv = 100 * json.loads((run / "dual_vqa_summary.json").read_text())["retention_rate"]
ls = 100 * json.loads((run / "vision_sim_summary.json").read_text())["vision_sim"]
overall = ((math.sqrt(dv) + math.sqrt(max(0.0, ls))) / 2) ** 2
print({"Dual VQA": dv, "Latent Similarity": ls, "Overall": overall})
```

The square-root mean is applied to the two configuration-level scores. It is
not an average of per-scene Overall scores, and it is neither the geometric
mean nor the arithmetic mean. As defined in the paper, a negative LS is clipped
to zero before taking its square root.

The [results CSV](../assets/bvb-results.csv) contains full-precision values for
all 51 configurations in the paper. The
[project page](https://yoloytang.me/BVB/#leaderboard) adds interactive rankings
and cost comparisons.

## Human blind ranking

In the paper's human study, each of 15 raters ranks five anonymized
reconstructions on nine scenes. LS correlates with human preference at the
scene-model level, with a Spearman ρ of 0.83. Overall matches the human
ordering of the five tested configurations, with ρ = 1.00. See the
[human-study guide](human/README.md) and the [protocol](human/PROTOCOL.md) for
details.
