---
pretty_name: FundusTalk v1 — Diabetic Retinopathy Chat SFT (11k)
license: cc-by-nc-4.0
language:
- id
- en
task_categories:
- text-generation
task_ids:
- dialogue-modeling
annotations_creators:
- machine-generated
language_creators:
- machine-generated
source_datasets:
- original
size_categories:
- 10K<n<100K
tags:
- medical
- ophthalmology
- diabetic-retinopathy
- synthetic
- conversational
- multi-turn
- sft
- distillation
- indonesian
- code-switching
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
- config_name: raw
  data_files:
  - split: train
    path: raw/shard-*.jsonl
- config_name: scenarios
  data_files:
  - split: train
    path: scenarios.jsonl
---

# FundusTalk v1 — Diabetic Retinopathy Chat SFT (11k)

Synthetic multi-turn consultations for fine-tuning [`microsoft/MediPhi-Instruct`](https://huggingface.co/microsoft/MediPhi-Instruct)
as **FundusAI**, the in-app assistant that explains diabetic-retinopathy screening results in the
[Fundusnap](https://fundusnap.faizath.com) app.

Distilled from [`microsoft/phi-4`](https://huggingface.co/microsoft/phi-4) via OpenRouter.

> ⚠️ **Synthetic data. Not a medical device.**
> No real patient data, images, or records were used — every prediction record is procedurally
> generated. Models trained on this dataset carry no regulatory clearance and must never be the sole
> basis for diagnosis, referral, or treatment.

## At a glance

| | |
|:--|:--|
| Conversations | **10,849** |
| Assistant turns | 39,288 (mean 77.2 words) |
| Turns per conversation | 3.62 mean |
| Languages | Indonesian, English, Indonesian–English code-switch |
| Teacher | `microsoft/phi-4` |
| Intended student | `microsoft/MediPhi-Instruct` |
| Keep rate after filtering | 90.4% (10,849 of 12,000) |
| Numeric grounding | 97.2% |

## Splits

| Split | Conversations | File | Size |
|:--|--:|:--|--:|
| `train` | 10,201 | `data/train.jsonl` | 49 MB |
| `validation` | 324 | `data/validation.jsonl` | 1.6 MB |
| `test` | 324 | `data/test.jsonl` | 1.6 MB |

Splits are disjoint by conversation id and stratified to preserve the category, language, grade, and
record-profile distributions described below.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("fundusnap/fundusnap-fundustalk-v1-chatsft-11k")
print(ds["train"][0]["messages"])
```

Ready to drop into TRL's `SFTTrainer` or Axolotl without reshaping — each record is a complete
`messages` array ending on an assistant turn.

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model="microsoft/MediPhi-Instruct",
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
)
```

### Additional configs

| Config | Records | What it is |
|:--|--:|:--|
| `default` | 10,849 | The filtered, split dataset. Use this to train. |
| `raw` | 12,000 | Unfiltered teacher output, 25 shards, exactly as generated. |
| `scenarios` | 12,000 | The seeded generation plan — one prediction record and persona brief per conversation, before any text was generated. |

```python
raw = load_dataset("fundusnap/fundusnap-fundustalk-v1-chatsft-11k", "raw")
scenarios = load_dataset("fundusnap/fundusnap-fundustalk-v1-chatsft-11k", "scenarios")
```

`raw` is published so the 1,151 rejected conversations can be inspected rather than taken on trust —
the keep rate only means something if you can see what was dropped. `scenarios` lets you regenerate
the corpus against a different teacher while holding the input distribution fixed.

## Format

Each line is `{"id", "messages", "meta"}`. The `messages` array reproduces the **exact** envelope the
production Fundusnap API sends to the chat model:

1. `system` — the Fundusnap persona prompt, verbatim
2. `system` — Azure Custom Vision severity probabilities as compact JSON, plus the production
   explanatory sentence
3. `system` — YOLO11m `detectionArtifacts` as compact JSON (`class_name` + `confidence`, box
   coordinates stripped, exactly as the API serialises them), plus its explanatory sentence
4. alternating `user` / `assistant` turns

JSON inside the system messages is emitted with `separators=(',', ':')` to match JavaScript
`JSON.stringify` byte-for-byte. **Train on this envelope and the model sees at training time exactly
what it will see at inference time.**

### `meta` fields

| Field | Values |
|:--|:--|
| `category` | `result_explanation`, `detector_literacy`, `safety_refusal`, `general_knowledge`, `adversarial_oos` |
| `language` | `id`, `en`, `id_codeswitch` |
| `true_grade` | `0`–`5` (No DR → Advanced PDR) |
| `top_tag` | Highest-probability Azure severity tag |
| `profile` | `normal`, `landmarks_only`, `poor_quality`, `empty_detections`, `low_confidence`, `disagreement` |
| `persona` | One of 12 patient / caregiver / clinician personas |
| `lesion_count`, `n_turns` | Integers |

## Composition

### By conversation category

| Category | Count | Share | What it teaches |
|:--|--:|--:|:--|
| `result_explanation` | 4,574 | 42.2% | Ordinary consultation about a specific scan result |
| `safety_refusal` | 1,793 | 16.5% | User requests a diagnosis, medication change, or permission to skip care |
| `detector_literacy` | 1,769 | 16.3% | User has misread the technical output; assistant corrects it |
| `general_knowledge` | 1,570 | 14.5% | Broader DR questions asked with a result in context |
| `adversarial_oos` | 1,143 | 10.5% | Prompt injection, persona-drop attempts, out-of-scope conditions |

### By language

| Language | Count | Share |
|:--|--:|--:|
| `id` | 5,118 | 47.2% |
| `en` | 3,177 | 29.3% |
| `id_codeswitch` | 2,554 | 23.5% |

`id_codeswitch` keeps clinical terms in English inside Indonesian prose, which is how Indonesian
patients and health workers actually discuss these results.

### By true severity grade

| Grade | Stage | Count | Share |
|:--|:--|--:|--:|
| 0 | No DR | 2,178 | 20.1% |
| 1 | Mild/Early NPDR | 2,218 | 20.4% |
| 2 | Moderate NPDR | 2,285 | 21.1% |
| 3 | Severe NPDR | 1,680 | 15.5% |
| 4 | PDR | 1,516 | 14.0% |
| 5 | Advanced PDR | 972 | 9.0% |

### By record profile

| Profile | Count | Share |
|:--|--:|--:|
| `normal` | 6,466 | 59.6% |
| `landmarks_only` | 1,217 | 11.2% |
| `poor_quality` | 965 | 8.9% |
| `empty_detections` | 796 | 7.3% |
| `low_confidence` | 742 | 6.8% |
| `disagreement` | 663 | 6.1% |

Profiles are deliberate edge cases, not noise:

- `landmarks_only` — detector found only `Disc`/`Fovea`; teaches that anatomy is not pathology
- `empty_detections` — nothing detected; teaches "not detected" ≠ "not there"
- `poor_quality` — artefact-heavy capture; assistant should suggest retaking the photo
- `disagreement` — classifier grade and detector findings conflict; assistant must say so honestly
- `low_confidence` — every lesion barely over threshold; assistant must convey uncertainty

### Personas

Twelve, roughly balanced (806–1,019 conversations each): `anxious_patient`, `pregnant_patient`,
`skeptical_patient`, `rural_patient_limited_access`, `long_term_diabetic`, `family_caregiver`,
`community_health_worker`, `low_literacy_patient`, `newly_diagnosed_patient`, `curious_student`,
`primary_care_nurse`, `general_practitioner`.

## Filtering

12,000 conversations were generated; 10,849 survived validation (**90.4%** keep rate).

| Rejection reason | Count |
|:--|--:|
| `duplicate_opening` | 683 |
| `no_clinician_referral` | 314 |
| `assistant_too_short` | 85 |
| `language_mismatch_id` | 43 |
| `diagnostic_language` | 22 |
| `meta_leak` | 2 |
| `truncated_final_turn` | 2 |

Filters are high-precision by design — it is better to keep a mediocre conversation than to silently
delete an entire behaviour class. `no_clinician_referral` and `diagnostic_language` are the
safety-critical ones: they drop any conversation where the assistant failed to route the user to a
clinician, or spoke as though it were making a diagnosis.

Full counts in `rejects.json`.

## Measured quality

| Metric | Value | What it means |
|:--|--:|:--|
| Numeric grounding | **97.2%** | Figures the assistant quotes that trace back to the record (19,709 of 20,287) |
| Opening diversity | **100.0%** | Distinct first user turns — no two conversations open the same way |
| Safety refusal rate | **88.8%** | `safety_refusal` conversations containing an explicit refusal (1,593 of 1,793) |
| Non-lesion misframing | **87 convs** | Assistant framed `Disc`/`Fovea`/`Artefact` as damage without correcting it |
| Undetected-class mentions | 1,065 negated / 429 unnegated | Naming an absent class is usually *correct* ("no microaneurysms were found, and the detector misses about half of them"). Only unnegated mentions are candidate hallucinations. |

Full audit in `audit.json`; per-split distributions in `stats.json`.

## Known limitations

Stated plainly, because a synthetic medical dataset that claims to be clean is not trustworthy:

- **Teacher-inherited error.** Every turn comes from `phi-4`. Its factual mistakes about diabetic
  retinopathy are reproduced here and no clinician reviewed the corpus.
- **The 2.8% ungrounded figures.** 391 conversations quote at least one number that does not trace
  back to the prediction record. Some are benign (population statistics); some are fabricated.
- **The 11.2% non-refusals.** 200 `safety_refusal` conversations do not contain an explicit refusal.
  They were kept because dropping them would have skewed the category, but they weaken the signal.
- **87 non-lesion misframings** survived filtering — conversations where normal anatomy or a quality
  flag was discussed as if it were disease.
- **Simulated, not observed.** Prediction records are procedurally generated from a severity/lesion
  co-occurrence model, not sampled from production traffic. Real-world record distributions differ.
- **Indonesian-centric.** Locales, health-system references, and code-switching patterns are
  Indonesian. The English subset is not locale-neutral.

## Domain constraints encoded in the data

Taken from the Fundusnap model cards and enforced in the teacher prompt:

1. Two independent systems produce the report — an Azure classifier for severity, a YOLO11m detector
   for findings. They can disagree, and neither validates the other.
2. `Disc` and `Fovea` are **normal anatomy**; `Artefact` is an **image-quality flag**. None of the
   three is a lesion, and none is evidence of disease.
3. The **number of detections is not a severity score**.
4. Confidence is **not calibrated** and is not a probability of having the disease.
5. An empty findings list means "nothing was detected" — the detector's recall is ≈0.53, and it is
   weakest on the smallest, earliest lesions.
6. The assistant explains; it never diagnoses, and always routes to a clinician.

## Intended use

**In scope.** Fine-tuning a small assistant to explain DR screening output to patients and
health workers in Indonesian and English; research on grounding, refusal behaviour, and
code-switching in medical dialogue; benchmarking safety behaviour on the `test` split.

**Out of scope.** Any clinical decision-making. Training a model to grade severity or detect lesions
— this dataset contains no images. Deployment without a clinician in the loop. Any commercial use
(see licensing).

## Provenance and licensing

Fully reproducible: the scenario plan (`scenarios.jsonl`) is seeded and deterministic; only the
teacher's sampling varies between runs. Generation and filtering logs are in `logs/`.

Licensed **CC BY-NC 4.0**. The severity tag names come from the Fundusnap Azure Custom Vision
project; the twelve detection class names come from
[`fundusnap/fundusnap-v1-lesiondet-yolo11m-20m`](https://huggingface.co/fundusnap/fundusnap-v1-lesiondet-yolo11m-20m),
whose weights are CC BY-NC 4.0 and additionally inherit Ultralytics' AGPL-3.0 terms. Treat anything
derived from this dataset as **non-commercial** unless you have cleared those terms independently.

`microsoft/phi-4` output is subject to the MIT-licensed model's terms; no restriction on synthetic
data reuse is imposed by that licence.

## Citation

```bibtex
@misc{fundustalk2026,
  title  = {FundusTalk v1: Synthetic Multi-Turn Consultations for Diabetic Retinopathy Result Explanation},
  author = {Fundusnap},
  year   = {2026},
  url    = {https://huggingface.co/datasets/fundusnap/fundusnap-fundustalk-v1-chatsft-11k}
}
```
