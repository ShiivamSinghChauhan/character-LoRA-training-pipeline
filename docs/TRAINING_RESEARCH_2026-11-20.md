# wgveh character LoRA: training setup research (20 Nov 2026)

Goal (user): one character learned very deeply. The face must be unique and stable across emotions, expressions and angles. A small difference from the training photos is acceptable; identity consistency matters most.

## 1. Model choice

| Model | Licence (paid client job) | Evidence for identity | Kaggle T4 fit | Decision |
|---|---|---|---|---|
| FLUX.2 Klein **Base 4B** | Apache 2.0 (clean) | BFL: character LoRA 1500-3000 steps, LR 8e-5..1e-4. LoRA trained on base also runs on distilled 4-step ("usually even better", BFL blog). No public face benchmark for 4B. | Yes. Unsloth measured 2.0-2.8 s/step at 512 px on T4 (fp16 + loss scaling, nf4 base, r16, 0 NaN). | **Primary** |
| Z-Image **Base** (Tongyi-MAI/Z-Image, 6B) | Apache 2.0 | malcolmrey (thousands of person LoRAs): "ZImage seems a bit better in likeness" than Klein 9B; others disagree. | Yes, Unsloth fp16 path covers Z-Image (T4, 7.8 s/step measured on Turbo). | **Parallel run #2** |
| Krea 2 **Raw** (12.9B) | Krea 2 Community Licence: commercial use of model, derivatives and outputs only if revenue < US$1M; read section 3.2 (derivative distribution) before delivery | Strongest number found: Maystro 81-frame test, Raw-trained LoRA 0.726 face-sim vs real photos (Klein Base 9B: 0.67 at weight 1.6). | Experimental: community T4 notebook (krea2-t4-lora-studio, ~8.9 s/step); Unsloth refuses Krea 2 on T4. | Optional run #3, only after #1 works |
| FLUX.2 Klein Base 9B | FLUX Non-Commercial v2.1: not allowed for paid work | Maystro: 0.67 at weight 1.6 | Heavy | Skip |
| Qwen-Image-2.1 (7B) | Qwen Research Licence: research/evaluation only | Top Arena portraits | n/a | Skip |
| FLUX 3 Image Klein Base (5.6B) | Kommunity Licence: commercial *outputs* for < $5M revenue, "not blanket permission for commercial fine-tuning or distribution" | none for faces | Unsloth/diffusers require bf16 (refused on T4) | Skip for now |
| MiniMax H3-Image (33B) | n/a | general leaderboards only | 80 GB preset | Impossible on T4 |

## 2. Trainer on Kaggle T4 (no bf16)

- T4 has no native bf16. bf16 emulation in ai-toolkit gave ~20 s/it on Z-Image earlier. Plain fp16 gave NaN on Z-Image.
- Unsloth PR #18355 (merged 19 Nov 2026): DiT LoRA training on T4 in fp16 with GradScaler. Modulation, time-embedding and output layers stay fp32, and the LoRA weights stay fp32. Klein-base-4B: 0 NaN, 2.0 s/step at 512 px, 10.7 GB peak. It is driven from the Studio UI; the code is in `studio/backend/core/training/diffusion_dit_trainer.py`.
- ai-toolkit supports FLUX.2-klein-base-4B, Z-Image base and Krea 2 Raw, plus masks (`mask_path`, `mask_min_value`), LoKr, DOP and multi-resolution. T4 fp16 is untested.
- musubi-tuner supports `--model_version klein-base-4b`, `--fp8_base --fp8_scaled` and `--blocks_to_swap` (max 13 for 4B). Its docs recommend bf16; fp16 is untested.
- The ChatGPT kit (diffusers `train_dreambooth_lora_flux2_klein.py`, `render.py`) hard-requires bf16 and will not run on Kaggle T4 as-is.

**Step 0 (about 30 min GPU):** smoke test on one Kaggle session. Run 30 steps of ai-toolkit Klein-4B with `dtype: fp16` (plus fp32 islands if needed) against musubi `--mixed_precision fp16`, and fall back to the Unsloth trainer. Record s/step at 512 and 1024, peak VRAM and NaN count. Keep the fastest stable one. Masked loss is a must-have, which favours ai-toolkit.

## 3. Dataset: the biggest lever

The current frames come from 3 short videos and are mostly neutral or soft-smile, so the expression range is small and neighbouring frames are near-duplicates. Target (revised 21 Nov): **40-70 distinct images** after near-duplicate removal; go higher only if each extra image adds a new angle, expression or setting (practitioners: 15-40 typical, repetition causes overfit, no controlled study exists). See section 9 for the full dataset Q&A.

- **Framing:** about 40% face/head-and-shoulders close-ups at 1024, about 35% waist-up, about 25% full body. For close-ups, crop the head region from the RAW frame and re-enhance it at about 1 MP; do not upscale the tiny face from the full-frame output.
- **Expressions:** neutral, closed smile, teeth smile, laugh (eyes narrowed), surprise, sad, annoyed, pout, eyes closed, mid-speech. Angles: front, 3/4 left and right, profile, looking up and down.
  - Missing expressions must be synthesised with an edit model, gated by ArcFace (keep at or above about 0.6 against the identity centroid) plus a manual check. Keep synthetic images at 30% or less.
  - Use an Apache-2.0 editor for this: Klein 4B edit or Qwen-Image-Edit-2511.
- **Identity tightening:** compute the ArcFace centroid of all candidates and drop outliers. This gives one consistent face for the LoRA to lock onto, which matches the "unique, deep" goal.
- **Hygiene:** no horizontal flips (they destroy asymmetry). Keep 8-10 RAW frames out of training as the test gallery.
- **Masks for face-weighted loss:** head/face 1.0, hair and body about 0.5, background about 0.2. Maystro's identity mode weights the head region 3x and samples face-forming noise levels more often; they report the face learned about 4x faster and a better match to the photos.

## 4. Captions and trigger

- Every caption starts with `wgveh woman, ...`. Klein, Z-Image and Krea 2 use Qwen3 text encoders, which are causal: each token only sees earlier tokens. A trigger at the start therefore conditions every later token, while a trigger at the end conditions nothing after it. This is my reasoning from the architecture, not a measured result.
- Caption what varies: expression (mouth and eyes), head angle, gaze, framing, clothing, background, lighting.
- Do NOT caption permanent face traits, so they bind to the trigger. Leave hair uncaptioned unless hairstyle control is wanted.
- Caption dropout 0.05.
- Known issue (Maystro): the trigger sometimes prints as text on signs and shirts. Test prompts with and without the trigger.

## 5. Run 1 config (ai-toolkit style, Klein Base 4B)

```yaml
trigger_word: "wgveh woman"
network: {type: lora, linear: 32, linear_alpha: 32}        # GPU1 twin: {type: lokr, lokr_full_rank: true, lokr_factor: 4}
datasets:
  - folder_path: .../images      # .txt captions, same stem
    mask_path: .../masks         # grey-level weight maps
    mask_min_value: 0.1
    caption_dropout_rate: 0.05
    resolution: [512, 768, 1024]
    flip_x: false
    cache_latents_to_disk: true
train:
  steps: 2500                    # save_every 250 -> pick by eval, not loss
  batch_size: 1
  lr: 1.0e-4                     # constant, adamw8bit, weight_decay 1e-4
  optimizer: adamw8bit
  noise_scheduler: flowmatch
  timestep_type: weighted
  content_or_style: balanced
  gradient_checkpointing: true
  cache_text_embeddings: true
  dtype: fp16                    # T4; decided by smoke test
model:
  name_or_path: black-forest-labs/FLUX.2-klein-base-4B
  quantize: false                # 4B fp16 ~8 GB fits a 15 GB T4
```

- Estimated time: 512 px about 2-3 s/step (measured), 1024 px about 9-10 s/step (estimated from L4 numbers). Mixed resolutions put 2500 steps at about 3-4 h on one T4.
- Parallel layout: each Kaggle session has 2 T4s, so run 2 experiments per session (CUDA_VISIBLE_DEVICES=0 and 1):
  - Session 1 (acct C): GPU0 LoRA r32 + mask, GPU1 LoKr f4 + mask.
  - Session 2 (acct A, after its weekly reset): GPU0 Z-Image Base LoRA r32 + mask, GPU1 Klein LoRA r32 without mask (ablation).

## 6. Evaluation (choose a checkpoint by identity, not loss)

- **Suite:** 12 prompts x 4 fixed seeds. The prompts cover neutral, smile, teeth smile, laugh, surprise, sad, annoyed, pout, eyes closed, talking, profile, and a full body at the beach. Add 4 leakage prompts that say "a woman" without the trigger.
- **Checkpoints:** 500 / 1000 / 1500 / 2000 / 2500.
- **Inference settings:** Klein 4B distilled at 4 steps, CFG 1, LoRA weight 1.0 / 1.3 / 1.6. Also base at 30-50 steps, CFG 4.
- **Metrics (ArcFace):**
  - (a) self-consistency: mean pairwise similarity between generated faces across expressions and seeds. This is the main "unique deep face" score.
  - (b) similarity to the held-out RAW gallery.
  - (c) leakage: similarity of no-trigger outputs to the character (lower is better).
  - Plus a manual pass/fail per image for expression correctness and realism.

## 7. ChatGPT kit (`character-lora-kit.zip`): what we keep and what we drop

- **Keep:** fixed prompts and seeds, held-out split by source clip, checkpoints every 250, rank 16-32 at LR 1e-4, no flips, trigger + class noun, caption visible variables, per-expression pass counts, the reverse-reference test.
- **Drop:**
  - Klein 9B as the first model (non-commercial licence).
  - The bf16-only scripts (they won't run on T4).
  - The "shoot 40 new photos" plan (the client won't).
  - 1000 steps as final (BFL says 1500-3000 for characters).
  - Qwen-Image-2.1 and H3 (licence and hardware).

## 8. Licence caveat (important for a paid job)

The FLUX Non-Commercial Licence v2.1 (Klein 9B, used for the enhancement stage) says:

- A Non-Commercial Purpose applies only "so far as you do not receive any direct or indirect payment arising from the use of the FLUX Model".
- Use "to train, fine tune, or distill other models for commercial use ... is not a Non-Commercial Purpose".

So the Klein-9B-enhanced dataset is outside that licence for this paid job. Options:

1. Get a commercial licence from BFL. No public self-serve price was found.
2. Redo the enhancement and edits with an Apache-2.0 model (Klein 4B, Qwen-Image-Edit-2511). There is no 4B version of the dx8152 consistency LoRA.
3. Accept the risk (user's decision).

## 9. Dataset Q&A (21 Nov 2026)

Evidence level: community guides and anecdotes only. No controlled study was found for any of these points, so labels below say which items are sourced and which are reasoning.

- **Mixed aspect ratios:** fine. ai-toolkit buckets by ratio. With batch 1, bucketing does not affect batching.
  - Rules: no stretching, no padding or letterbox borders, no images upscaled from small sources.
  - Close-ups at 1:1 or 3:4. Full body at 2:3 or 9:16.
  - BFL: 1024 px or more.
- **Filters (x-ray, inverted colours):** avoid. "x-ray" is not a photo style the base model knows well, so a caption is unlikely to separate it cleanly. One or two images out of 50+ add very little identity (reasoning).
  - Safer rescue for a noisy but unique frame: convert to **black-and-white** and caption "black and white photograph". This removes chroma noise, and the anime community tags monochrome images successfully (sourced).
  - Or re-enhance the frame. Drop near-duplicates instead of rescuing them.
- **Size and split (rule of thumb, mine):** 40-70 distinct images.
  - Framing: about 45-50% face or head-and-shoulders, about 30% waist-up, about 15-20% full body.
  - Angles: front about 35%, 3/4 (both sides) about 35%, profile (both sides) about 10%, looking up/down about 10-15%.
  - Expressions: each target expression needs at least 3-5 images. The LoRA only learns the expressions present in the data, and many LoRAs cannot even tell a closed-mouth smile from an open-mouth smile (civarchive 4563).
- **Captions:**
  - Caption what varies: clothing, background, lighting, expression, pose, angle, framing, filters.
  - Do not caption fixed identity traits (face shape, eyes, nose, skin tone, moles); these bind to the trigger.
  - Hair: caption it only if you want hair to be changeable.
  - Captioning does not erase a bias. An outfit or background shown in most images can still bleed, so the data must vary too.
  - Use the same words for the same thing every time. Accessories that appear in every image must be captioned or they bind.
- **Trigger:** first, in every caption ("wgveh woman, ..."). The BFL/HF examples put it first, and the encoder is causal (my reasoning). Consistency matters more than position.

## Sources
- BFL Klein training docs: https://docs.bfl.ai/flux_2/flux2_klein_training
- BFL/HF blog, Klein LoRA <60 min: https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora
- Maystro character tests (Klein 9B, Krea 2): https://maystro.ai/docs/training-studio/train-image-lora/
- Unsloth T4 fp16 PR: https://github.com/unslothai/unsloth/pull/18355 ; speed PR: https://github.com/unslothai/unsloth/pull/18301
- musubi-tuner FLUX.2 doc: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/flux_2.md
- ai-toolkit: https://github.com/ostris/ai-toolkit ; malcolmrey templates: https://huggingface.co/datasets/malcolmrey/various/tree/main/training-scripts/aitoolkit
- malcolmrey likeness discussion: https://huggingface.co/malcolmrey/klein9/discussions/2
- Klein base 4B card: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B ; 9B licence: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/blob/main/LICENSE.md
- Z-Image base card: https://huggingface.co/Tongyi-MAI/Z-Image
- Krea 2 licence: https://krea.ai/krea-2-licensing ; T4 notebook thread: https://discuss.huggingface.co/t/krea-2-lora-training-on-a-colab-t4-looking-for-clean-run-testers/180786
- FLUX 3 Image Klein base card: https://huggingface.co/black-forest-labs/flux-3-image-klein-base
- Qwen-Image-2.1 licence coverage: https://efsora.com/insights/qwen-image-2-1-alibaba-gave-back-the-weights-and-kept-the-business-rights
- Qwen-Image-Edit-2511 (Apache 2.0): https://huggingface.co/Qwen/Qwen-Image-Edit-2511
