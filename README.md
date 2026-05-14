# Sparse Primitive Flow Composition

This repository is an early research prototype for training-free, inference-time steering of rectified-flow text-to-image models such as Stable Diffusion 3 Medium.

The current main method is **Sparse Primitive Flow Composition for Rectified Text-to-Image Generation**.

## Method

Older AIM-Flow versions exposed three failure modes:

- Standalone primitive prompts caused semantic drift.
- Anchor-residual methods over-weighted the base prompt and collapsed toward the anchor.
- Hand-written early/middle/late schedules made behavior fragile.

The simplified method keeps the full target prompt as the main generation path. Given:

```text
S = {S0, S1, ..., Sk, P}
```

where `S0` is a coarse source prompt, `S1...Sk` are primitive-focused complete prompts, and `P` is the full target prompt.

At ordinary timesteps:

```text
v_P = f_theta(x_t, t, P)
x_next = Step(x_t, v_P)
```

At selected aggregation timesteps:

```text
v_i = f_theta(x_t, t, S_i)
v_P = f_theta(x_t, t, P)
v_VFA = sum_i alpha_i v_i
```

`alpha_i` comes from base condition weights, consensus agreement between flows, and target consistency agreement with `v_P`.

LTP uses the full target prompt as the reference:

```text
x_ref = Step(x_t, v_P)
x_cand = Step(x_t, v_VFA)
x_next = x_ref + Clip(x_cand - x_ref)
```

This is not an anchor/source residual method. The source prompt is just one condition in VFA. The method uses no VQA, no CLIP reward, no external image judge, no training, no fine-tuning, and no image feedback during sampling.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/aim-flow.git
cd aim-flow
pip install -e ".[dev]"
```

Stable Diffusion 3 Medium is gated. Accept the model license on Hugging Face and set `HF_TOKEN`.

## Kaggle

The Kaggle config defaults to 512x512, 24 steps, `float16`, CPU offload, VAE slicing, sequential condition forwards, and `text_encoder_3=None` for memory.

Kaggle P100 sessions may need a Pascal-compatible PyTorch wheel:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir --force-reinstall torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then restart the runtime before loading SD3.

## Commands

Final-only aggregation:

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key german_shepherd \
  --output-dir outputs/german_shepherd_final_only \
  --modes base source_only primitive_flow_final_only \
  --final-only
```

Sparse aggregation:

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key german_shepherd \
  --output-dir outputs/german_shepherd_sparse \
  --modes base source_only primitive_flow_sparse \
  --aggregation-steps 12 16 20 23
```

Dense aggregation:

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key german_shepherd \
  --output-dir outputs/german_shepherd_dense \
  --modes base source_only primitive_flow_sparse \
  --aggregate-every-n-steps 1
```

Useful memory reductions:

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key german_shepherd \
  --output-dir outputs/german_shepherd_small \
  --modes base source_only primitive_flow_sparse \
  --num-inference-steps 16 \
  --height 384 --width 384 \
  --ltp-mode velocity \
  --max-primitives 2
```

## Outputs

PrimitiveFlow comparisons create:

- `base_full_target.png`
- `source_only.png`
- `primitive_flow_final_only.png`
- `primitive_flow_sparse.png`
- `primitive_flow_dense.png`, when requested
- `comparison_grid.png`
- matching `metadata_*.json` files

Metadata records aggregation steps, target/source/primitive prompts, VFA weights, consensus gates, target consistency gates, LTP diagnostics, and any latent-to-velocity LTP fallback.

## Tests

Tests are CPU-only and do not load SD3:

```bash
pytest -q
```

## Limitations

- Custom SD3 denoising depends on Diffusers API compatibility.
- Sparse flow composition is slower than base generation.
- Final-only aggregation may be too late for structural changes.
- Evaluation is still qualitative until benchmark code is added.
- Latent LTP falls back to velocity LTP when scheduler stepping is not safely repeatable.

## License

Repository code is MIT licensed. Stable Diffusion 3 Medium is not included and is governed by Stability AI's model license.
