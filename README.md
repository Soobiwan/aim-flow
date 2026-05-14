# Your Model Knows What You're Aiming For: AIM-Flow for Image Generation by Rectified Models

**AIM-Flow** (**Anchor-Informed Modular Flow Guidance**) is an early research prototype for training-free, inference-time steering of rectified-flow-style text-to-image models such as Stable Diffusion 3 Medium.

The current method is **AIM-Flow v2: Anchor-Preserved Primitive Steering**. It decomposes a dense prompt into a global anchor scene and semantic primitives, then steers the model using only its own conditional velocity fields.

## Method Overview

The first version computed primitive flows from standalone primitive prompts such as `"four cyborg dogs"` or `"two purple cats"`:

```text
primitive flow = f_theta(x_t, t, p_i)
```

That was brittle. A standalone primitive prompt does not live in the same conditional distribution as the anchor scene, so its residual can pull the denoising trajectory into a different scene.

AIM-Flow v2 fixes this by preserving the anchor in every primitive condition:

- `A`: an anchor prompt describing the global scene destination
- `P`: the full prompt
- `p_i`: a primitive prompt describing a count, object, attribute, relation, or action
- `q_i = A \oplus p_i`: the anchor-augmented primitive prompt, implemented as `A + ", " + p_i`

At each sampling step, AIM-Flow computes:

```text
q_i = A \oplus p_i

v_A = f_theta(x_t, t, A)
v_F = f_theta(x_t, t, P)
v_i = f_theta(x_t, t, q_i)

Delta_i = v_i - v_A
Delta_F = v_F - v_A

g_i(t) = clamp((cos(Delta_i, Delta_F) - tau) / (1 - tau), 0, 1)

v_candidate = v_A + lambda(t) * ClipNorm(sum_i alpha_i s_i(t) g_i(t) Delta_i)
```

`Delta_F` is an internal self-consistency direction: the model's own full-prompt residual relative to the anchor. It is not a reward model, VQA score, CLIP objective, or image-level judge.

### VFA: Velocity Field Aggregation

VFA aggregates anchor-preserved primitive residuals. Each primitive receives:

- a base primitive weight `alpha_i`
- a schedule weight `s_i(t)`
- a smooth compatibility gate `g_i(t)` based on agreement with `Delta_F`

The aggregate correction is clipped relative to `||v_A||` before it is added to the anchor velocity.

### LTP: Latent Trajectory Projection

LTP keeps the steered update close to the anchor trajectory. The sampler computes both:

```text
x_next_anchor = scheduler.step(v_A, t, x_t)
x_next_candidate = scheduler.step(v_candidate, t, x_t)
```

Then it limits the candidate offset around the anchor step:

```text
candidate_offset = x_next_candidate - x_next_anchor
max_offset_norm = radius(t) * ||x_next_anchor - x_t||
x_next = x_next_anchor + ClipNorm(candidate_offset, max_offset_norm)
```

If a scheduler cannot safely compute both updates without mutating state, AIM-Flow can fall back to velocity-level LTP and records that in metadata.

This repository supports:

- `base`: normal SD3 full-prompt generation
- `anchor`: SD3 anchor-only generation
- `naive_v1`: the old standalone primitive method, kept for failure comparison
- `aim_v2`: anchor-augmented primitives with VFA and LTP
- `full`: alias for `aim_v2`

## What This Prototype Does Not Use

AIM-Flow v2 is deliberately narrow:

- no VQA models
- no CLIP reward optimization
- no external image-level judges during sampling
- no training or fine-tuning
- no LLM decomposer in the first version

Prompt decompositions are manually provided in YAML so the steering mechanism can be studied in isolation.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/aim-flow.git
cd aim-flow
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

For inference without editable install:

```bash
pip install -r requirements.txt
```

Stable Diffusion 3 Medium is gated. You must accept the Stability AI model license on Hugging Face and provide a token:

```bash
export HF_TOKEN=hf_...
```

On Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_..."
```

## Kaggle Usage

This project is designed to run on Kaggle with a 16 GB VRAM GPU by defaulting to:

- 512x512 generation
- 24 inference steps
- `float16`
- model CPU offload
- VAE slicing
- sequential primitive forward passes
- SD3's T5 text encoder disabled by default (`text_encoder_3=None`) to reduce memory pressure

Kaggle P100 sessions may ship with a PyTorch wheel that no longer supports
Pascal GPUs (`sm_60`). The demo notebook force-reinstalls:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir --force-reinstall torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

If the GPU check still reports that `sm_60` is unsupported, restart the Kaggle
runtime/kernel and rerun the install and GPU check cells before loading SD3.

Open [notebooks/kaggle_aim_flow_demo.ipynb](notebooks/kaggle_aim_flow_demo.ipynb), set:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/aim-flow.git"
HF_TOKEN = secret_value_0
```

Then run the notebook. The demo reads a Kaggle secret named `Huggingface`.

## GitHub-to-Kaggle Workflow

1. Push this repository to GitHub.
2. Create a Kaggle notebook with GPU enabled.
3. Add your Hugging Face token as a Kaggle secret named `Huggingface`.
4. Paste your GitHub repository URL into the notebook.
5. Run the cells. The notebook clones the repo into `/kaggle/working/aim-flow`.

## CLI Examples

Run the default comparison:

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-key cyborg_dogs \
  --output-dir outputs/cyborg_dogs \
  --modes base anchor naive_v1 aim_v2
```

Run only the base SD3 full-prompt baseline:

```bash
python scripts/run_base.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-key cyborg_dogs \
  --output-dir outputs/base
```

Run AIM-Flow only:

```bash
python scripts/run_aim_flow.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-key cyborg_dogs \
  --mode aim_v2 \
  --output-dir outputs/aim_v2
```

Useful comparison overrides:

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-key cyborg_dogs \
  --output-dir outputs/cyborg_dogs \
  --modes base anchor naive_v1 aim_v2 \
  --ltp-mode velocity \
  --num-inference-steps 16 \
  --height 384 --width 384
```

## Expected Outputs

The comparison run creates:

- `base_full_prompt.png`
- `anchor_only.png`
- `naive_v1_standalone_primitives.png`
- `aim_v2_anchor_augmented_vfa_ltp.png`
- `comparison_grid.png`
- `metadata_base.json`
- `metadata_anchor.json`
- `metadata_naive_v1.json`
- `metadata_aim_v2.json`

AIM-Flow metadata includes per-step effective weights, cosine similarities, gates, correction norms, LTP diagnostics, primitive original text, and primitive anchor-augmented text.

## Sample Prompt Decomposition

```yaml
full_prompt: "four cyborg dogs jumping over grass, each wearing bright orange hats, while two purple cats chase them, cinematic lighting, detailed digital art"
anchor_prompt: "cyborg dogs being chased by cats on grass, cinematic lighting, detailed digital art"
primitive_prompts:
  - text: "four cyborg dogs"
    type: "count_entity"
    weight: 0.85
    schedule: "early_middle"
  - text: "cyborg dogs wearing bright orange hats"
    type: "attribute_binding"
    weight: 0.75
    schedule: "middle_late"
    anchor_augmented_text: "cyborg dogs being chased by cats on grass, each cyborg dog wearing a bright orange hat, cinematic lighting, detailed digital art"
  - text: "two purple cats"
    type: "count_entity"
    weight: 0.80
    schedule: "early_middle"
  - text: "purple cats chasing cyborg dogs"
    type: "relation_action"
    weight: 0.70
    schedule: "middle"
  - text: "cyborg dogs jumping over grass"
    type: "relation_action"
    weight: 0.65
    schedule: "middle"
```

## Development and Tests

The unit tests cover the scheduler, prompt schema, manual decomposition loading, and tensor aggregation math. They do not load SD3 and do not require a GPU.

```bash
pip install -e ".[dev]"
pytest -q
```

## Limitations

- The custom AIM-Flow loop uses conditional predictions for v2. The base pipeline still uses the standard Diffusers pipeline path.
- Exact CFG-compatible multi-condition aggregation is a planned extension.
- The Kaggle config disables SD3's T5 encoder for memory. Set `model.load_t5_text_encoder: true` in the YAML on larger GPUs for the full three-encoder SD3 prompt stack.
- Manual decompositions are required; the rule-based fallback is only for smoke tests.
- Qualitative comparison is provided, but no VQA, reward model, or image judge is used.
- Primitive residuals can increase runtime roughly linearly with the number of primitives.
- Latent LTP depends on scheduler calls being safely repeatable. If that is not true for a scheduler version, use `--ltp-mode velocity`.

## Citation

```bibtex
@misc{aimflow2026,
  title = {Your Model Knows What You're Aiming For: AIM-Flow for Image Generation by Rectified Models},
  author = {AIM-Flow contributors},
  year = {2026},
  note = {Early research prototype}
}
```

## License and Model License

This repository code is released under the MIT License. Stable Diffusion 3 Medium is not included in this repository and is governed by its own Stability AI license. Users must accept the model terms and have access to `stabilityai/stable-diffusion-3-medium-diffusers` on Hugging Face before running inference.
