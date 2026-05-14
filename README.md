# Your Model Knows What You're Aiming For: AIM-Flow for Image Generation by Rectified Models

**AIM-Flow** (**Anchor-Informed Modular Flow Guidance**) is an early research prototype for training-free, inference-time steering of rectified-flow-style text-to-image models such as Stable Diffusion 3 Medium.

The central idea is simple: instead of asking the model to satisfy one dense prompt all at once, AIM-Flow decomposes a prompt into a global anchor and several semantic primitives, then aggregates primitive flow residuals during sampling.

## Method Overview

Given a full prompt `P`, provide:

- `A`: an anchor prompt describing the global scene destination
- `p_i`: primitive prompts describing counts, objects, attributes, relations, and actions

At each sampling step, AIM-Flow computes:

```text
delta_i = v_i - v_anchor
v_agg = v_anchor + lambda_global * sum_i w_i(t) * stabilized(delta_i)
```

where `v_anchor` is the SD3 rectified-flow prediction under the anchor prompt and `v_i` is the prediction under primitive prompt `p_i`. The aggregate prediction `v_agg` is passed to the scheduler instead of the standard full-prompt prediction.

This repository supports:

- `base`: normal SD3 full-prompt generation
- `anchor`: SD3 anchor-only generation
- `naive`: anchor plus weighted primitive residuals
- `full`: scheduled residuals with cosine conflict gating and norm clipping

## What This Prototype Does Not Use

AIM-Flow v1 is deliberately narrow:

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
HF_TOKEN = ""
```

Then run the notebook. If `HF_TOKEN` is empty, the notebook tries to read a Kaggle secret named `HF_TOKEN`.

## GitHub-to-Kaggle Workflow

1. Push this repository to GitHub.
2. Create a Kaggle notebook with GPU enabled.
3. Add your Hugging Face token as a Kaggle secret named `HF_TOKEN`.
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
  --modes base anchor naive full
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
  --mode full \
  --output-dir outputs/full
```

## Expected Outputs

The comparison run creates:

- `base.png`
- `anchor.png`
- `naive.png`
- `full.png`
- `comparison_grid.png`
- one JSON metadata file per generated mode

AIM-Flow metadata includes per-step effective weights, cosine similarities, gates, correction norms, and the prompt decomposition.

## Sample Prompt Decomposition

```yaml
full_prompt: "four cyborg dogs jumping over grass, each wearing bright orange hats, while two purple cats chase them, cinematic lighting, detailed digital art"
anchor_prompt: "cyborg dogs being chased by cats on grass, cinematic lighting, detailed digital art"
primitive_prompts:
  - text: "four cyborg dogs"
    type: "count_entity"
    weight: 0.85
    schedule: "early"
  - text: "cyborg dogs wearing bright orange hats"
    type: "attribute_binding"
    weight: 0.75
    schedule: "late"
  - text: "two purple cats"
    type: "count_entity"
    weight: 0.80
    schedule: "early"
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
pytest
```

## Limitations

- The custom SD3 loop uses conditional predictions for AIM-Flow v1. The base pipeline still uses standard Diffusers classifier-free guidance.
- Exact CFG-compatible multi-condition aggregation is a planned extension.
- Manual decompositions are required; the rule-based fallback is only for smoke tests.
- Qualitative comparison is provided, but no VQA, reward model, or image judge is used.
- Primitive residuals can increase runtime roughly linearly with the number of primitives.

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
