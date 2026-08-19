# Sparse Primitive Flow Composition for Rectified Text-to-Image Generation

> **Training-free, inference-time compositional steering of rectified-flow text-to-image diffusion models via sparse velocity field aggregation.**

---

## Abstract

We present **Sparse Primitive Flow Composition (SPFC)**, a training-free, inference-time method for improving compositional fidelity in rectified-flow text-to-image models such as Stable Diffusion 3. Current text-to-image generators struggle with compositionally complex prompts — those requiring correct attribute binding (e.g., *green* boots on a *marble* fox), numeracy (counting objects), spatial relations, and multi-concept scenes. SPFC addresses this by decomposing a complex target prompt into a set of *primitive* prompts, each isolating a single semantic concept. At sparsely selected denoising timesteps, we evaluate all primitive velocity fields alongside the target prompt velocity field, aggregate them via a temperature-scaled softmax weighted by consensus and target-consistency gating, clip the resulting correction, and project the latent update onto a trust region around the target trajectory using **Latent Trajectory Projection (LTP)**. At all other timesteps, the model follows the unmodified target prompt. The method uses **no VQA, no CLIP reward, no external judge, no training, no fine-tuning, and no image feedback** — only the model's own conditional velocity fields.

---

## 1. Introduction and Motivation

Rectified-flow text-to-image models (e.g., Stable Diffusion 3) learn a velocity field $v_\theta(x_t, t, c)$ that transports samples from a noise distribution to the data distribution conditioned on a text prompt $c$. During inference, an ODE solver integrates this velocity field over a discrete schedule of timesteps $\{t_0, t_1, \ldots, t_{T-1}\}$ to produce a latent $x_0$, which is then decoded by the VAE.

For simple prompts, this works well. However, when prompts become compositionally complex — combining multiple objects, attributes, spatial relationships, and fine-grained visual details — the model's single-pass conditioning often fails to faithfully bind all concepts. This manifests as:

- **Attribute leakage**: colours or textures applied to the wrong object
- **Missing concepts**: entire sub-concepts (e.g., a shadow, a reflection) are omitted
- **Numeracy failures**: wrong object counts
- **Spatial errors**: incorrect relative positions

### 1.1 Design Principles

SPFC is built on four principles:

1. **Target-prompt primacy.** The full target prompt is always the primary generation path. Primitives *nudge* the trajectory; they do not replace it.
2. **Sparse intervention.** Aggregation is performed only at selected timesteps, minimizing computational overhead and avoiding over-steering.
3. **Self-consistency.** All gating signals derive from the model's own velocity fields — no external models or reward signals.
4. **Bounded deviation.** Latent Trajectory Projection ensures the final image remains within a controlled neighbourhood of what the target prompt alone would produce.

---

## 2. Background: Rectified Flow Sampling

A rectified-flow model parameterises a velocity field $v_\theta(x_t, t, c)$ where $x_t \in \mathbb{R}^{B \times C \times H \times W}$ is the noisy latent at timestep $t$, and $c$ is a text conditioning embedding. The Euler integration step at step index $n$ with timestep $t_n$ is:

$$
x_{t_{n+1}} = x_{t_n} + (t_{n+1} - t_n) \cdot v_\theta(x_{t_n}, t_n, c)
$$

In practice, the scheduler (e.g., `FlowMatchEulerDiscreteScheduler`) encapsulates this update as:

$$
x_{\text{next}} = \text{Step}(x_t,\; v_\theta(x_t, t, c),\; t)
$$

For SD3 specifically, classifier-free guidance (CFG) is applied:

$$
\tilde{v} = v_\theta(x_t, t, \varnothing) + s \cdot \bigl(v_\theta(x_t, t, c) - v_\theta(x_t, t, \varnothing)\bigr)
$$

where $s$ is the guidance scale and $\varnothing$ is the null/negative conditioning. All velocity fields in SPFC are CFG-guided.

---

## 3. Problem Formulation

**Given:**

- A pre-trained rectified-flow model $v_\theta$
- A compositionally complex **target prompt** $P$
- A simpler **source prompt** $S_0$ (a reduced version of $P$ with fewer attributes/concepts)
- A set of **primitive prompts** $\{S_1, S_2, \ldots, S_K\}$, each a complete sentence isolating one semantic concept from $P$
- A discrete denoising schedule of $T$ timesteps
- A set of **aggregation step indices** $\mathcal{A} \subset \{0, 1, \ldots, T-1\}$

**Goal:** Generate an image that faithfully represents *all* concepts in $P$ — correct attribute binding, object counts, spatial relations, reflections, shadows, etc. — while remaining close to what $P$ alone would produce.

**Constraints:**

- No additional training or fine-tuning of $v_\theta$
- No external reward models (CLIP, VQA, aesthetic classifiers)
- No image-space feedback during sampling
- Only the model's own conditional velocity field evaluations

---

## 4. Method

### 4.1 Prompt Decomposition

The user manually decomposes $P$ into a **source prompt** and a set of **primitive prompts**. Each primitive is a *complete, natural-language sentence* (not a fragment) that emphasises one semantic aspect.

**Running Example — "Reflective Shadow Fox":**

| Symbol | Role | Prompt Text |
|--------|------|-------------|
| $P$ | **Target** | *"A marble statue of a fox wearing green rain boots standing in front of a cracked mirror, with its shadow forming the shape of a dragon on the wall, cinematic lighting, realistic photo, sharp focus"* |
| $S_0$ | **Source** | *"A marble statue of a fox standing in front of a cracked mirror, cinematic lighting, realistic photo, sharp focus"* |
| $S_1$ | Primitive | *"A marble statue of a fox standing in front of a wall, cinematic lighting, realistic photo, sharp focus"* |
| $S_2$ | Primitive | *"A fox statue wearing green rain boots, cinematic lighting, realistic photo, sharp focus"* |
| $S_3$ | Primitive | *"A marble fox statue reflected in a mirror perfectly, cinematic lighting, realistic photo, sharp focus"* |
| $S_4$ | Primitive | *"A marble fox statue reflected in a cracked mirror with multiple veins of cracks, cinematic lighting, realistic photo, sharp focus"* |
| $S_5$ | Primitive | *"The shadow of a fox statue forms on a wall opposite the cracked mirror, cinematic lighting, realistic photo, sharp focus"* |

Each primitive isolates exactly one concept: the marble fox itself ($S_1$), the green boots ($S_2$), the mirror reflection ($S_3$), the cracked mirror detail ($S_4$), and the shadow ($S_5$). All primitives share the scene context (lighting, style) to maintain coherence.

### 4.2 Condition List Construction

From the decomposition, we build an ordered **condition list** $\mathcal{C}$:

$$
\mathcal{C} = [S_0,\; S_1,\; S_2,\; \ldots,\; S_K,\; P]
$$

Each condition $i$ carries a **base weight** $w_i$:

| Index | Name | Role | Base Weight $w_i$ |
|-------|------|------|--------------------|
| 0 | source | source | 0.7 |
| 1 | $S_1$ marble fox statue | primitive | 1.0 |
| 2 | $S_2$ green rain boots | primitive | 1.0 |
| 3 | $S_3$ mirror reflection | primitive | 1.0 |
| 4 | $S_4$ cracked mirror | primitive | 1.0 |
| 5 | $S_5$ shadow | primitive | 1.0 |
| 6 | $P$ (target) | target | 1.2 |

The target prompt receives a slightly elevated base weight ($w_P = 1.2$) to encode its primacy. The source prompt receives a reduced weight ($w_{S_0} = 0.7$) since it is an intentionally simplified scene.

### 4.3 Aggregation Step Selection

Not every denoising step performs aggregation. We define a set of **aggregation step indices** $\mathcal{A} \subset \{0, \ldots, T-1\}$. Three modes are supported:

1. **Sparse** (default): Explicitly listed steps, e.g. $\mathcal{A} = \{12, 16, 20, 23\}$
2. **Dense**: Every step, $\mathcal{A} = \{0, 1, \ldots, T-1\}$
3. **Final-only**: Only the last step, $\mathcal{A} = \{T-1\}$

In the fox example run, dense aggregation was used: $\mathcal{A} = \{1, 2, \ldots, 20, 23\}$ (steps 21–22 excluded, step 0 excluded).

### 4.4 Main Denoising Loop

The complete algorithm is:

---

**Algorithm 1: Sparse Primitive Flow Composition**

**Input:** Model $v_\theta$, target prompt $P$, condition list $\mathcal{C} = [S_0, \ldots, S_K, P]$, base weights $\{w_i\}$, aggregation steps $\mathcal{A}$, temperature $\tau$, clip ratio $\rho$, LTP radius ratio $r$, total steps $T$

**Output:** Decoded image

1. Encode all conditions: $c_i \leftarrow \text{TextEncode}(S_i)$ for each $i \in \{0, \ldots, K, P\}$
2. $x_{t_0} \sim \mathcal{N}(0, I)$ with fixed seed
3. **for** $n = 0$ **to** $T - 1$ **do**
4. $\quad$ **if** $n \notin \mathcal{A}$ **then**
5. $\quad\quad$ $v_P \leftarrow v_\theta(x_{t_n}, t_n, c_P)$ — *target-only step*
6. $\quad\quad$ $x_{t_{n+1}} \leftarrow \text{Step}(x_{t_n}, v_P, t_n)$
7. $\quad$ **else**
8. $\quad\quad$ $v_i \leftarrow v_\theta(x_{t_n}, t_n, c_i)$ for all $i \in \{0, \ldots, K+1\}$ — *evaluate all conditions*
9. $\quad\quad$ $v_{\text{VFA}} \leftarrow \text{VFA}(\{v_i\}, \{w_i\}, v_P, \tau, \rho)$ — *§4.5: Velocity Field Aggregation*
10. $\quad\quad$ $x_{t_{n+1}} \leftarrow \text{LTP}(x_{t_n}, v_P, v_{\text{VFA}}, r, t_n)$ — *§4.6: Latent Trajectory Projection*
11. $\quad$ **end if**
12. **end for**
13. **return** $\text{VAE.Decode}(x_{t_T})$

---

At non-aggregation steps (line 4–6), the model simply follows the target prompt — this is identical to standard SD3 generation. The method intervenes only at aggregation steps.

### 4.5 Velocity Field Aggregation (VFA)

VFA computes a weighted combination of all condition velocity fields. It proceeds in four stages: **(a)** pairwise consensus gating, **(b)** target-consistency gating, **(c)** temperature-scaled softmax weighting, and **(d)** norm-bounded correction clipping.

#### 4.5.1 Pairwise Cosine Consensus Matrix

First, we compute the pairwise cosine similarity between all $N = K + 2$ condition velocity fields (source + $K$ primitives + target):

$$
M_{ij} = \frac{\langle \text{flat}(v_i),\; \text{flat}(v_j) \rangle}{\|\text{flat}(v_i)\| \cdot \|\text{flat}(v_j)\|}
$$

where $\text{flat}(\cdot)$ flattens the $C \times H \times W$ dimensions into a single vector.

**Fox example, step 1** ($t = 985.12$): The $7 \times 7$ pairwise cosine matrix is:

|  | source | $S_1$ | $S_2$ | $S_3$ | $S_4$ | $S_5$ | target |
|---|--------|-------|-------|-------|-------|-------|--------|
| **source** | 1.000 | 0.895 | 0.855 | 0.979 | 0.982 | 0.896 | 0.930 |
| **$S_1$** | 0.895 | 1.000 | 0.691 | 0.874 | 0.849 | 0.954 | 0.918 |
| **$S_2$** | 0.855 | 0.691 | 1.000 | 0.855 | 0.839 | 0.681 | 0.806 |
| **$S_3$** | 0.979 | 0.874 | 0.855 | 1.000 | 0.955 | 0.856 | 0.917 |
| **$S_4$** | 0.982 | 0.849 | 0.839 | 0.955 | 1.000 | 0.866 | 0.898 |
| **$S_5$** | 0.896 | 0.954 | 0.681 | 0.856 | 0.866 | 1.000 | 0.918 |
| **target** | 0.930 | 0.918 | 0.806 | 0.917 | 0.898 | 0.918 | 1.000 |

Note that $S_2$ (green rain boots) has the lowest agreement with other conditions (0.681–0.855), reflecting its semantically distinct content. The mirror-related primitives ($S_3$, $S_4$) are very similar to the source (0.955–0.982).

#### 4.5.2 Consensus Gating

For each condition $i$, its **consensus gate** $g_i^{\text{cons}}$ measures mean positive agreement with all other conditions:

$$
g_i^{\text{cons}} = \text{clamp}\Bigl(\frac{1}{N-1} \sum_{j \neq i} \max(0,\; M_{ij}),\;\; g_{\min},\; g_{\max}\Bigr)
$$

**Fox example, step 1:**

| Condition | $g_i^{\text{cons}}$ |
|-----------|---------------------|
| source | 0.923 |
| $S_1$ marble fox | 0.863 |
| $S_2$ green boots | **0.788** |
| $S_3$ mirror | 0.906 |
| $S_4$ cracked mirror | 0.898 |
| $S_5$ shadow | 0.862 |
| target $P$ | 0.898 |

The green boots primitive receives the lowest consensus gate (0.788), indicating its velocity field diverges most from the group. This automatically down-weights semantically risky contributions.

#### 4.5.3 Target-Consistency Gating

Each condition's velocity is further gated by its cosine similarity with the target prompt velocity:

$$
g_i^{\text{tgt}} = \text{clamp}\bigl(\max(0,\; \cos(v_i, v_P)),\;\; g_{\min},\; g_{\max}\bigr)
$$

**Fox example, step 1:**

| Condition | $\cos(v_i, v_P)$ | $g_i^{\text{tgt}}$ |
|-----------|-------------------|---------------------|
| source | 0.930 | 0.930 |
| $S_1$ | 0.918 | 0.918 |
| $S_2$ | **0.806** | **0.806** |
| $S_3$ | 0.917 | 0.917 |
| $S_4$ | 0.899 | 0.899 |
| $S_5$ | 0.918 | 0.918 |
| target $P$ | 1.000 | 1.000 |

Again, the boots primitive ($S_2$) receives the lowest target-consistency gate.

#### 4.5.4 Raw Score Computation

The **raw score** for each condition combines its base weight with both gates:

$$
r_i = w_i \cdot g_i^{\text{cons}} \cdot g_i^{\text{tgt}}
$$

A floor is applied to the target score to guarantee it always receives substantial weight:

$$
r_P \leftarrow \max(r_P,\; \epsilon) \quad \text{where } \epsilon = 10^{-4}
$$

**Fox example, step 1:**

| Condition | $w_i$ | $g_i^{\text{cons}}$ | $g_i^{\text{tgt}}$ | $r_i$ |
|-----------|--------|---------------------|---------------------|--------|
| source | 0.70 | 0.923 | 0.930 | **0.601** |
| $S_1$ | 1.00 | 0.863 | 0.918 | **0.793** |
| $S_2$ | 1.00 | 0.788 | 0.806 | **0.635** |
| $S_3$ | 1.00 | 0.906 | 0.917 | **0.831** |
| $S_4$ | 1.00 | 0.898 | 0.899 | **0.807** |
| $S_5$ | 1.00 | 0.862 | 0.918 | **0.791** |
| target $P$ | 1.20 | 0.898 | 1.000 | **1.078** |

#### 4.5.5 Temperature-Scaled Softmax Weights

The raw scores are converted to a probability distribution via temperature-scaled softmax:

$$
\alpha_i = \frac{\exp(r_i / \tau)}{\sum_j \exp(r_j / \tau)}
$$

with temperature $\tau = 0.7$. Lower temperature sharpens the distribution toward higher-scoring conditions.

**Fox example, step 1** ($\tau = 0.7$):

| Condition | $r_i$ | $\alpha_i$ |
|-----------|--------|------------|
| source | 0.601 | **0.107** |
| $S_1$ | 0.793 | **0.140** |
| $S_2$ | 0.635 | **0.112** |
| $S_3$ | 0.831 | **0.148** |
| $S_4$ | 0.807 | **0.143** |
| $S_5$ | 0.791 | **0.140** |
| target $P$ | 1.078 | **0.211** |

The target prompt receives the largest weight (0.211), followed by the mirror reflection ($S_3$, 0.148). The source and boots primitives receive the smallest weights.

#### 4.5.6 Weighted Velocity Aggregation and Correction Clipping

The aggregated velocity is a weighted sum:

$$
v_{\text{raw}} = \sum_i \alpha_i \cdot v_i
$$

We then compute the **correction** relative to the target velocity and clip its norm:

$$
\delta = v_{\text{raw}} - v_P
$$

$$
\delta_{\text{clip}} = \begin{cases} \delta & \text{if } \|\delta\| \leq \rho \cdot \|v_P\| \\[4pt] \frac{\delta}{\|\delta\|} \cdot \rho \cdot \|v_P\| & \text{otherwise} \end{cases}
$$

$$
v_{\text{VFA}} = v_P + \delta_{\text{clip}}
$$

where $\rho$ is the velocity clip ratio (default 0.50). This ensures the aggregated velocity never deviates by more than $\rho \cdot \|v_P\|$ from the target velocity.

**Fox example, step 1:** $\|v_P\| = 603.05$, $\|\delta\| = 147.52$, $\rho \cdot \|v_P\| = 301.53$. Since $147.52 < 301.53$, no clipping was needed — the correction passed through unmodified.

### 4.6 Latent Trajectory Projection (LTP)

After VFA produces the aggregated velocity $v_{\text{VFA}}$, LTP ensures the resulting latent update stays within a trust region of the target-prompt trajectory. This is the final safety mechanism preventing the primitives from destabilising generation.

#### 4.6.1 Latent-Space LTP (Default Mode)

The preferred mode operates directly on the scheduler's latent outputs:

1. Compute the **target next latent**: $x_{\text{next}}^{P} = \text{Step}(x_t, v_P, t)$
2. Compute the **candidate next latent**: $x_{\text{next}}^{\text{VFA}} = \text{Step}(x_t, v_{\text{VFA}}, t)$
3. Compute the **offset**: $\Delta = x_{\text{next}}^{\text{VFA}} - x_{\text{next}}^{P}$
4. Compute the **target step norm**: $d = \|x_{\text{next}}^{P} - x_t\|$
5. **Project** the offset onto a ball of radius $r \cdot d$:

$$
\Delta_{\text{proj}} = \begin{cases} \Delta & \text{if } \|\Delta\| \leq r \cdot d \\[4pt] \frac{\Delta}{\|\Delta\|} \cdot r \cdot d & \text{otherwise} \end{cases}
$$

6. Final update: $x_{t+1} = x_{\text{next}}^{P} + \Delta_{\text{proj}}$

The radius ratio $r$ (default 0.35) controls how far the candidate may deviate from the target trajectory at each step.

**Fox example, step 1:** Target step norm $d = 9.524$, candidate offset $\|\Delta\| = 2.332$, max allowed $r \cdot d = 3.333$. Since $2.332 < 3.333$, no projection was needed.

**Fox example, step 23 (final):** Target step norm $d = 3.743$, candidate offset $\|\Delta\| = 0.115$, max allowed $r \cdot d = 1.310$. The offset is tiny — by the final step, all velocities have converged and the correction is negligible.

#### 4.6.2 Velocity-Space LTP (Fallback)

If the scheduler's `step()` function is not safely replayable (i.e., calling it twice mutates internal state), LTP falls back to velocity space:

$$
\delta_v = v_{\text{VFA}} - v_P, \quad \delta_{v,\text{proj}} = \text{clip\_norm}(\delta_v,\; r \cdot \|v_P\|)
$$

$$
v_{\text{final}} = v_P + \delta_{v,\text{proj}}
$$

The system automatically probes the scheduler before sampling to determine which mode is safe.

### 4.7 Temporal Dynamics: From Diversity to Convergence

A key empirical property of SPFC is the natural convergence of velocity fields over the denoising trajectory. Early steps show significant diversity between condition velocities; late steps show near-unity agreement.

**Fox example — evolution across denoising:**

| Step | Timestep | $\|v_P\|$ | $\|\delta\|$ | Min pairwise cos | Min $g^{\text{tgt}}$ | Target $\alpha_P$ | LTP offset |
|------|----------|-----------|--------------|-------------------|----------------------|-------------------|------------|
| 0 | 1000.0 | — | — | — | — | — | — (target-only) |
| 1 | 985.1 | 603.1 | 147.5 | 0.681 | 0.806 | 0.211 | 2.33 |
| 2 | 969.3 | 548.1 | 158.1 | 0.739 | 0.801 | 0.206 | 2.66 |
| 10 | mid | ~530 | ~140 | ~0.92 | ~0.94 | ~0.20 | ~2.0 |
| 20 | 329.1 | 514.3 | 33.0 | 0.994 | 0.995 | 0.191 | ~0.3 |
| 21–22 | — | — | — | — | — | — | — (target-only) |
| 23 | 8.9 | 419.1 | 7.3 | 0.999 | 0.999 | 0.191 | 0.11 |

**Key observations:**

1. **Early steps** ($t \approx 1000$): Condition velocities are diverse (min pairwise cos ≈ 0.68). The method has maximum steering influence here, establishing global composition.
2. **Mid steps** ($t \approx 500$): Velocities converge (min cos > 0.92). The correction magnitude decreases. Fine details are being negotiated.
3. **Late steps** ($t < 100$): All velocities are nearly identical (min cos > 0.999). The correction is negligible ($\|\delta\| = 7.3$ vs $\|v_P\| = 419.1$). The image is essentially locked in.

This convergence is a natural property of rectified flows — it is not engineered. The method exploits it: early aggregation steps steer composition, late steps confirm without disrupting.

---

## 5. Results

### 5.1 Reflective Shadow Fox

**Target prompt:** *"A marble statue of a fox wearing green rain boots standing in front of a cracked mirror, with its shadow forming the shape of a dragon on the wall, cinematic lighting, realistic photo, sharp focus"*

**Settings:** SD3 Medium · 512×1024 · 24 steps · seed 42 · guidance 4.5 · 7 conditions (source + 5 primitives + target)

#### Comparison Grid

![Comparison grid: base vs source-only vs SPFC sparse](results/reflective%20shadow%20fox/comparison_grid(3).png)

*Left to right: base generation (target prompt alone), source-only generation, SPFC sparse result.*

#### Individual Outputs

| Base (target prompt alone) | Source only |
|:--------------------------:|:-----------:|
| ![Base generation](results/reflective%20shadow%20fox/base_full_target(1).png) | ![Source only](results/reflective%20shadow%20fox/source_only.png) |

| SPFC Sparse |
|:-----------:|
| ![SPFC sparse result](results/reflective%20shadow%20fox/primitive_flow_sparse.png) |

#### What to Look For

- **Green rain boots** on the fox statue — a fine-grained attribute binding that standard generation frequently drops or misplaces
- **Cracked mirror** with realistic fracture lines — the cracking detail isolated in $S_4$ is faithfully rendered
- **Dragon-shaped shadow** on the wall — the most semantically unusual concept, requiring $S_5$ (shadow primitive) to anchor correctly (all methods fail so far, proving this to be too semantically challenging as of right now.)
- The **marble texture** and **cinematic lighting** are preserved from the target prompt without degradation

The base generation (left) typically produces the fox and mirror but loses the boots and/or fails to render the dragon shadow. SPFC (right) steers all five primitives into the final image while remaining visually consistent with the target prompt's style.

---

## 6. Hyperparameters

| Parameter | Symbol | Default | Role |
|-----------|--------|---------|------|
| VFA temperature | $\tau$ | 0.7 | Controls softmax sharpness; lower = more peaked |
| Velocity clip ratio | $\rho$ | 0.50 | Max correction norm as fraction of $\|v_P\|$ |
| LTP radius ratio | $r$ | 0.35 | Max latent offset as fraction of target step norm |
| Source weight | $w_{S_0}$ | 0.7 | Base weight for the source prompt |
| Target weight | $w_P$ | 1.2 | Base weight for the target prompt |
| Max primitives | — | 5 | Memory cap on concurrent primitive evaluations |
| Guidance scale | $s$ | 4.5 | Standard CFG strength |
| Inference steps | $T$ | 24 | Number of Euler steps |

---

## 7. Installation

```bash
git clone https://github.com/Soobiwan/aim-flow.git
cd aim-flow
pip install -e ".[dev]"
```

Stable Diffusion 3 Medium is gated. Accept the model license on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) and set `HF_TOKEN`.

### 7.1 Kaggle

The Kaggle config defaults to 512×512, 24 steps, `float16`, CPU offload, VAE slicing, sequential condition forwards, and `text_encoder_3=None` for memory.

Kaggle P100 sessions may need a Pascal-compatible PyTorch wheel:

```bash
pip uninstall -y torch torchvision torchaudio
pip install --no-cache-dir --force-reinstall torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Then restart the runtime before loading SD3.

---

## 8. Usage

### 8.1 Sparse Aggregation (Recommended)

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key reflection_shadow_fox \
  --output-dir outputs/fox_sparse \
  --modes base source_only primitive_flow_sparse \
  --aggregation-steps 12 16 20 23
```

### 8.2 Dense Aggregation

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key reflection_shadow_fox \
  --output-dir outputs/fox_dense \
  --modes base source_only primitive_flow_sparse \
  --aggregate-every-n-steps 1
```

### 8.3 Final-Only Aggregation

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key reflection_shadow_fox \
  --output-dir outputs/fox_final_only \
  --modes base source_only primitive_flow_final_only \
  --final-only
```

### 8.4 Memory-Constrained Settings

```bash
python scripts/run_compare.py \
  --config configs/sd3_medium_kaggle.yaml \
  --prompts configs/sample_prompts.yaml \
  --prompt-section primitive_flow_prompts \
  --prompt-key reflection_shadow_fox \
  --output-dir outputs/fox_small \
  --modes base source_only primitive_flow_sparse \
  --num-inference-steps 16 \
  --height 384 --width 384 \
  --ltp-mode velocity \
  --max-primitives 2
```

### 8.5 Marginal Flow (Experimental, Independent of SPFC)

Marginal Flow follows one full-target latent trajectory. At configured sparse
steps it compares the guided full-target prediction with guided predictions for
explicit contextual ablations, solves a small simplex min-norm problem over the
normalized residuals, removes only target-opposing correction, and applies a
trust-ratio norm limit. It does not call SPFC aggregation, SPFC gating, or LTP.

Prompt entries use `target_prompt` plus primitives containing human-readable
`primitive` metadata and an explicit `ablated_prompt`; no source prompt or
standalone primitive generation prompt is required.

```bash
python scripts/run_compare.py \
  --config configs/marginal_flow_diagnostic.yaml \
  --prompts configs/marginal_flow_prompts.yaml \
  --prompt-section marginal_flow_prompts \
  --prompt-key red_cube_blue_sphere \
  --output-dir outputs/marginal_flow_diagnostic/marginal_flow \
  --modes marginal_flow
```

The diagnostic config uses 8 steps at 256×256 for a quick end-to-end check.
For experiments, increase the resolution/step count and select roughly 3–5
early/mid `marginal_flow.intervention_steps`.

---

## 9. Outputs

Each run produces:

| File | Description |
|------|-------------|
| `base_full_target.png` | Standard SD3 generation with the target prompt |
| `source_only.png` | Standard SD3 generation with the source prompt |
| `primitive_flow_sparse.png` | SPFC result |
| `comparison_grid.png` | Side-by-side comparison grid |
| `metadata_*.json` | Full diagnostics: per-step VFA weights, consensus gates, target-consistency gates, pairwise cosine matrices, LTP norms, correction magnitudes |

The metadata JSON files contain complete reproducibility information: every softmax weight, every gating value, every norm at every aggregation step.

---

## 10. Tests

Tests are CPU-only and do not load SD3:

```bash
pytest -q
```

The test suite validates VFA aggregation math, LTP projection, consensus gating, Marginal Flow simplex/balancing/trust math, prompt schema parsing, sparse step selection, and scheduler compatibility probing.

---

## 11. Limitations

- Custom SD3 denoising depends on Diffusers API compatibility; scheduler internal changes may require updates.
- Sparse flow composition is slower than base generation (each aggregation step evaluates $N$ velocity fields instead of 1).
- Final-only aggregation may be too late for structural changes that require early-step intervention.
- Evaluation is currently qualitative; benchmark integration (T2I-CompBench, GenEval) is planned.
- Latent LTP falls back to velocity LTP when the scheduler's `step()` is not safely replayable.
- Prompt decomposition is manual; automated decomposition is future work.

---

## 12. License

Repository code is MIT licensed. Stable Diffusion 3 Medium is not included and is governed by Stability AI's model license.
