# CLIP: Learning Transferable Visual Models From Natural Language Supervision

**Paper:** Radford, A., Kim, J.W., Hallacy, C., et al. (2021). [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
**Authors:** OpenAI
**Released:** January 2021

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Core Idea: Natural Language Supervision](#2-core-idea-natural-language-supervision)
3. [Training Data: WebImageText (WIT)](#3-training-data-webimagetext-wit)
4. [Architecture](#4-architecture)
   - [Dual-Encoder Design](#41-dual-encoder-design)
   - [Image Encoder: Vision Transformer](#42-image-encoder-vision-transformer)
   - [Image Encoder: Modified ResNet](#43-image-encoder-modified-resnet)
   - [Text Encoder](#44-text-encoder)
   - [Shared Embedding Space](#45-shared-embedding-space)
5. [Training Objective: Contrastive Learning](#5-training-objective-contrastive-learning)
   - [The Loss Function](#51-the-loss-function)
   - [Temperature Parameter](#52-temperature-parameter)
   - [Pseudocode](#53-pseudocode)
6. [Zero-Shot Transfer](#6-zero-shot-transfer)
   - [How It Works](#61-how-it-works)
   - [Prompt Engineering](#62-prompt-engineering)
   - [Prompt Ensembling](#63-prompt-ensembling)
7. [Key Results](#7-key-results)
   - [Zero-Shot Performance](#71-zero-shot-performance)
   - [Linear Probe Performance](#72-linear-probe-performance)
   - [Distribution Shift Robustness](#73-distribution-shift-robustness)
   - [Scaling Behavior](#74-scaling-behavior)
8. [Model Variants](#8-model-variants)
9. [Tokenization and Preprocessing](#9-tokenization-and-preprocessing)
10. [Parameter Initialization](#10-parameter-initialization)
11. [Limitations](#11-limitations)
12. [Impact and Significance](#12-impact-and-significance)
13. [Code-to-Paper Mapping](#13-code-to-paper-mapping)

---

## 1. Motivation and Problem Statement

Traditional computer vision systems are trained on fixed, human-curated label sets (e.g., ImageNet's 1,000 classes). This creates fundamental limitations:

- **Narrow supervision:** Models can only recognize categories present in training data.
- **Costly annotation:** Expanding the label space requires collecting and labeling new datasets.
- **Brittle generalization:** Models overfit to the training distribution and degrade under domain shift.
- **No compositionality:** Fixed labels cannot express novel visual concepts or their combinations.

The paper asks: can we train visual representations using **natural language** as supervision, bypassing fixed taxonomies entirely?

## 2. Core Idea: Natural Language Supervision

Instead of mapping images to a fixed set of N class labels, CLIP learns to map images and text into a **shared embedding space** where semantically related (image, text) pairs are close together.

This is not a new idea — prior work (e.g., ConVIRT, VirTex) explored language-supervised visual learning. CLIP's contribution is demonstrating that **scaling** this approach (400M image-text pairs, large models, contrastive pre-training) produces representations that transfer competitively to dozens of downstream tasks **without any task-specific training**.

Key insight from the paper: a contrastive objective is far more efficient than a generative (predictive) objective. The authors found that a contrastive model reached the same accuracy as a predictive model in **4x less compute**.

## 3. Training Data: WebImageText (WIT)

CLIP is trained on a dataset called **WIT** (WebImageText):

- **~400 million (image, text) pairs** collected from the internet
- Sourced from publicly available web pages, with additional data from existing datasets like YFCC100M
- The text consists of naturally-occurring image descriptions (alt-text, captions, titles)
- Filtered to exclude excessively violent or adult content
- Roughly comparable in total word count to the GPT-2 WebText dataset

The dataset is **not publicly released**. The model card notes that internet-sourced data skews toward developed nations, English-language content, and younger/male demographics (`model-card.md:56`).

## 4. Architecture

### 4.1 Dual-Encoder Design

CLIP uses two independent encoders that project into a shared embedding space:

```
                    ┌─────────────────┐
  Image ──────────► │  Image Encoder   │ ──► image_features (d-dim)
                    └─────────────────┘              │
                                                     │ cosine
                                                     │ similarity
                    ┌─────────────────┐              │
  Text  ──────────► │  Text Encoder    │ ──► text_features (d-dim)
                    └─────────────────┘
```

The CLIP class (`clip/model.py:243-372`) orchestrates both encoders. The constructor dispatches to either a ResNet or ViT image encoder based on the `vision_layers` type:

```python
# clip/model.py:262-280
if isinstance(vision_layers, (tuple, list)):
    self.visual = ModifiedResNet(...)   # ResNet variant
else:
    self.visual = VisionTransformer(...)  # ViT variant
```

### 4.2 Image Encoder: Vision Transformer

The Vision Transformer (ViT) (`clip/model.py:206-240`) follows the architecture from Dosovitskiy et al. (2020) with the standard CLIP modifications:

1. **Patch embedding:** A single `Conv2d` with `kernel_size=patch_size` and `stride=patch_size` splits the image into non-overlapping patches and linearly embeds them:
   ```python
   # clip/model.py:211
   self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                           kernel_size=patch_size, stride=patch_size, bias=False)
   ```
   For ViT-B/32 with 224x224 input: produces a 7x7 = 49-token sequence.
   For ViT-B/16 with 224x224 input: produces a 14x14 = 196-token sequence.

2. **Class token:** A learnable `[CLS]` embedding is prepended to the patch sequence (`clip/model.py:214`). The final representation is taken from this token.

3. **Positional embeddings:** Absolute learned positional embeddings are added (`clip/model.py:215`). Shape: `(num_patches + 1, width)`.

4. **Transformer blocks:** Standard pre-norm transformer layers with `ResidualAttentionBlock` (`clip/model.py:171-192`), each containing:
   - LayerNorm → Multi-Head Self-Attention → Residual
   - LayerNorm → MLP (4x expansion with QuickGELU) → Residual

5. **Output projection:** After the final LayerNorm, the `[CLS]` token is linearly projected to `embed_dim`:
   ```python
   # clip/model.py:235-238
   x = self.ln_post(x[:, 0, :])
   if self.proj is not None:
       x = x @ self.proj
   ```

### 4.3 Image Encoder: Modified ResNet

The Modified ResNet (`clip/model.py:94-154`) departs from the standard torchvision ResNet in three ways documented in the class docstring:

1. **3-layer stem** instead of a single 7x7 convolution:
   ```python
   # clip/model.py:108-117
   self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, ...)
   self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, ...)
   self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, ...)
   self.avgpool = nn.AvgPool2d(2)
   ```

2. **Anti-aliased strided convolutions:** All downsampling uses `AvgPool2d` before stride-1 convolution instead of stride-2 convolutions directly (`clip/model.py:25`, `clip/model.py:37`). This follows the insight from Zhang (2019) that pooling before striding improves shift-equivariance.

3. **Attention pooling** instead of global average pooling. The `AttentionPool2d` module (`clip/model.py:58-91`) uses multi-head QKV attention where:
   - The spatial feature map is flattened and a mean-pooled token is prepended
   - Positional embeddings are added
   - A single attention layer attends from the mean token (query) to all spatial tokens (keys/values)
   - Output is projected to `embed_dim` via `c_proj`

### 4.4 Text Encoder

The text encoder is a standard Transformer (`clip/model.py:195-203`, `clip/model.py:282-294`):

- **Token embedding:** `nn.Embedding(vocab_size, transformer_width)` — maps token IDs to vectors (`clip/model.py:290`)
- **Positional embedding:** Learned absolute positional embeddings for context_length=77 positions (`clip/model.py:291`)
- **Causal attention mask:** The text transformer uses autoregressive (left-to-right) masking, built via `build_attention_mask()` (`clip/model.py:328-334`):
  ```python
  mask = torch.empty(self.context_length, self.context_length)
  mask.fill_(float("-inf"))
  mask.triu_(1)  # upper triangular = masked
  ```
- **Feature extraction:** The representation is taken from the **EOT (end-of-text) token** position — specifically, the position of the highest token ID in each sequence:
  ```python
  # clip/model.py:354
  x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
  ```
  This is analogous to BERT's `[CLS]` token but at the end of the sequence, which is natural for autoregressive models.

- **Output projection:** A learned linear projection (`text_projection`) maps from `transformer_width` to the shared `embed_dim`.

### 4.5 Shared Embedding Space

Both encoders project into the same `embed_dim`-dimensional space. Features are L2-normalized before comparison:

```python
# clip/model.py:363-364
image_features = image_features / image_features.norm(dim=1, keepdim=True)
text_features = text_features / text_features.norm(dim=1, keepdim=True)
```

This means all representations live on the unit hypersphere, and their dot product equals cosine similarity.

## 5. Training Objective: Contrastive Learning

### 5.1 The Loss Function

Given a batch of N (image, text) pairs, CLIP computes an N x N matrix of cosine similarities between all possible image-text combinations. The training objective is a **symmetric cross-entropy loss**:

$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}\right)$$

Where:

$$\mathcal{L}_{\text{image}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\tau \cdot \cos(I_i, T_i))}{\sum_{j=1}^{N}\exp(\tau \cdot \cos(I_i, T_j))}$$

$$\mathcal{L}_{\text{text}} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\tau \cdot \cos(I_i, T_i))}{\sum_{j=1}^{N}\exp(\tau \cdot \cos(I_j, T_i))}$$

- $I_i$ and $T_i$ are the normalized image and text embeddings for the $i$-th pair
- $\tau$ is the learned temperature (logit scale)
- The correct pairs are on the diagonal (i=j); all off-diagonal pairs are negatives

Intuitively: for each image, the model tries to identify its matching text from N candidates (and vice versa).

### 5.2 Temperature Parameter

The temperature $\tau$ is a **learned scalar** initialized to $\exp(\ln(1/0.07)) \approx 14.3$:

```python
# clip/model.py:295
self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
```

At inference time it is exponentiated and applied as a multiplicative scale:

```python
# clip/model.py:367-368
logit_scale = self.logit_scale.exp()
logits_per_image = logit_scale * image_features @ text_features.t()
```

Higher temperature → sharper softmax distribution → model is more confident in its predictions. The paper found that learning this parameter (rather than fixing it) was important for training stability and final performance.

### 5.3 Pseudocode

From the paper (Figure 3):

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned image-text pairs
# T[n, l]       - corresponding tokenized text
# W_i[d_i, d_e] - learned image projection
# W_t[d_t, d_e] - learned text projection
# t             - learned temperature parameter

# extract feature representations of each modality
I_f = image_encoder(I)  # [n, d_i]
T_f = text_encoder(T)   # [n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(I_f @ W_i, axis=1)
T_e = l2_normalize(T_f @ W_t, axis=1)

# scaled pairwise cosine similarities [n, n]
logits = I_e @ T_e.T * exp(t)

# symmetric loss function
labels = arange(n)  # [0, 1, 2, ..., n-1]
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

## 6. Zero-Shot Transfer

### 6.1 How It Works

After pre-training, CLIP can classify images into **arbitrary categories** without any additional training:

1. **Define classes as text:** Construct a text prompt for each class (e.g., `"a photo of a dog"`, `"a photo of a cat"`).
2. **Encode everything:** Pass the image through the image encoder and all text prompts through the text encoder.
3. **Compare:** Compute cosine similarity between the image embedding and each text embedding.
4. **Predict:** The class whose text embedding is most similar to the image embedding is the prediction.

```python
# From README.md — zero-shot prediction
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)
```

This is fundamentally different from standard classifiers: the label space is defined **at inference time** through natural language, not at training time through a fixed output layer.

### 6.2 Prompt Engineering

The paper found that the choice of text prompt significantly affects zero-shot performance. A bare class name like `"dog"` is ambiguous (it could be a word, a concept, a proper noun), while `"a photo of a dog"` provides contextual grounding.

The paper explored numerous prompt templates. For ImageNet alone, 80 different templates were tested (see `notebooks/Prompt_Engineering_for_ImageNet.ipynb`). The `data/prompts.md` file contains the class names and prompt templates used for 26 of the 27 datasets evaluated in the paper.

Examples of dataset-specific prompts:
- **Generic:** `"a photo of a {class}."`
- **Fine-grained:** `"a photo of a {class}, a type of bird."` (for Birdsnap)
- **Scene:** `"a photo of a {class}."` (for SUN397 scene recognition)
- **Satellite:** `"a satellite photo of {class}."` (for EuroSAT)
- **Action:** `"a photo of a person doing {class}."` (for UCF101)

### 6.3 Prompt Ensembling

Rather than using a single prompt template, CLIP ensembles predictions across multiple templates for each class. The procedure:

1. For each class, generate text from K different templates.
2. Encode all K prompts and **average the resulting embeddings** (then re-normalize).
3. Use the averaged embedding as the class representation.

This improves performance by ~3.5% on ImageNet and helps smooth out sensitivity to prompt wording.

## 7. Key Results

### 7.1 Zero-Shot Performance

Headline result: **Zero-shot CLIP (ViT-L/14@336px) matches the accuracy of a fully supervised ResNet-50 trained on ImageNet** (~76.2% top-1) — without seeing any ImageNet training examples.

Performance varies dramatically by dataset:
- **Strong (>90% zero-shot):** STL-10, CIFAR-10, Food101, Oxford Pets
- **Competitive (70–90%):** ImageNet, CIFAR-100, Caltech-101, Birdsnap
- **Weak (<50%):** MNIST (~59%), KITTI Distance, CLEVR Counting, Hateful Memes, satellite imagery

The pattern: CLIP excels on tasks that are well-represented in internet image-text data and struggles on tasks requiring specialized knowledge (medical imagery, counting, distance estimation) or domains far from natural photographs (rendered text, synthetic images).

### 7.2 Linear Probe Performance

Using CLIP features as a frozen backbone with a simple logistic regression classifier on top:

- CLIP ViT-L/14 achieves **~85.4% on ImageNet** with linear probe — competitive with heavily supervised models.
- On 12 of 27 evaluated datasets, CLIP's linear probe outperforms the best publicly available ImageNet-pretrained model's linear probe.
- Linear probes require labeled data but no fine-tuning of the backbone.

This is demonstrated in the repo's README linear-probe example using scikit-learn:
```python
# From README.md
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000)
classifier.fit(train_features, train_labels)
```

### 7.3 Distribution Shift Robustness

One of the paper's most significant findings: CLIP is substantially more robust to distribution shift than supervised ImageNet models.

On ImageNet variant benchmarks:
- **ImageNet-V2:** Minimal degradation
- **ImageNet-R** (renditions): CLIP retains much more accuracy
- **ImageNet-Sketch:** CLIP outperforms supervised models by a large margin
- **ImageNet-A** (adversarial): Significant improvement over standard models
- **ObjectNet:** Better generalization to novel viewpoints/backgrounds

The paper argues this robustness comes from natural language supervision: instead of memorizing dataset-specific visual patterns, CLIP learns more generalizable visual-semantic associations.

### 7.4 Scaling Behavior

CLIP's performance scales smoothly and predictably with compute:
- Larger models consistently improve zero-shot transfer across benchmarks.
- The scaling follows approximately log-linear trends in compute.
- ViT architectures are more compute-efficient than ResNets at the same performance level.
- The authors estimated that to match state-of-the-art ImageNet accuracy (~88%) in zero-shot would require ~1000x more compute than the largest model trained.

## 8. Model Variants

Nine pre-trained checkpoints are available in this repo (`clip/clip.py:30-40`):

| Model | Image Encoder | embed_dim | Image Resolution | Approx. Params |
|-------|--------------|-----------|-----------------|---------------|
| `RN50` | ResNet-50, layers=(3,4,6,3) | 1024 | 224 | ~38M |
| `RN101` | ResNet-101, layers=(3,4,23,3) | 512 | 224 | ~56M |
| `RN50x4` | ResNet-50 4x width | 640 | 288 | ~87M |
| `RN50x16` | ResNet-50 16x width | 768 | 384 | ~167M |
| `RN50x64` | ResNet-50 64x width | 1024 | 448 | ~420M |
| `ViT-B/32` | ViT-Base, patch=32 | 512 | 224 | ~151M |
| `ViT-B/16` | ViT-Base, patch=16 | 512 | 224 | ~150M |
| `ViT-L/14` | ViT-Large, patch=14 | 768 | 224 | ~428M |
| `ViT-L/14@336px` | ViT-Large, patch=14 | 768 | 336 | ~428M |

The architecture is automatically inferred from the checkpoint via `build_model()` (`clip/model.py:399-436`), which inspects key shapes in the state dict:

```python
# clip/model.py:400-407
vit = "visual.proj" in state_dict  # ViT has this key; ResNet does not
if vit:
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys()
                         if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
```

The ResNet models are scaled following the EfficientNet compound scaling rule (Tan & Le, 2019) — increasing width, depth, and resolution jointly. The `@336px` variant uses the same ViT-L/14 weights fine-tuned at higher resolution.

## 9. Tokenization and Preprocessing

### Text Tokenization

CLIP uses a **Byte Pair Encoding (BPE)** tokenizer (`clip/simple_tokenizer.py:62-132`) derived from the GPT-2 tokenizer:

- **Vocabulary size:** ~49,152 tokens (256 byte tokens + 256 end-of-word tokens + 48,894 BPE merges + 2 special tokens)
- **Special tokens:** `<|startoftext|>` (SOT) and `<|endoftext|>` (EOT)
- **Context length:** 77 tokens (including SOT and EOT)
- **Text cleaning:** `ftfy.fix_text()` → HTML unescape → lowercase → whitespace normalization
- **Tokenization regex** (`clip/simple_tokenizer.py:78`):
  ```python
  re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|
                  [\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
  ```

The `clip.tokenize()` function (`clip/clip.py:205-245`) wraps each text with SOT/EOT tokens and pads to `context_length`:

```python
# clip/clip.py:228-230
sot_token = _tokenizer.encoder["<|startoftext|>"]
eot_token = _tokenizer.encoder["<|endoftext|>"]
all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
```

### Image Preprocessing

The image transform pipeline (`clip/clip.py:79-86`) applied at inference:

```python
Compose([
    Resize(n_px, interpolation=BICUBIC),
    CenterCrop(n_px),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])
```

The normalization constants are dataset-specific (not ImageNet's standard values), reflecting the WIT training distribution. Resolution `n_px` varies by model (224, 288, 336, 384, or 448).

## 10. Parameter Initialization

The `initialize_parameters()` method (`clip/model.py:299-326`) uses careful initialization critical for training stability at scale:

| Component | Distribution | Standard Deviation |
|-----------|-------------|-------------------|
| Token embedding | Normal | 0.02 |
| Positional embedding | Normal | 0.01 |
| Attention in_proj | Normal | $d_{model}^{-0.5}$ |
| Attention out_proj | Normal | $(d_{model} \cdot 2L)^{-0.5}$ where L = num layers |
| MLP fc (in) | Normal | $(2 \cdot d_{model})^{-0.5}$ |
| MLP fc (out) | Normal | $(d_{model} \cdot 2L)^{-0.5}$ |
| Text projection | Normal | $d_{model}^{-0.5}$ |
| ResNet bn3.weight | Zeros | — (residual branches start as identity) |

The output projection scaling by $\frac{1}{\sqrt{d \cdot 2L}}$ is a depth-dependent initialization that prevents activations from growing with depth — similar to GPT-2's initialization scheme.

## 11. Limitations

The paper and model card identify several important limitations:

### Performance Gaps
- **Fine-grained classification:** CLIP struggles to distinguish between visually similar subcategories (e.g., car models, bird species, flower types) without specialized prompts.
- **Abstract/systematic tasks:** Counting objects, estimating distances, and other tasks requiring spatial reasoning are weak.
- **Out-of-distribution domains:** Satellite imagery, medical imaging, and other specialized domains far from internet photo distributions perform poorly.
- **Novel/unseen tasks:** CLIP's zero-shot accuracy on MNIST is only ~59%, suggesting that truly out-of-distribution visual concepts remain challenging.

### Data Biases
- Training data is skewed toward English-language internet content from developed nations.
- The model reflects and can amplify societal biases present in internet data.
- Testing revealed disparities in classification accuracy across demographic groups (see `model-card.md:110-113`).

### Architectural Constraints
- **Context length of 77 tokens** limits the complexity of text descriptions.
- **No fine-grained spatial understanding** — the model produces a single global embedding per image.
- **Computationally expensive** — the largest models require significant GPU resources.

### Broader Concerns
- Can be used for surveillance or facial recognition tasks that raise ethical concerns.
- Zero-shot flexibility means the model can be repurposed for uses not anticipated by its creators.
- The model card explicitly states that "any deployed use case — whether commercial or not — is currently out of scope" (`model-card.md:46`).

## 12. Impact and Significance

CLIP represented a paradigm shift in computer vision with lasting influence:

1. **Foundation models for vision:** Demonstrated that pre-training on broad, weakly-supervised data creates transferable representations — catalyzing the "foundation model" movement in vision (alongside concurrent work like ALIGN).

2. **Language as an interface to vision:** Showed that natural language can replace fixed label taxonomies, enabling open-vocabulary recognition. This insight directly influenced subsequent work in open-vocabulary detection, segmentation, and generation.

3. **Multimodal embedding spaces:** The shared image-text representation became a building block for:
   - **Image generation:** DALL-E 2 uses CLIP embeddings as a conditioning signal.
   - **Image search:** Encode a text query, find nearest images in CLIP space.
   - **Video understanding:** Extend to frame-level image-text matching.
   - **Robotics:** Language-conditioned manipulation using CLIP features.

4. **Robustness through diversity:** Showed that learning from diverse natural language supervision produces more robust representations than learning from curated, narrow datasets.

5. **Scaling laws for multimodal learning:** Provided evidence that multimodal contrastive learning follows predictable scaling trends, motivating larger-scale follow-up work (OpenCLIP, SigLIP, EVA-CLIP).

## 13. Code-to-Paper Mapping

Quick reference mapping paper concepts to this repository's implementation:

| Paper Concept | Code Location | Key Detail |
|--------------|---------------|------------|
| Dual-encoder architecture | `clip/model.py:243-372` | `CLIP` class |
| Image encoder (ViT) | `clip/model.py:206-240` | `VisionTransformer` class |
| Image encoder (ResNet) | `clip/model.py:94-154` | `ModifiedResNet` class |
| Attention pooling (ResNet) | `clip/model.py:58-91` | `AttentionPool2d` class |
| Text encoder | `clip/model.py:195-203, 282-294` | `Transformer` + embeddings in `CLIP.__init__` |
| Transformer block | `clip/model.py:171-192` | `ResidualAttentionBlock` |
| QuickGELU activation | `clip/model.py:166-168` | `x * sigmoid(1.702 * x)` |
| Causal attention mask | `clip/model.py:328-334` | `build_attention_mask()` |
| Contrastive logits | `clip/model.py:358-372` | `CLIP.forward()` |
| Learned temperature | `clip/model.py:295` | `logit_scale = log(1/0.07)` |
| Feature normalization | `clip/model.py:363-364` | L2 norm before dot product |
| EOT token extraction | `clip/model.py:354` | `text.argmax(dim=-1)` indexes EOT position |
| BPE tokenizer | `clip/simple_tokenizer.py:62-132` | `SimpleTokenizer` class |
| Tokenize API | `clip/clip.py:205-245` | `clip.tokenize()` |
| Image preprocessing | `clip/clip.py:79-86` | `_transform()` — Resize, CenterCrop, Normalize |
| Model loading | `clip/clip.py:94-202` | `clip.load()` |
| Architecture inference | `clip/model.py:399-436` | `build_model()` — infers config from state_dict |
| FP16 conversion | `clip/model.py:375-396` | `convert_weights()` |
| FP16-safe LayerNorm | `clip/model.py:157-163` | Casts to float32 internally |
| Weight initialization | `clip/model.py:299-326` | `initialize_parameters()` |
| Prompt templates | `data/prompts.md` | Templates for 26 datasets |
| Prompt engineering | `notebooks/Prompt_Engineering_for_ImageNet.ipynb` | 80 ImageNet templates |
| Zero-shot prediction | `README.md:89-137` | Full CIFAR-100 example |
| Linear probe evaluation | `README.md:141-191` | scikit-learn LogisticRegression example |
| Model checkpoints | `clip/clip.py:30-40` | `_MODELS` dict with download URLs |
| JIT/non-JIT loading | `clip/clip.py:126-142` | Tries JIT first, falls back to state_dict |
| Consistency tests | `tests/test_consistency.py` | Verifies JIT and Python model parity |

---

*This summary is based on the paper [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) and the source code in this repository (OpenAI CLIP). For the original blog post, see [openai.com/blog/clip](https://openai.com/blog/clip/).*
