# Repository Map

A detailed reference for navigating the OpenAI CLIP codebase.

---

## Directory Tree

```
CLIP/
├── clip/                                  # Core Python package
│   ├── __init__.py                        #   Package entry — re-exports clip.py's public API
│   ├── clip.py                (245 lines) #   High-level API: load(), tokenize(), available_models()
│   ├── model.py               (436 lines) #   All model architectures and weight utilities
│   ├── simple_tokenizer.py    (132 lines) #   BPE tokenizer (GPT-2 derived)
│   └── bpe_simple_vocab_16e6.txt.gz       #   Pre-computed BPE vocabulary (~1.3 MB compressed)
│
├── notebooks/                             # Jupyter notebooks (Colab-ready)
│   ├── Interacting_with_CLIP.ipynb        #   Walkthrough: load model, encode, similarity matrix, zero-shot
│   └── Prompt_Engineering_for_ImageNet.ipynb  # 1000 ImageNet classes, 80 prompt templates, ensembling
│
├── data/                                  # Dataset documentation and download instructions
│   ├── prompts.md                         #   Prompt templates + class names for 26 evaluation datasets
│   ├── country211.md                      #   Country211 geolocation dataset (YFCC100M subset)
│   ├── rendered-sst2.md                   #   Rendered SST2 OCR dataset
│   └── yfcc100m.md                        #   YFCC100M 15% English-filtered subset (14.8M images)
│
├── tests/                                 # Test suite
│   └── test_consistency.py    (25 lines)  #   Verifies JIT and Python model outputs match
│
├── docs/                                  # Documentation (added locally)
│   ├── summary.md                         #   Deep-dive paper summary
│   └── repo_map.md                        #   This file
│
├── setup.py                   (20 lines)  # Package installer (pip install -e .)
├── requirements.txt                       # Runtime dependencies
├── hubconf.py                 (41 lines)  # PyTorch Hub integration entry points
├── model-card.md                          # Official model card (intended use, biases, limitations)
├── CLIP.png                               # Architecture diagram from the paper
├── LICENSE                                # MIT License
└── README.md                              # Project overview, usage examples, API docs
```

---

## Module Details

### `clip/__init__.py`

```python
from .clip import *
```

Single-line package initializer. Makes `clip.load()`, `clip.tokenize()`, and `clip.available_models()` available as top-level imports when you write `import clip`.

---

### `clip/clip.py` — High-Level API

The user-facing interface. All external interaction with CLIP goes through this module.

**Constants:**

| Name | Line | Description |
|------|------|-------------|
| `_MODELS` | 30–40 | Dict mapping model names (e.g. `"ViT-B/32"`) to Azure CDN download URLs. SHA256 checksums are embedded in the URL path. |
| `_tokenizer` | 28 | Module-level singleton `SimpleTokenizer` instance, shared across all `tokenize()` calls. |

**Functions:**

| Function | Lines | Signature | Description |
|----------|-------|-----------|-------------|
| `_download` | 43–72 | `(url, root) -> str` | Downloads a model checkpoint with progress bar. Verifies SHA256 checksum. Returns local file path. Skips download if cached file passes hash check. |
| `_convert_image_to_rgb` | 75–76 | `(image) -> Image` | Ensures PIL image is RGB (handles grayscale/RGBA inputs). |
| `_transform` | 79–86 | `(n_px) -> Compose` | Builds the image preprocessing pipeline: Resize (bicubic) -> CenterCrop -> RGB -> ToTensor -> Normalize. Normalization uses WIT dataset statistics, not ImageNet. |
| `available_models` | 89–91 | `() -> List[str]` | Returns the names of all downloadable model variants. |
| `load` | 94–202 | `(name, device, jit, download_root) -> (model, preprocess)` | Main entry point. Downloads checkpoint if needed, builds model, handles JIT/non-JIT loading, patches device and dtype for CPU/CUDA compatibility. Returns the model and its preprocessing transform. |
| `tokenize` | 205–245 | `(texts, context_length, truncate) -> Tensor` | Tokenizes strings into padded integer tensors of shape `[N, 77]`. Wraps each text with `<\|startoftext\|>` and `<\|endoftext\|>` tokens. Raises `RuntimeError` if any input exceeds context length (unless `truncate=True`). |

---

### `clip/model.py` — Model Architecture

Contains all neural network definitions and weight utilities. No training code — this is inference-only.

**Classes:**

| Class | Lines | Description |
|-------|-------|-------------|
| `Bottleneck` | 10–55 | ResNet bottleneck block (expansion=4). Anti-aliased: uses `AvgPool2d` before strided convolution instead of strided conv directly. Downsample path also uses `AvgPool2d` + 1x1 conv. |
| `AttentionPool2d` | 58–91 | Replaces global average pooling in ResNet. Flattens spatial features, prepends a mean-pooled token, adds positional embeddings, then runs a single multi-head attention layer where the mean token is the query and all spatial tokens are keys/values. Outputs via `c_proj`. |
| `ModifiedResNet` | 94–154 | The CNN image encoder. 3-layer stem (3x3 convs + avgpool) instead of standard 7x7 conv + maxpool. Four residual stages (`layer1`–`layer4`). Final output via `AttentionPool2d`. |
| `LayerNorm` | 157–163 | Subclass of `nn.LayerNorm` that casts input to float32 before normalization, then casts back. Prevents numerical issues in fp16 training/inference. |
| `QuickGELU` | 166–168 | Approximation of GELU: `x * sigmoid(1.702 * x)`. Faster than exact GELU; used throughout all transformer blocks. |
| `ResidualAttentionBlock` | 171–192 | Single transformer layer. Pre-norm architecture: `LN -> MHSA -> residual`, then `LN -> MLP(4x expansion, QuickGELU) -> residual`. Accepts optional `attn_mask` for causal masking. |
| `Transformer` | 195–203 | Stack of `ResidualAttentionBlock` layers. A generic container — used by both the text encoder and the ViT image encoder. |
| `VisionTransformer` | 206–240 | The ViT image encoder. Patch embedding via `Conv2d` (kernel=stride=patch_size). Prepends learnable `[CLS]` token, adds positional embeddings, processes through `Transformer`, extracts `[CLS]` output, projects to `embed_dim` via `self.proj`. |
| `CLIP` | 243–372 | The full dual-encoder model. Constructs the image encoder (ResNet or ViT based on `vision_layers` type), text encoder (Transformer + embeddings), and computes contrastive logits. Contains `encode_image()`, `encode_text()`, and `forward()`. |

**Standalone Functions:**

| Function | Lines | Description |
|----------|-------|-------------|
| `convert_weights` | 375–396 | Converts all `Conv1d`, `Conv2d`, `Linear`, `MultiheadAttention` parameters and projection matrices to fp16. Applied during `build_model()`. |
| `build_model` | 399–436 | Infers full model architecture from a state_dict by inspecting key names and tensor shapes. Constructs a `CLIP` instance, loads weights, sets to eval mode. This is the core deserialization logic. |

---

### `clip/simple_tokenizer.py` — BPE Tokenizer

A self-contained byte-level BPE tokenizer adapted from GPT-2.

**Functions:**

| Function | Lines | Description |
|----------|-------|-------------|
| `default_bpe` | 10–12 | Returns the path to the bundled vocabulary file `bpe_simple_vocab_16e6.txt.gz`. LRU-cached. |
| `bytes_to_unicode` | 15–35 | Builds a reversible mapping from byte values (0–255) to Unicode characters. Avoids mapping to whitespace/control characters that break BPE. LRU-cached. |
| `get_pairs` | 38–47 | Returns all adjacent symbol pairs in a word tuple. Used during BPE merging. |
| `basic_clean` | 50–53 | Text cleanup: `ftfy.fix_text()` -> double HTML unescape -> strip. |
| `whitespace_clean` | 56–59 | Collapse all whitespace runs to single spaces. |

**Class: `SimpleTokenizer`**

| Method | Lines | Description |
|--------|-------|-------------|
| `__init__` | 63–78 | Loads the BPE vocabulary (48,894 merges), builds encoder/decoder dicts (~49K tokens), compiles the tokenization regex. Special tokens: `<\|startoftext\|>`, `<\|endoftext\|>`. |
| `bpe` | 80–119 | Applies BPE merges to a single token string. Iteratively merges the highest-priority bigram pair until no more merges apply. Results are cached. |
| `encode` | 121–127 | Full encoding pipeline: clean text -> lowercase -> regex split -> byte encode -> BPE merge -> token IDs. Returns `List[int]`. |
| `decode` | 129–132 | Reverses encoding: token IDs -> BPE tokens -> bytes -> UTF-8 string. End-of-word markers (`</w>`) become spaces. |

---

### `hubconf.py` — PyTorch Hub Entry Points

Allows loading CLIP models via `torch.hub.load('openai/CLIP', 'ViT_B_32')`.

| Component | Lines | Description |
|-----------|-------|-------------|
| `model_functions` | 8 | Maps model names to Hub-safe function names (replaces punctuation with underscores: `"ViT-B/32"` -> `"ViT_B_32"`). |
| `_create_hub_entrypoint` | 10–35 | Factory that returns a closure calling `clip.load(model_name, **kwargs)` for each model variant. |
| `globals().update(...)` | 42 | Dynamically injects entry point functions into the module namespace. |

---

### `tests/test_consistency.py`

A single parameterized pytest that verifies JIT-compiled and standard Python models produce equivalent outputs.

| Test | Lines | Description |
|------|-------|-------------|
| `test_consistency` | 9–25 | For each model in `clip.available_models()`: loads both JIT and non-JIT versions, runs forward pass on `CLIP.png` with 3 text inputs, asserts softmax probabilities match within `atol=0.01, rtol=0.1`. |

Run with: `pytest tests/ -v`

---

### `setup.py`

Standard setuptools configuration. Installs the `clip` package (version 1.0). Reads dependencies from `requirements.txt`. Optional dev dependency: `pytest`.

Install locally: `pip install -e .` or `pip install -e ".[dev]"` (includes pytest).

---

### `requirements.txt`

```
ftfy          # Unicode text fixing (mojibake repair)
packaging     # Version parsing (used for torch version checks)
regex         # Advanced regex engine (Unicode properties \p{L}, \p{N})
tqdm          # Progress bars (model download, data loading)
torch         # PyTorch framework
torchvision   # Image transforms (Resize, CenterCrop, Normalize, ToTensor)
```

---

## Call Flow Diagrams

### Model Loading

The path from `clip.load("ViT-B/32")` to a ready-to-use model:

```
clip.load("ViT-B/32", device)                    clip/clip.py:94
│
├── name in _MODELS? ──yes──► _download(url, root)    clip/clip.py:119-120
│                               │
│                               ├── File cached + SHA256 matches? ──► return path
│                               └── Download from Azure CDN ──► verify hash ──► return path
│
├── torch.jit.load(file)                          clip/clip.py:129
│   └── Success? ──► JIT path (patch device/dtype)
│   └── Failure? ──► state_dict path (below)
│
├── build_model(state_dict)                       clip/model.py:399
│   │
│   ├── "visual.proj" in state_dict?
│   │   ├── yes ──► infer ViT config from tensor shapes
│   │   └── no  ──► infer ResNet config from layer counts
│   │
│   ├── Infer text encoder config from state_dict keys
│   │
│   ├── CLIP(embed_dim, vision_*, text_*)         clip/model.py:243
│   │   ├── isinstance(vision_layers, tuple)?
│   │   │   ├── yes ──► ModifiedResNet(...)       clip/model.py:264
│   │   │   └── no  ──► VisionTransformer(...)    clip/model.py:273
│   │   ├── Transformer(... attn_mask=causal)     clip/model.py:282
│   │   ├── nn.Embedding(vocab_size, width)       clip/model.py:290
│   │   ├── positional_embedding, ln_final        clip/model.py:291-292
│   │   ├── text_projection, logit_scale          clip/model.py:294-295
│   │   └── initialize_parameters()               clip/model.py:299
│   │
│   ├── convert_weights(model)  # -> fp16         clip/model.py:434
│   ├── model.load_state_dict(state_dict)         clip/model.py:435
│   └── return model.eval()                       clip/model.py:436
│
├── model.to(device)                              clip/clip.py:139
├── model.float() if CPU                          clip/clip.py:141
│
└── return (model, _transform(resolution))        clip/clip.py:142
```

### Inference: Forward Pass

The data flow through `model(image, text)`:

```
model.forward(image, text)                        clip/model.py:358
│
├──► encode_image(image)                          clip/model.py:340
│    │
│    ├── [If VisionTransformer]                   clip/model.py:223
│    │   image ──► Conv2d (patchify)
│    │         ──► reshape + permute to [N, num_patches, width]
│    │         ──► prepend [CLS] token
│    │         ──► + positional_embedding
│    │         ──► ln_pre
│    │         ──► Transformer (L blocks of ResidualAttentionBlock)
│    │              └── each block: LN -> MHSA -> residual -> LN -> MLP -> residual
│    │         ──► extract [CLS] token: x[:, 0, :]
│    │         ──► ln_post
│    │         ──► @ self.proj  (linear projection to embed_dim)
│    │         ──► image_features
│    │
│    └── [If ModifiedResNet]                      clip/model.py:138
│        image ──► 3-layer stem (conv1->conv2->conv3->avgpool)
│              ──► layer1 -> layer2 -> layer3 -> layer4
│              ──► AttentionPool2d
│                   └── flatten spatial -> prepend mean token
│                       -> + positional_embedding
│                       -> QKV attention (query=mean, key/value=all)
│                       -> c_proj
│              ──► image_features
│
├──► encode_text(text)                            clip/model.py:342
│    text token IDs ──► token_embedding            [N, 77, d_model]
│                   ──► + positional_embedding
│                   ──► permute to [77, N, d_model] (sequence-first)
│                   ──► Transformer (with causal attn_mask)
│                        └── each block: LN -> Masked MHSA -> res -> LN -> MLP -> res
│                   ──► permute back to [N, 77, d_model]
│                   ──► ln_final
│                   ──► extract EOT token: x[arange(N), text.argmax(dim=-1)]
│                   ──► @ text_projection  (project to embed_dim)
│                   ──► text_features
│
├──► L2-normalize both feature sets               clip/model.py:363-364
│    image_features /= image_features.norm(dim=1, keepdim=True)
│    text_features  /= text_features.norm(dim=1, keepdim=True)
│
├──► logit_scale = self.logit_scale.exp()         clip/model.py:367
│
├──► logits_per_image = scale * (I @ T^t)         clip/model.py:368
│    shape: [N_images, N_texts]
│
└──► logits_per_text = logits_per_image.t()       clip/model.py:369
     shape: [N_texts, N_images]
```

### Tokenization

The path from raw text to padded token tensor:

```
clip.tokenize("a photo of a cat")                 clip/clip.py:205
│
├── Wrap in list if single string                  clip/clip.py:225-226
│
├── For each text:
│   ├── [SOT token] +                             clip/clip.py:228-230
│   │
│   ├── _tokenizer.encode(text)                   simple_tokenizer.py:121
│   │   ├── basic_clean(text)                      ftfy.fix_text -> html.unescape -> strip
│   │   ├── whitespace_clean(text)                 collapse whitespace
│   │   ├── .lower()
│   │   ├── regex split into word chunks           pattern handles contractions, Unicode
│   │   ├── for each chunk:
│   │   │   ├── UTF-8 encode -> byte_encoder map
│   │   │   └── bpe() merge loop                   simple_tokenizer.py:80
│   │   │       └── repeatedly merge highest-priority bigram
│   │   └── return list of token IDs
│   │
│   └── + [EOT token]                             clip/clip.py:230
│
├── Pad/truncate to context_length (77)            clip/clip.py:236-243
│   └── If len > 77 and truncate=False: raise RuntimeError
│
└── return LongTensor [N, 77]                      clip/clip.py:245
```

### Zero-Shot Classification

The pattern for using CLIP as a zero-shot classifier (from README / notebooks):

```
# 1. Build class embeddings (once per task)
for each class_name in classes:
    text = f"a photo of a {class_name}"    ──► clip.tokenize()
                                           ──► model.encode_text()
                                           ──► L2 normalize
                                           ──► append to weight matrix

# 2. Classify images
image ──► preprocess ──► model.encode_image() ──► L2 normalize

# 3. Compute similarities
similarity = (100.0 * image_features @ text_features.T)

# 4. Predict
probs = similarity.softmax(dim=-1)
prediction = probs.argmax()
```

With prompt ensembling (as in `Prompt_Engineering_for_ImageNet.ipynb`):

```
for each class_name in classes:
    for each template in 80_templates:
        text = template.format(class_name)
        ──► encode ──► normalize
    class_embedding = mean(all_template_embeddings)
    class_embedding /= class_embedding.norm()     # re-normalize after averaging
```

---

## Data Files Reference

### `data/prompts.md`

Contains prompt templates and class name lists for **26 of 27 evaluation datasets** from the paper (Table 9). Each dataset entry has:
- `classes`: Python list of class name strings
- `templates`: Python list of template strings with `{}` as the class name placeholder

The 27th dataset (ImageNet) has its prompts in `notebooks/Prompt_Engineering_for_ImageNet.ipynb` instead.

### `data/country211.md`

Download instructions for the **Country211** geolocation dataset: 211 countries, 150 train / 50 val / 100 test images per country, sourced from YFCC100M. ~11 GB download.

### `data/rendered-sst2.md`

Download instructions for **Rendered SST2**: Stanford Sentiment Treebank sentences rendered as images for OCR evaluation. ~131 MB download.

### `data/yfcc100m.md`

Download instructions for the **YFCC100M English subset** used in dataset ablations: 14.8M images (~15% of full YFCC100M), filtered for English-language titles/descriptions. Provided as a TSV of (line number, photo ID, photo hash).

---

## Notebooks Reference

### `notebooks/Interacting_with_CLIP.ipynb`

End-to-end walkthrough covering:
1. Installing CLIP in Colab
2. Loading `ViT-B/32` and inspecting model parameters (151M params, 224px input, 77 context length, 49408 vocab)
3. Image preprocessing pipeline demonstration
4. Text tokenization example
5. Encoding 8 sample images from scikit-image with text descriptions
6. Computing and visualizing the 8x8 cosine similarity matrix
7. Zero-shot classification on CIFAR-100

### `notebooks/Prompt_Engineering_for_ImageNet.ipynb`

Detailed prompt engineering study:
1. 1,000 manually curated ImageNet class names (modified from defaults for better CLIP accuracy — e.g., "nail" -> "metal nail", "kite" -> "kite (bird of prey)")
2. 80 prompt templates covering variations: photos, drawings, sculptures, renderings, video games, origami, close-ups, different qualities, scales, and styles
3. `zeroshot_classifier()` function implementing prompt ensembling (average embeddings across all templates per class, then re-normalize)
4. Evaluation on ImageNet-V2 (ViT-B/32 achieves ~55.9% top-1, ~83.4% top-5)
5. Commentary on template selection via sequential forward selection (best 7 of 80 templates identified)

---

## Key Constants and Magic Numbers

| Value | Location | Meaning |
|-------|----------|---------|
| `77` | `clip/clip.py:205`, `model.py:291` | Max context length (tokens). All text is padded/truncated to this. |
| `49408` | Vocabulary size | 256 byte tokens + 256 end-of-word + 48,894 BPE merges + 2 special tokens. |
| `49406` | Token ID | `<\|startoftext\|>` (SOT) token. |
| `49407` | Token ID | `<\|endoftext\|>` (EOT) token. |
| `1/0.07 ≈ 14.3` | `model.py:295` | Initial logit scale (temperature). Stored as `log(1/0.07)`, exponentiated at runtime. |
| `1.702` | `model.py:168` | QuickGELU coefficient: `x * sigmoid(1.702 * x)`. |
| `4` | `model.py:11`, `model.py:177-180` | Bottleneck expansion factor (ResNet) and MLP expansion factor (Transformer). |
| `64` | `model.py:263, 272, 421` | Head dimension. Attention heads = width // 64. |
| `(0.481, 0.458, 0.408)` | `clip.py:85` | Image normalization mean (WIT dataset). |
| `(0.269, 0.261, 0.276)` | `clip.py:85` | Image normalization std (WIT dataset). |
| `32` | `model.py:126-127, 415` | Spatial downsampling factor of the ResNet stem + 4 stages: `image_resolution // 32 = attnpool spatial dim`. |

---

## Quick Cross-Reference

**"I want to..."**

| Goal | Where to look |
|------|--------------|
| Load a model and run inference | `clip/clip.py:94` (`load`), README examples |
| Understand the ViT architecture | `clip/model.py:206-240` (`VisionTransformer`) |
| Understand the ResNet architecture | `clip/model.py:94-154` (`ModifiedResNet`) |
| Understand the text encoder | `clip/model.py:282-294` (in `CLIP.__init__`), `clip/model.py:342-356` (`encode_text`) |
| See how contrastive logits are computed | `clip/model.py:358-372` (`CLIP.forward`) |
| Modify the tokenizer | `clip/simple_tokenizer.py:62-132` (`SimpleTokenizer`) |
| Add a new model variant | Add entry to `_MODELS` dict in `clip/clip.py:30-40` |
| See prompt templates for a dataset | `data/prompts.md` or `notebooks/Prompt_Engineering_for_ImageNet.ipynb` |
| Run tests | `pytest tests/ -v` |
| Install as editable package | `pip install -e ".[dev]"` |
| Load via PyTorch Hub | `torch.hub.load('openai/CLIP', 'ViT_B_32')` — see `hubconf.py` |
| Understand weight initialization | `clip/model.py:299-326` (`initialize_parameters`) |
| See how architecture is inferred from weights | `clip/model.py:399-436` (`build_model`) |
| Understand fp16 handling | `clip/model.py:157-163` (`LayerNorm`), `clip/model.py:375-396` (`convert_weights`) |
