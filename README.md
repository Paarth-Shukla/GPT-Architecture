# GPT-2 Style LLM from Scratch

A **from-scratch implementation of a GPT-2 style Large Language Model** in Python — no Hugging Face wrappers, no pre-built transformer libraries.  
Every component, from tokenization to multi-head attention to text generation, is built and explained step by step.

---

## What's Covered

| Stage | Details |
|-------|---------|
| **Tokenization** | BPE via `tiktoken` |
| **Token IDs** | Vocabulary construction, encode/decode |
| **Embeddings** | Token embeddings + positional embeddings |
| **Data Loading** | Sliding-window input-target pairs, PyTorch `DataLoader` |
| **Attention** | Simplified → trainable weights → causal masking → multi-head |
| **Transformer Block** | LayerNorm, GELU FFN, residual shortcut connections |
| **Full GPT Model** | End-to-end architecture assembly |
| **Text Generation** | Greedy decoding from output logits |

---

## Notebook Structure

```
LLM_Architecture.ipynb
│
├── Reading in text sample
│
├── Byte Pair Encoding (BPE)
│     └── tiktoken integration (GPT-2 / GPT-3 / GPT-4 vocab sizes)
│
├── Creating Input-Target Pairs
│
├── Implementing a Data Loader
│     └── GPTDatasetV1 + sliding window batching
│
├── Token Embeddings & Positional Embeddings
│
├── Implementing a Simplified Attention Mechanism
│     └── 3D vector visualizations
│
├── Self-Attention with Trainable Weights
│     ├── Why divide by sqrt(d_k)?
│     └── SelfAttention class
│
├── Causal Attention (masking future tokens)
│
├── Multi-Head Attention
│
└── GPT Architecture (6 parts)
      ├── Part 1: Dummy GPT Model
      ├── Part 2: Layer Normalization
      ├── Part 3: Feed-Forward + GELU Activation
      ├── Part 4: Shortcut Connections
      ├── Part 5: Transformer Block
      ├── Part 6: Full GPT Model
      └── Part 7: Text Generation
```

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/Paarth-Shukla/GPT-Architecture.git
cd LLM_Architecture
```

### Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Install dependencies

```bash
pip install torch tiktoken matplotlib numpy jupyter
```

### Launch the notebook

```bash
jupyter notebook LLM_Architecture.ipynb
```

---

## How It Works

1. **BPE** — Switches to `tiktoken` (the same tokenizer used in GPT-2/3/4) for a production-grade vocabulary of 50,257 tokens.
2. **Data Preparation** — A sliding-window `GPTDatasetV1` generates overlapping input-target pairs and feeds them into a PyTorch `DataLoader`.
3. **Embeddings** — Token IDs are mapped to dense vectors via a learned embedding matrix; positional embeddings are added to encode sequence order.
4. **Attention** — Built progressively: simplified dot-product → trainable Q/K/V weights → causal masking → multi-head with concatenated heads.
5. **Transformer Block** — Each block combines multi-head causal attention and a GELU feed-forward network, wrapped with Layer Normalization and residual connections.
6. **Full GPT Model** — Stacks N transformer blocks on top of the embedding layer, followed by a final linear projection to vocabulary logits.
7. **Text Generation** — The model autoregressively samples the next token from logits to generate free-form text.

---

## Key Design Decisions Explained

- **Why causal masking?** Prevents the model from attending to future tokens — essential for autoregressive generation.
- **Why scale by √d_k?** Prevents dot products from growing large in high dimensions, which would push softmax into near-zero gradient regions.
- **Why GELU over ReLU?** GELU provides a smoother, probabilistic activation used in GPT-2 and BERT that empirically outperforms ReLU on language tasks.
- **Why residual connections?** Allow gradients to flow directly through deep stacks, making training of 12+ layer models stable.

---

## Tech Stack

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/) — Tensors, embedding layers, DataLoader
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's BPE tokenizer
- [Matplotlib](https://matplotlib.org/) — Attention and embedding visualizations
- [Jupyter Notebook](https://jupyter.org/)

---

## License

MIT License © 2025 Paarth Shukla
