# Source provenance

Downloaded 2026-11-06 from the official DeepSeek Hugging Face repository:

- Repository: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- Pinned revision: `314113f8a7dadb388215396ba5fbee2c610a04d6`
- `model.py`: upstream `inference/model.py` (read-only reference, not imported at runtime).
- `engram.py`: upstream `inference/engram.py` (read-only reference).
- `config.json`: upstream `inference/config.json`, full-scale dimensions for comparison.
- `LICENSE`: upstream MIT license, retained for adaptations.
- Supplied technical report: `DeepSeek_V41_Tech_Report.pdf`, SHA-256 `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.

The trainable implementation uses ordinary PyTorch instead of TileLang/FP8/FP4 GEMM kernels. It follows the reference's low-rank Q/output, partial RoPE, shared latent K=V, attention sink, weighted compression, sqrt-softplus expert routing and delayed mHC pre-mix. It adds request-owned caches, differentiable training, stable causal tie-breaking, an indexer training objective and the report's explicit encoder-derived decoder global KV. The upstream minimal forward runs all layers; CED bounded replay is a separately tested report-based adaptation here.

Engram hashing is a portable modular multi-head variant, not byte-compatible with released Engram tables. Token normalization follows the reference's Unicode/space/lowercase policy. No released model weights or tokenizer files are downloaded or included.
