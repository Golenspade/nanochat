"""Behavioral checks for dense cross-layer KV reuse. No tokenizer or dataset download."""

from dataclasses import asdict
import os

import pytest
import torch

from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig
from nanochat import checkpoint_manager
from nanochat import flash_attention as fa
from nanochat import gpt
from test_engine import ByteTokenizer

COMPUTE_DTYPE = torch.float32


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch, request):
    # Keep CPU checks independent of the host GPU's automatic backend/dtype choice.
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", COMPUTE_DTYPE)
    if request.node.name == "test_compiled_training_matches_eager_gradients":
        yield
    else:
        # Keep ordinary behavior checks independent of a local C++ compiler.
        with torch.compiler.set_stance("force_eager"):
            yield


def make_model(pattern="FR", n_layer=4, n_kv_head=2, window_pattern="L"):
    torch.manual_seed(42)
    config = GPTConfig(sequence_len=128, vocab_size=262, n_layer=n_layer,
                       n_head=4, n_kv_head=n_kv_head, n_embd=64,
                       window_pattern=window_pattern, kv_share_pattern=pattern)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    # Initial zero output projections hide attention mistakes; exercise nonzero paths.
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.04)
            block.mlp.c_proj.weight.normal_(std=0.02)
        model.smear_lambda.fill_(0.3)
    return model


def make_cache(model, batch_size=1, seq_len=64, layout="bhtd"):
    c = model.config
    return KVCache(batch_size, c.n_kv_head, seq_len, c.n_embd // c.n_head,
                   c.n_layer, "cpu", COMPUTE_DTYPE, layout=layout,
                   layer_mapping=c.kv_cache_layer_map())


@pytest.mark.parametrize("pattern,expected", [
    ("F", (0, 1, 2, 3, 4)), ("fr", (0, 0, 1, 1, 2)),
    ("FRR", (0, 0, 0, 1, 1)), ("FFR", (0, 1, 1, 2, 3)),
])
def test_pattern(pattern, expected):
    assert GPTConfig(n_layer=5, kv_share_pattern=pattern).kv_cache_layer_map() == expected


@pytest.mark.parametrize("pattern", ["", "R", "RF", "FX", "F R"])
def test_invalid_pattern(pattern):
    with pytest.raises(ValueError, match="kv_share_pattern"):
        GPT(GPTConfig(kv_share_pattern=pattern))


@pytest.mark.parametrize("layout", ["bhtd", "bthd"])
def test_physical_cache_savings_and_clone(layout):
    model = make_model("FRR", n_layer=6)
    cache = make_cache(model, layout=layout)
    dense = KVCache(1, 2, 64, 16, 6, "cpu", COMPUTE_DTYPE, layout=layout)
    assert cache.k_cache.numel() * 3 == dense.k_cache.numel()
    assert cache.v_cache.numel() * 3 == dense.v_cache.numel()
    for layer in (1, 2):
        assert cache.get_layer_cache(layer)[0].data_ptr() == cache.get_layer_cache(0)[0].data_ptr()
    cache.k_cache.fill_(1.25)
    cache.v_cache.fill_(2.5)
    cache.advance(7)
    cache.prev_embedding = torch.randn(1, 1, 64)
    clone = make_cache(model, batch_size=3, seq_len=80, layout=layout)
    clone.prefill(cache)
    assert clone.get_pos() == 7
    torch.testing.assert_close(clone.prev_embedding, cache.prev_embedding.expand(3, -1, -1))
    time_slice = (slice(None), slice(None), slice(None), slice(0, 7)) if layout == "bhtd" else (slice(None), slice(None), slice(0, 7))
    assert (clone.k_cache[time_slice] == 1.25).all()
    assert (clone.v_cache[time_slice] == 2.5).all()
    clone.reset()
    assert clone.get_pos() == 0 and clone.prev_embedding is None


@pytest.mark.parametrize("mapping", [(0,), (1, 1), (0, 2), (0, -1), (0, 0.0)])
def test_invalid_cache_mapping(mapping):
    with pytest.raises(ValueError, match="layer_mapping"):
        KVCache(1, 2, 16, 16, 2, "cpu", COMPUTE_DTYPE, layer_mapping=mapping)


@pytest.mark.parametrize("pattern", ["F", "FR", "FRR", "FFR"])
@pytest.mark.parametrize("layout", ["bhtd", "bthd"])
@pytest.mark.parametrize("n_kv_head", [2, 4])
def test_cached_chunks_match_full_sequence(pattern, layout, n_kv_head):
    model = make_model(pattern, n_kv_head=n_kv_head).eval()
    # Different attention windows per layer, including sharing across an S/L boundary.
    model.window_sizes = [(2, 0), (4, 0), (3, 0), (-1, 0)]
    ids = torch.randint(0, model.config.vocab_size, (2, 11))
    cache = make_cache(model, batch_size=2, layout=layout)
    with torch.inference_mode():
        expected = model(ids)
        outputs = []
        for start, end in [(0, 5), (5, 6), (6, 9), (9, 11)]:
            outputs.append(model(ids[:, start:end], kv_cache=cache))
            assert cache.get_pos() == end
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=1e-6, rtol=1e-5)


def test_future_tokens_do_not_change_prefix():
    model = make_model("FRR").eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 12))
    changed = ids.clone()
    changed[:, 7:] = (changed[:, 7:] + 1) % model.config.vocab_size
    with torch.no_grad():
        torch.testing.assert_close(model(ids)[:, :7], model(changed)[:, :7])


def test_consumer_gradients_reach_producer_kv_and_value_embedding():
    model = make_model("FR", n_layer=2)
    producer, consumer = (block.attn for block in model.transformer.h)
    assert consumer.c_k is None and consumer.c_v is None and consumer.ve_gate is None
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    ve = model.value_embeds["0"](ids)
    cos_sin = model.cos[:, :6], model.sin[:, :6]
    # Loss uses only the consumer output; the producer's own output is discarded.
    _, shared = producer(torch.randn(1, 6, 64), ve, cos_sin, (-1, 0), None)
    output, _ = consumer(torch.randn(1, 6, 64), None, cos_sin, (-1, 0), None, shared)
    output.square().sum().backward()
    for p in (producer.c_k.weight, producer.c_v.weight,
              model.value_embeds["0"].weight, consumer.c_q.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_cache_mismatch_fails_before_writing():
    model = make_model()
    other = make_cache(make_model("F"))
    ids = torch.ones(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="layer_mapping"):
        model(ids, kv_cache=other)
    assert other.get_pos() == 0 and other.prev_embedding is None
    with pytest.raises(ValueError, match="different layer mappings"):
        make_cache(model).prefill(other)


@pytest.mark.parametrize("pattern", ["F", "FRR"])
def test_checkpoint_round_trip_and_legacy_config(tmp_path, monkeypatch, pattern):
    model = make_model(pattern)
    config = asdict(model.config)
    if pattern == "F":
        del config["kv_share_pattern"]  # old checkpoints must keep their original architecture
    checkpoint_manager.save_checkpoint(str(tmp_path), 1, model.state_dict(), None, {"model_config": config})
    class Tokenizer:
        def get_vocab_size(self):
            return model.config.vocab_size
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", Tokenizer)
    loaded, _, _ = checkpoint_manager.build_model(str(tmp_path), 1, torch.device("cpu"), "eval")
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), model(ids), rtol=0, atol=0)
    assert loaded.kv_layer_mapping == model.kv_layer_mapping


def test_training_optimizer_covers_all_parameters():
    model = make_model("FRR")
    optimizer = model.setup_optimizer()
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(optimized) == len({id(p) for p in optimized})
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters()}
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(ids, targets=ids.roll(-1, dims=1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    before = model.lm_head.weight.detach().clone()
    optimizer.step()
    assert torch.isfinite(model.lm_head.weight).all()
    assert not torch.equal(before, model.lm_head.weight)
    assert model.num_scaling_params()["total"] == sum(p.numel() for p in model.parameters())
    assert model.estimate_flops() < make_model("F").estimate_flops()


@pytest.mark.parametrize("prompt", [[40], [40, 20, 80, 60]])
def test_engine_and_gpt_generation_agree(prompt):
    model = make_model().eval()
    tokenizer = ByteTokenizer()
    # Keep greedy generations in byte tokens so Engine cannot terminate on a special token.
    with torch.no_grad():
        model.lm_head.weight[256:].zero_()
    expected = list(model.generate(prompt, max_tokens=4, temperature=0))
    result, masks = Engine(model, tokenizer).generate_batch(prompt, num_samples=3, max_tokens=4, temperature=0)
    for row in result:
        assert row == prompt + expected


def test_read_only_sdpa_cache_uses_populated_length():
    q = torch.randn(1, 2, 4, 16)
    k, v = torch.randn(1, 5, 2, 16), torch.randn(1, 5, 2, 16)
    kc, vc = torch.full((1, 16, 2, 16), float("nan")), torch.full((1, 16, 2, 16), float("nan"))
    kc[:, :5], vc[:, :5] = k, v
    y = fa.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=torch.tensor([5], dtype=torch.int32), causal=True)
    expected = fa._sdpa_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), (-1, -1), True)
    torch.testing.assert_close(y, expected.transpose(1, 2))
    assert torch.isnan(kc[:, 5:]).all() and torch.isnan(vc[:, 5:]).all()


@pytest.mark.skipif(not fa.HAS_FA3, reason="Hopper GPU with FA3 required")
@pytest.mark.parametrize("chunk_len", [1, 3])
def test_read_only_fa3_matches_sdpa(monkeypatch, chunk_len):
    q = torch.randn(2, chunk_len, 4, 32, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(2, 16, 2, 32, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    lengths = torch.full((2,), 9, device="cuda", dtype=torch.int32)
    monkeypatch.setattr(fa, "USE_FA3", True)
    actual = fa.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=lengths, causal=True, window_size=(3, 0))
    monkeypatch.setattr(fa, "USE_FA3", False)
    expected = fa.flash_attn_with_kvcache(q, kc, vc, cache_seqlens=lengths, causal=True, window_size=(3, 0))
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(os.environ.get("TORCHDYNAMO_DISABLE") == "1", reason="run separately with Dynamo enabled")
def test_compiled_training_matches_eager_gradients():
    model = make_model()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.compiler.set_stance("force_eager"):
        expected = model(ids, targets=ids)
        expected.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    actual = compiled(ids, targets=ids)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, grads[name])
