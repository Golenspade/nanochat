"""Structural and causal acceptance tests for the scaled V4.1 text backbone."""
from dataclasses import asdict

import pytest
import torch

from nanochat.deepseek_v41 import DeepSeekV41, V41Config
from nanochat.v41_components import pack_fp4, unpack_fp4, fp4_ste, SinglePassMHC


@pytest.fixture(autouse=True)
def cpu_dtype(monkeypatch):
    import nanochat.deepseek_v41 as module
    monkeypatch.setattr(module, "COMPUTE_DTYPE", torch.float32)


def model(**kwargs):
    torch.manual_seed(7)
    config = V41Config(vocab_size=128, n_embd=64, n_head=2, head_dim=32,
        n_layer=8, sequence_len=128, window_size=4, index_topk=2,
        index_dim=16, index_heads=2, candidate_blocks=1, candidate_block_size=2,
        q_rank=32, o_rank=32, n_experts=4, experts_per_token=2,
        expert_dim=32, engram_dim=8, engram_buckets=17, **kwargs)
    with torch.device("meta"):
        result = DeepSeekV41(config)
    result.to_empty(device="cpu")
    result.init_weights()
    return result


def test_fp4_storage_and_gradient():
    x = torch.randn(2, 7, 32, requires_grad=True)
    packed, scales = pack_fp4(x)
    assert packed.dtype == torch.uint8 and scales.dtype == torch.float8_e4m3fn
    assert packed.numel() == x.numel() // 2
    assert scales.numel() == x.numel() // 16
    restored = unpack_fp4(packed, scales)
    assert restored.shape == x.shape and torch.isfinite(restored).all()
    assert (restored - x).square().mean() < 0.04
    fp4_ste(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.parametrize("fp4", [False, True])
def test_full_chunk_and_token_cache_agree(fp4):
    m = model(fp4_cache=fp4).eval()
    ids = torch.randint(0, 128, (2, 13))
    c = m.create_kv_cache(batch_size=2, seq_len=40)
    with torch.inference_mode():
        expected = m(ids)
        actual = torch.cat([m(ids[:, :5], kv_cache=c), m(ids[:, 5:6], kv_cache=c),
                            m(ids[:, 6:9], kv_cache=c), m(ids[:, 9:], kv_cache=c)], dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    assert c.get_pos() == ids.size(1)
    assert all(k.size(1) <= m.config.window_size for k in c.local.values())
    assert c.global_counts[1] == ids.size(1) // 2
    assert c.global_counts[4] == ids.size(1)


def test_future_tokens_cannot_affect_compressed_or_local_prefix():
    m = model().eval()
    ids = torch.randint(0, 128, (1, 14))
    changed = ids.clone(); changed[:, 5:] = (changed[:, 5:] + 1) % 128
    with torch.inference_mode():
        torch.testing.assert_close(m(ids)[:, :5], m(changed)[:, :5], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fp4", [False, True])
def test_cache_clone_reset_and_isolation(fp4):
    m = model(fp4_cache=fp4).eval()
    ids = torch.randint(0, 128, (1, 7))
    source = m.create_kv_cache(1, 40)
    with torch.inference_mode():
        m(ids, kv_cache=source)
        clone = m.create_kv_cache(3, 40)
        clone.prefill(source)
        next_id = torch.tensor([[42]])
        a = m(next_id, kv_cache=source)
        b = m(next_id.expand(3, 1), kv_cache=clone)
    torch.testing.assert_close(b, a.expand(3, -1, -1), atol=2e-5, rtol=2e-4)
    clone.reset()
    assert clone.get_pos() == 0 and not clone.local and clone.history is None
    other = m.create_kv_cache(1, 40)
    with torch.inference_mode():
        torch.testing.assert_close(m(ids, kv_cache=other), m(ids), atol=2e-5, rtol=2e-4)


def test_ced_global_kv_comes_from_encoder():
    m = model().eval()
    ids = torch.randint(0, 128, (1, 11))
    with torch.inference_mode():
        c1 = m.create_kv_cache(1, 40); m(ids, kv_cache=c1)
        # Changing decoder residual mixing must not change decoder global KV.
        m.blocks[4].attn_mix.base.add_(0.4)
        c2 = m.create_kv_cache(1, 40); m(ids, kv_cache=c2)
    torch.testing.assert_close(c1.globals[4], c2.globals[4], atol=0, rtol=0)


def test_mhc_sinkhorn_and_training_gradients():
    m = model().train()
    ids = torch.randint(0, 128, (2, 12))
    loss = m(ids, targets=ids.roll(-1, 1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.blocks[1].attn.compressor.weight.grad.abs().sum() > 0
    assert m.blocks[1].attn.index_q.weight.grad.abs().sum() > 0
    assert m.engram["1"].tables[0].weight.grad.abs().sum() > 0
    mix = m.blocks[0].attn_mix
    x = torch.randn(2, 3, m.config.hc_mult, m.config.n_embd)
    _, _, comb = mix.coefficients(x)
    torch.testing.assert_close(comb.sum(-1), torch.ones_like(comb.sum(-1)), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=2e-4, rtol=2e-4)
    before = m.blocks[0].ffn.bias.clone()
    m.post_optimizer_step()
    assert not torch.equal(before, m.blocks[0].ffn.bias)


def test_bounded_replay_keeps_full_encoder_and_only_decoder_tail():
    m = model().eval()
    ids = torch.randint(0, 128, (1, 19))
    c = m.create_kv_cache(1, 40, bounded_replay=True)
    with torch.inference_mode():
        logits = m(ids, kv_cache=c)
        assert logits.shape == (1, 4, 128)
        assert c.get_pos() == 19 and c.global_counts[4] == 19
        assert c.layer_tokens[0] == 19 and c.layer_tokens[4] == 4
        assert m(torch.tensor([[5]]), kv_cache=c).shape == (1, 1, 128)


def test_config_roundtrip_and_reject_invalid():
    m = model()
    config = V41Config(**asdict(m.config))
    assert config.layer_modes() == ("S", "F", "I", "R", "F", "I", "R", "R")
    assert config.sources() == {1: 2, 4: 1}
    with pytest.raises(ValueError):
        V41Config(n_layer=5)


def test_one_token_prompt_and_engine_factory():
    from nanochat.engine import Engine
    from test_engine import ByteTokenizer
    m = model().eval()
    ids = torch.tensor([[30]])
    c = m.create_kv_cache(1, 8)
    with torch.inference_mode():
        assert m(ids, kv_cache=c).shape == (1, 1, 128)
        assert c.global_counts == {1: 0, 4: 1}
    sequences, masks = Engine(m, ByteTokenizer()).generate_batch([30], num_samples=2, max_tokens=3, temperature=0)
    assert sequences[0] == sequences[1] and len(sequences[0]) == 4


def test_checkpoint_loader_routes_architecture(tmp_path, monkeypatch):
    import nanochat.checkpoint_manager as manager
    m = model().eval()
    manager.save_checkpoint(str(tmp_path), 1, m.state_dict(), None, {"model_config": asdict(m.config)})
    class Tokenizer:
        def get_vocab_size(self): return 128
    monkeypatch.setattr(manager, "get_tokenizer", Tokenizer)
    restored, _, _ = manager.build_model(str(tmp_path), 1, torch.device("cpu"), "eval")
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(restored(ids), m(ids), atol=0, rtol=0)


def test_dspark_parallel_drafts_and_frozen_backbone():
    from nanochat.v41_dspark import DSparkDrafter
    m = model().eval()
    ids = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        teacher, features = m(ids, return_hidden=True)
    draft = DSparkDrafter(m, block_size=3, markov_rank=8)
    draft.init_weights()
    targets = ids[:, 9:12]
    loss = draft.training_loss(features[:, :8], ids[:, 8], 8, targets, teacher[:, 8:11])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in m.parameters())
    assert draft.markov.weight.grad.abs().sum() > 0
    result = draft(features[:, :8], ids[:, 8], 8)
    assert result["tokens"].shape == (2, 3)
    lengths = draft.verification_length(result["confidence"], 1., [2., 3., 4.])
    assert ((lengths >= 1) & (lengths <= 3)).all()


def test_engram_normalized_token_map():
    from nanochat.v41_components import build_token_map
    class Tokenizer:
        def get_vocab_size(self): return 4
        def decode(self, ids): return [" The", "THE", "thé", " "][ids[0]]
    mapping = build_token_map(Tokenizer())
    assert mapping.tolist() == [0, 0, 0, 1]


@pytest.mark.parametrize("fp4", [False, True])
def test_encoder_bounded_replay_does_not_overwrite_global_prefix(fp4):
    m = model(fp4_cache=fp4).eval()
    prefix, suffix = torch.randint(0,128,(1,11)), torch.randint(0,128,(1,3))
    c = m.create_kv_cache(1, 40, bounded_replay=True)
    with torch.inference_mode():
        m(prefix, kv_cache=c)
        saved = {owner: data[:, :c.global_counts[owner]].clone() for owner,data in c.globals.items()}
        c.drop_local()
        with pytest.raises(ValueError, match="Missing SWA"):
            m(suffix, kv_cache=c)
        result = m.replay_prefix(c, prefix, suffix)
        assert torch.isfinite(result).all() and c.get_pos() == 14
        assert c.global_counts == {1:7, 4:14}
        for owner, old in saved.items():
            torch.testing.assert_close(c.globals[owner][:,:old.size(1)],old,atol=0,rtol=0)
