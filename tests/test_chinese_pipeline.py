from types import SimpleNamespace

from nanochat import dataset as dataset_module
from nanochat.dataset import _interleave_iters, skypile_iter_batched
from nanochat.report import Report
from nanochat.sft_mixture import build_sft_task_lists


def test_cn_ratio_zero_keeps_english_only():
    out = list(_interleave_iters(iter(["en1", "en2"]), iter(["cn1"]), 0.0))
    assert out == ["en1", "en2"]


def test_cn_ratio_half_interleaves_batches():
    out = list(_interleave_iters(iter(["en1", "en2", "en3"]), iter(["cn1", "cn2", "cn3"]), 0.5))
    assert out[:6] == ["en1", "cn1", "en2", "cn2", "en3", "cn3"]


def test_skypile_streaming_rank_stride_fallback(monkeypatch):
    docs = [{"text": f"doc{i}"} for i in range(6)]
    monkeypatch.setattr(dataset_module, "_load_skypile_streaming", lambda split: docs)

    rank0 = list(skypile_iter_batched(batch_size=2, start=0, step=2))
    rank1 = list(skypile_iter_batched(batch_size=2, start=1, step=2))

    assert rank0 == [["doc0", "doc2"], ["doc4"]]
    assert rank1 == [["doc1", "doc3"], ["doc5"]]
    assert set(rank0[0]).isdisjoint(rank1[0])


class MarkerTask:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __len__(self):
        return 1


def _factory(name):
    return lambda **kwargs: MarkerTask(name, **kwargs)


FACTORIES = {
    "SmolTalk": _factory("SmolTalk"),
    "CustomJSON": _factory("CustomJSON"),
    "MMLU": _factory("MMLU"),
    "GSM8K": _factory("GSM8K"),
    "SimpleSpelling": _factory("SimpleSpelling"),
    "SpellingBee": _factory("SpellingBee"),
    "SmolTalkCN": _factory("SmolTalkCN"),
    "GSM8KCN": _factory("GSM8KCN"),
    "CEval": _factory("CEval"),
}


def _args(**overrides):
    defaults = dict(
        mmlu_epochs=0,
        gsm8k_epochs=0,
        cn_talk_epochs=0,
        gsm8k_cn_epochs=0,
        ceval_epochs=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_sft_mixture_default_has_no_chinese_tasks():
    train_tasks, val_tasks, summary = build_sft_task_lists(_args(), "/tmp/nanochat", factories=FACTORIES)

    assert all(not task.name.endswith("CN") and task.name != "CEval" for task in train_tasks)
    assert all(not task.name.endswith("CN") and task.name != "CEval" for task in val_tasks)
    assert summary["SmolTalkCN"] == 0
    assert summary["GSM8KCN"] == 0
    assert summary["CEval"] == 0


def test_sft_mixture_adds_chinese_tasks_when_enabled():
    train_tasks, val_tasks, summary = build_sft_task_lists(
        _args(cn_talk_epochs=2, gsm8k_cn_epochs=1, ceval_epochs=1),
        "/tmp/nanochat",
        factories=FACTORIES,
    )
    train_names = [task.name for task in train_tasks]
    val_names = [task.name for task in val_tasks]

    assert train_names.count("SmolTalkCN") == 2
    assert train_names.count("GSM8KCN") == 1
    assert train_names.count("CEval") == 1
    assert "SmolTalkCN" in val_names
    assert "GSM8KCN" in val_names
    assert summary["SmolTalkCN"] == 2
    assert summary["GSM8KCN"] == 1
    assert summary["CEval"] == 1


def test_report_includes_chinese_evaluation_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "header.md").write_text(
        "# nanochat training report\n\n### Bloat\n- Lines: 1\n\nRun started: 2026-01-01 00:00:00\n\n",
        encoding="utf-8",
    )
    (report_dir / "chinese-evaluation-sft.md").write_text(
        "## Chinese evaluation sft\n"
        "timestamp: 2026-01-01 00:01:00\n\n"
        "- CEval: 0.3000\n"
        "- GSM8K-CN: 0.1000\n"
        "- Chinese ChatCORE: 0.0667\n",
        encoding="utf-8",
    )

    report_path = Report(str(report_dir)).generate()
    content = open(report_path, encoding="utf-8").read()

    assert "Chinese evaluation sft" in content
    assert "Chinese ChatCORE" in content
    assert "CHINESE-SFT" in content
