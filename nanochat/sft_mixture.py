"""
Helpers for constructing SFT task mixtures.
"""

import os

from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee
from tasks.smoltalk_cn import SmolTalkCN
from tasks.gsm8k_cn import GSM8KCN
from tasks.ceval import CEval


DEFAULT_FACTORIES = {
    "SmolTalk": SmolTalk,
    "CustomJSON": CustomJSON,
    "MMLU": MMLU,
    "GSM8K": GSM8K,
    "SimpleSpelling": SimpleSpelling,
    "SpellingBee": SpellingBee,
    "SmolTalkCN": SmolTalkCN,
    "GSM8KCN": GSM8KCN,
    "CEval": CEval,
}


def build_sft_task_lists(args, base_dir, factories=None):
    """Build train/val task lists for chat_sft.py."""
    factories = DEFAULT_FACTORIES if factories is None else factories
    identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")

    train_tasks = [
        factories["SmolTalk"](split="train"),
        factories["CustomJSON"](filepath=identity_conversations_filepath),
        factories["CustomJSON"](filepath=identity_conversations_filepath),
        *[factories["MMLU"](subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[factories["GSM8K"](subset="main", split="train") for _ in range(args.gsm8k_epochs)],
        factories["SimpleSpelling"](size=200000, split="train"),
        factories["SpellingBee"](size=80000, split="train"),
    ]

    train_tasks.extend(factories["SmolTalkCN"](split="train") for _ in range(args.cn_talk_epochs))
    train_tasks.extend(factories["GSM8KCN"](subset="main", split="train") for _ in range(args.gsm8k_cn_epochs))
    train_tasks.extend(factories["CEval"](split="val") for _ in range(args.ceval_epochs))

    val_tasks = [
        factories["SmolTalk"](split="test"),
        factories["MMLU"](subset="all", split="test", stop=5200),
        factories["GSM8K"](subset="main", split="test", stop=420),
    ]
    if args.cn_talk_epochs > 0 or args.gsm8k_cn_epochs > 0 or args.ceval_epochs > 0:
        val_tasks.append(factories["SmolTalkCN"](split="test", stop=2000))
        val_tasks.append(factories["GSM8KCN"](subset="main", split="test", stop=420))

    summary = {
        "MMLU": args.mmlu_epochs,
        "GSM8K": args.gsm8k_epochs,
        "SmolTalkCN": args.cn_talk_epochs,
        "GSM8KCN": args.gsm8k_cn_epochs,
        "CEval": args.ceval_epochs,
    }
    return train_tasks, val_tasks, summary
