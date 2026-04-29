"""
Chinese conversation dataset based on BelleGroup/train_1M_CN.
https://huggingface.co/datasets/BelleGroup/train_1M_CN
"""

from datasets import load_dataset
from tasks.common import Task

class SmolTalkCN(Task):
    """ BelleGroup/train_1M_CN dataset. ~1M rows of Chinese instruction-following data. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalkCN split must be train|test"
        ds = load_dataset("BelleGroup/train_1M_CN", split="train").shuffle(seed=42)
        if split == "test":
            # Take a 5% holdout for test since there is no native test split
            test_size = int(len(ds) * 0.05)
            self.ds = ds.select(range(test_size))
        else:
            # Train gets the remaining 95%
            test_size = int(len(ds) * 0.05)
            self.ds = ds.select(range(test_size, len(ds)))
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        instruction = row["instruction"]
        if row.get("input"):
            instruction = instruction + "\n" + row["input"]
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": row["output"]}
        ]
        # ---------------------------------------------------------------------
        # sanity checking asserts
        assert len(messages) >= 2, "SmolTalkCN messages must have at least 2 messages"
        for i, message in enumerate(messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
        # ---------------------------------------------------------------------
        conversation = {
            "messages": messages,
        }
        return conversation
