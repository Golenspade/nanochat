"""
GSM8K Chinese evaluation.
https://huggingface.co/datasets/swulling/gsm8k_chinese

This is a Chinese translation of GSM8K.
The dataset contains:
- question: original English question
- answer: original English answer (with #### marker)
- question_zh-cn: Chinese translation of the question
- answer_only: numeric answer (int or float)

We use the Chinese question for the prompt and compare the numeric answer.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Falls back to the last number in the text if no #### marker is found.
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    # fallback: last number in text
    numbers = re.findall(r"-?\d+\.?\d*", completion)
    if numbers:
        return numbers[-1]
    return None


class GSM8KCN(Task):

    def __init__(self, subset="main", split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "GSM8KCN split must be train|test"
        self.ds = load_dataset("swulling/gsm8k_chinese", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.ds[index]
        question = row["question_zh-cn"]
        answer = str(row["answer_only"])
        conversation = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct).
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # Ground truth answer is the assistant message content
        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant"
        ref_num = str(assistant_message["content"]).strip()
        pred_num = extract_answer(assistant_response)
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. Re-use the evaluation above.
        """
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)
