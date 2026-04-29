"""
C-Eval: A Comprehensive Chinese Evaluation Suite for Foundation Models.
https://huggingface.co/datasets/ceval/ceval-exam
"""

from datasets import load_dataset, get_dataset_config_names
from tasks.common import Task, render_mc

class CEval(Task):
    """ C-Eval Chinese multiple-choice evaluation dataset. 52 subjects. """

    letters = ('A', 'B', 'C', 'D')

    def __init__(self, split, subset="all", **kwargs):
        super().__init__(**kwargs)
        assert split in ["val", "test"], f"CEval split {split} must be val|test"
        self.split = split
        self.subset = subset

        if subset == "all":
            self.subjects = get_dataset_config_names("ceval/ceval-exam")
        else:
            self.subjects = [subset]

        self.datasets = []
        self.lengths = []
        for subject in self.subjects:
            ds = load_dataset("ceval/ceval-exam", name=subject, split=split)
            self.datasets.append(ds)
            self.lengths.append(len(ds))

        self.total_length = sum(self.lengths)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return self.total_length

    def _get_dataset_and_local_index(self, index):
        """Map global index to (dataset, local_index) pair."""
        for ds, length in zip(self.datasets, self.lengths):
            if index < length:
                return ds, index
            index -= length
        raise IndexError(f"Index {index} out of range for CEval with {self.total_length} examples")

    def get_example(self, index):
        ds, local_index = self._get_dataset_and_local_index(index)
        row = ds[local_index]
        question = row["question"]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        answer = row["answer"]
        assert answer in self.letters, f"CEval answer {answer} is expected to be one of {self.letters}"
        user_message = render_mc(question, self.letters, choices)
        assistant_message = answer
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # Narrow to first letter and clamp to valid choices
        assistant_letter = assistant_response[0].upper() if assistant_response else ""
        if assistant_letter not in self.letters:
            return 0.0
        correct_answer = conversation['messages'][-1]['content']
        return 1.0 if assistant_letter == correct_answer else 0.0
