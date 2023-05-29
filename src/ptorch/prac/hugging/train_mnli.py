from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import evaluate
from transformers import TrainingArguments, Trainer
import numpy as np
import torch


class ConstantDummy(torch.nn.Module):
    def __init__(self, num_labels):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(num_labels))

    def forward(self, **kwargs):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        return self.a

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()}'


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        # implement custom logic here
        print(model, inputs)

        return 0


def main():
    dataset = load_dataset("multi_nli")
    model_type = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    def tokenize_function(examples):
        return tokenizer(
            examples["premise"],
                         examples["hypothesis"], padding="max_length", truncation=True)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    print(dataset.keys())
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["validation_matched"].shuffle(seed=42).select(range(1000))

    # model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=5)
    model = ConstantDummy(num_labels=2)
    training_args = TrainingArguments(output_dir="test_trainer")
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()



if __name__ == "__main__":
    main()