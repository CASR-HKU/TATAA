import argparse
import evaluate
import numpy as np

from transformers import (
    BertConfig,
    BertTokenizer,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from datasets import load_dataset

from hibf_bert import *
from hibf_config import Config

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", "-t", type=str, default="MRPC", help="The GLUE task to run"
    )
    parser.add_argument(
        "--max_seq_length",
        "-m",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=32,
        help="The maximum number of sentence row in datasets to be evaluated.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        "-d",
        type=str,
        default="./dataset_cache",
        help="The directory where the dataset is saved.",
    )
    parser.add_argument(
        "--eval_cache_dir",
        "-e",
        type=str,
        default="./eval_cache",
        help="The directory where the evaluation results are saved.",
    )

    args = parser.parse_args()

    task_name_upper = args.task
    task_name = task_name_upper.lower()
    max_seq_length = args.max_seq_length
    dataset_cache_dir = args.dataset_cache_dir
    eval_cache_dir = args.eval_cache_dir

    print("The GLUE task running now is: ", task_name_upper)

    finetuned_model_path = "bert_finetuned/" + task_name_upper

    # NOTE: all the pre-processing steps are copied from the original run_glue.py
    # Load the model, config and tokenizer
    config = BertConfig.from_pretrained(finetuned_model_path)
    tokenizer = BertTokenizer.from_pretrained(finetuned_model_path)

    config.hibf_quant = False
    config.tanh_quant = True
    config.calibrate = False
    config.device = "cuda"
    config.quant_cfg = Config()
    qbert_model = QBertForSequenceClassification.from_pretrained(
        finetuned_model_path, config=config
    )
    qbert_model.to("cuda")

    # dataset
    raw_datasets = load_dataset("nyu-mll/glue", task_name, cache_dir=dataset_cache_dir)

    # Labels
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)

    # Load fintuned model and tokenizer
    config = BertConfig.from_pretrained(finetuned_model_path)
    tokenizer = BertTokenizer.from_pretrained(finetuned_model_path)

    # Preprocessing the raw_datasets
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # Padding strategy
    padding = "max_length"

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)
            }
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif task_name is not None:
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {id: label for label, id in config.label2id.items()}

    if max_seq_length > tokenizer.model_max_length:
        print(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [
                (label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # convert validation set for quantized inference now
    if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.select(range(args.batch_size))

    # evaluation metric
    metric = evaluate.load("glue", task_name, cache_dir=eval_cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # we already did the padding.
    data_collator = default_data_collator

    # Initialize our Trainer
    eval_args = TrainingArguments(
        per_device_eval_batch_size=args.batch_size,
        eval_accumulation_steps=1,
        output_dir="./eval_results",
        logging_dir="./logs",
        logging_steps=1,
        do_eval=True,
        do_train=False,
        evaluation_strategy="steps",
        eval_steps=1,
        load_best_model_at_end=False,
        metric_for_best_model="combined_score",
        greater_is_better=False,
        run_name="run_name",
        report_to="none",
    )

    trainer = Trainer(
        model=qbert_model,
        args=eval_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("Evaluating the model...")
    for i in range(10000):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)


if __name__ == "__main__":
    main()
