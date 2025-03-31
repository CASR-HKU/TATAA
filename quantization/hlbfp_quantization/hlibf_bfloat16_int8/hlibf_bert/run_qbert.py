import argparse
import evaluate
import numpy as np

from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification,
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
        "--task", "-t", type=str, default="SST2", help="The GLUE task to run"
    )
    parser.add_argument(
        "--max_seq_length",
        "-m",
        type=int,
        default=128,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
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
    parser.add_argument(
        "--quantize",
        "-q",
        default=False,
        action="store_true"
    )
    # parser.add_argument('--quant', default=False, action='store_true')
    parser.add_argument(
        "--calibrate_size", "-c", type=int, default=128, help="Calibration size."
    )
    args = parser.parse_args()

    task_name_upper = args.task
    task_name = task_name_upper.lower()
    max_seq_length = args.max_seq_length
    dataset_cache_dir = args.dataset_cache_dir
    eval_cache_dir = args.eval_cache_dir
    if_quant = args.quantize
    calib_size = args.calibrate_size

    print("args quantize: ", args.quantize)

    print("The GLUE task running now is: ", task_name_upper)

    finetuned_model_path = "bert_finetuned/" + task_name_upper

    # NOTE: all the pre-processing steps are copied from the original run_glue.py
    # Load the model, config and tokenizer
    config = BertConfig.from_pretrained(finetuned_model_path)
    tokenizer = BertTokenizer.from_pretrained(finetuned_model_path)
    if if_quant == False:
        qbert_model = BertForSequenceClassification.from_pretrained(
            finetuned_model_path
        )
    else:
        config.hibf_quant = False
        config.tanh_quant = True
        config.calibrate = False
        config.device = "cuda"
        config.quant_cfg = Config()
        qbert_model = QBertForSequenceClassification.from_pretrained(
            finetuned_model_path, config=config
        )
        qbert_model.to("cuda")
        qbert_model.model_open_calibrate()
        qbert_model.model_open_last_calibrate()

    # dataset
    raw_datasets = load_dataset("nyu-mll/glue", task_name, cache_dir=dataset_cache_dir)

    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

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
        and not is_regression
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
    elif task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif task_name is not None and not is_regression:
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
    eval_dataset = raw_datasets[
        "validation_matched" if task_name == "mnli" else "validation"
    ]
    train_dataset = raw_datasets["train"]

    # evaluation metric
    metric = evaluate.load("glue", task_name, cache_dir=eval_cache_dir)

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # we already did the padding.
    data_collator = default_data_collator

    # Initialize our Trainer
    eval_args = TrainingArguments(
        per_device_eval_batch_size=8,
        eval_accumulation_steps=1,
        output_dir="./eval_results",
        logging_dir="./logs",
        logging_steps=10,
        do_eval=True,
        do_train=False,
        evaluation_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="combined_score",
        greater_is_better=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        run_name="run_name",
        report_to="none",
    )

    trainer = Trainer(
        model=qbert_model,
        args=eval_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    print("if quant or not: ", if_quant)

    # Original fp32 model inference: Only need to run evaluation
    if not if_quant:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [task_name]
        eval_datasets = [eval_dataset]
        if task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

        kwargs = {
            "finetuned_from": "bert-base-uncased",
            "tasks": "text-classification",
        }
        if task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = task_name
            kwargs["dataset"] = f"GLUE {task_name.upper()}"

        trainer.create_model_card(**kwargs)
        print("Evaluation Results: ", metrics)

    # if quant, then we need to calibrate the model first, using training set
    else:
        # Now, calibrate the model
        print("Calibrating the model...")

        # select calib_size of training set for calibration
        predict_dataset = train_dataset.select(range(calib_size))

        metrics = trainer.evaluate(eval_dataset=predict_dataset)

        qbert_model.model_close_calibrate()
        qbert_model.model_quant()

        print("Evaluating the model...")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        kwargs = {
            "finetuned_from": "bert-base-uncased",
            "tasks": "text-classification",
        }
        if task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = task_name
            kwargs["dataset"] = f"GLUE {task_name.upper()}"

        trainer.create_model_card(**kwargs)
        print("Evaluation Results: ", metrics)


if __name__ == "__main__":
    main()
