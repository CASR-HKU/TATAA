import math
import argparse
import evaluate 
from itertools import chain
from datasets import load_dataset

from transformers import(
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from hlibf_gpt2 import *

def main():

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--parse_size", "-c", type=int, default=1, help="Parse size."
    )
    args = parser.parse_args()
    dataset_cache_dir = args.dataset_cache_dir
    eval_cache_dir = args.eval_cache_dir
    parse_size = args.parse_size

    dataset_name = "wikitext"
    dataset_config_name = "wikitext-2-raw-v1"

    finetuned_model_path = "tmp/test-clm-medium"
    config = GPT2Config.from_pretrained(finetuned_model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(finetuned_model_path)

    config.quant = False
    config.calibrate = False
    gpt2_model = QGPT2LMHeadModel.from_pretrained(finetuned_model_path, config=config)

    # dataset
    raw_datasets = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        cache_dir=dataset_cache_dir,
    )

    column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output
    
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    max_pos_embeddings = config.max_position_embeddings
    block_size = tokenizer.model_max_length // 2
    print("Block Size: ", block_size)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if "validation" not in tokenized_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", cache_dir=eval_cache_dir)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

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
        run_name="run_name",
        report_to="none",
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=gpt2_model,
        args=eval_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    print("Parsing GPT2 model...")
    gpt2_model.cuda(device=0).eval()
    parse_dataset = train_dataset.select(range(parse_size))
    for i in range(10000):
        with torch.no_grad():
            metrics = trainer.evaluate(eval_dataset=parse_dataset)

if __name__ == "__main__":
    main()