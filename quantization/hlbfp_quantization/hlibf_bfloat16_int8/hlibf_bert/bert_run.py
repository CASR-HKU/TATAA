from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
from hibf_bert import *
from hibf_config import Config, BitType

import argparse
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
    DistributedSampler,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertConfig, BertTokenizer
from transformers import BertForSequenceClassification
from hibf_bert import QBertForSequenceClassification
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import (
    glue_convert_examples_to_features as convert_examples_to_features,
)

# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logging.getLogger("transformers.modeling_utils").setLevel(
    logging.INFO
)  # Reduce logging

torch.set_num_threads(4)
print(torch.__config__.parallel_info())


# Set random seed for reproducibility.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (
        ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    )
    eval_outputs_dirs = (
        (args.output_dir, args.output_dir + "-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=True
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = (
            SequentialSampler(eval_dataset)
            if args.local_rank == -1
            else DistributedSampler(eval_dataset)
        )
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
                )

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def calibrate(args, model, tokenizer, prefix=""):
    # only use a small part of the training data to calibrate
    # NOTE: Only need to inference once to calibrate, no need to calculate accuracy
    eval_task_names = (
        ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    )
    eval_outputs_dirs = (
        (args.output_dir, args.output_dir + "-MM")
        if args.task_name == "mnli"
        else (args.output_dir,)
    )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        calib_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=False
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        args.calib_batch_size = 512
        # Note that DistributedSampler samples randomly
        calib_sampler = (
            SequentialSampler(calib_dataset)
            if args.local_rank == -1
            else DistributedSampler(calib_dataset)
        )
        calib_dataloader = DataLoader(
            calib_dataset, sampler=calib_sampler, batch_size=args.calib_batch_size
        )

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Calibrate!
        logger.info("***** Running calibration {} *****".format(prefix))
        logger.info("  Num examples = %d", len(calib_dataset))
        logger.info("  Batch size = %d", args.calib_batch_size)
        nb_eval_steps = 0
        for batch in tqdm(calib_dataloader, desc="Calibrating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )
                outputs = model(**inputs)

            nb_eval_steps += 1

            # if nb_eval_steps == 9:
            #     model.model_open_last_calibrate()

            if nb_eval_steps == 1:
                break

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir)
            if evaluate
            else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            task=None,
            # pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            # pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            # pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    )
    return dataset


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p") / 1e6)
    os.remove("temp.p")


def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    result = evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print(result)
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))


# Test the bert-base-uncased on glue dataset, which consists of nine sub-tasks
# The output directory for the fine-tuned model, taking the model fine-tuned on MPRC for example
output_dir = "./bert_finetuned/SST2/"
config = BertConfig.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)
# tokenizer = AutoTokenizer.from_pretrained(
#     "MasterGuda/bert-base-uncased-gluesst2-finetuned-gpu"
# )

config.output_dir = output_dir
# config.output_dir = "./CoLA/"

# The data directory for the MRPC task in the GLUE benchmark, $GLUE_DIR/$TASK_NAME.
config.data_dir = "./dataset/SST-2"

# The model name or path for the pre-trained model.
config.model_name_or_path = "bert-base-uncased"
# The maximum length of an input sequence
config.max_seq_length = 128

# Prepare GLUE task.
config.task_name = "CoLA".lower()
# config.task_name = "CoLA".lower()
config.processor = processors[config.task_name]()
config.output_mode = output_modes[config.task_name]
config.label_list = config.processor.get_labels()
config.model_type = "bert".lower()
config.do_lower_case = True

# Set the device, batch size, topology, and caching flags.
config.device = "cuda"
config.per_gpu_eval_batch_size = 8
config.n_gpu = 0
config.local_rank = -1
config.overwrite_cache = False

config.tanh_quant = True

# config.output_dir stores the fine-tuned model
# Original model
if sys.argv[1] == "fp32":
    # get the quantization config
    config.hibf_quant = False
    config.calibrate = False
    config.quant_cfg = Config()

    model = BertForSequenceClassification.from_pretrained(
        config.output_dir, config=config
    )

    model.to(config.device)
    print(model)
    time_model_evaluation(model, config, tokenizer)
# Quantized model
elif sys.argv[1] == "mib":
    # get the quantization config
    config.hibf_quant = False
    config.calibrate = False
    config.quant_cfg = Config()

    quantized_model = QBertForSequenceClassification.from_pretrained(
        config.output_dir, config=config
    )
    quantized_model.to(config.device)
    # print(quantized_model)

    # Calibrate
    print("Calibrating the model")
    quantized_model.model_open_calibrate()
    quantized_model.model_open_last_calibrate()
    calibrate(config, quantized_model, tokenizer, prefix="")
    quantized_model.model_close_calibrate()
    quantized_model.model_quant()

    print("Evaluating the model")
    time_model_evaluation(quantized_model, config, tokenizer)
