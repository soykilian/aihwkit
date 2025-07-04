# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example 24: Example using convert_to_analog to run BERT transformer on SQuAD task
**Source**:
    The example is adapted from code in
    https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""
# pylint: disable=invalid-name, too-many-locals, import-error
from transformers import TrainerCallback
import os
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from aihwkit.simulator.presets.inference import StandardHWATrainingPreset
from numpy import log10, logspace, argsort
from transformers.integrations import TensorBoardCallback

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
)

from torch import save as torch_save, load as torch_load
from torch.utils.tensorboard import SummaryWriter
import torch
from evaluate import load
from datasets import load_dataset

from aihwkit.simulator.configs import (
    InferenceRPUConfig, TorchInferenceRPUConfig,
    WeightModifierType, WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)

from aihwkit.simulator.presets import PresetIOParameters
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation, ReRamCMONoiseModel
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.optim import AnalogSGD


# BERT model from Hugging Face model hub fine-tuned on SQuAD v1
MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# Parse some arguments
PARSER = ArgumentParser("Analog BERT on SQuAD example")
PARSER.add_argument("-d", "--digital", help="Add to use digital inference", action="store_true")
PARSER.add_argument(
    "-i",
    "--ideal",
    help="Add to use ideal config instead of default noisy one",
    action="store_true",
)
PARSER.add_argument("-w", "--wandb", help="Add to use wandb", action="store_true")
PARSER.add_argument("-n", "--noise", help="Modifier noise", default=0.1, type=float)
PARSER.add_argument(
    "-r",
    "--run_name",
    help="Tensorboard run name",
    default=datetime.now().strftime("%Y%m%d-%H%M%S"),
    type=str,
)
PARSER.add_argument("-t", "--train_hwa", help="Use Hardware-Aware training", action="store_false")
PARSER.add_argument(
    "-L", "--load", help="Use when loadiung from training checkpoint", action="store_true"
)

PARSER.add_argument(
    "-c",
    "--checkpoint",
    help="File name specifying where to load/save a checkpoint",
    default="./saved_chkpt.pth",
    type=str,
)
PARSER.add_argument(
    "-l", "--learning_rate", help="Learning rate for training", default=2e-4, type=float
)

ARGS = PARSER.parse_args()

if ARGS.wandb:
    import wandb

    # Define weights noise sweep configuration
    SWEEP_CONFIG = {
        "method": "random",
        "name": "modifier noise sweep",
        "metric": {"goal": "maximize", "name": "exact_match"},
        "parameters": {"modifier_noise": {"values": [0, 0.05, 0.1, 0.2]}},
    }

    SWEEP_ID = wandb.sweep(sweep=SWEEP_CONFIG, project="bert-weight-noise-experiment")

# max length and stride specific to pretrained model
MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
MAX_LENGTH = 320
DOC_STRIDE = 128


def create_ideal_rpu_config(tile_size=512):
    """Create RPU Config with ideal conditions"""
    rpu_config = InferenceRPUConfig(
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=False,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(is_perfect=True),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=None,
    )
    return rpu_config


def create_rpu_config(modifier_noise, tile_size=512, dac_res=1024, adc_res=1024):
    """Create RPU Config emulated typical PCM Device"""
    if ARGS.wandb:
        modifier_noise = wandb.config.modifier_noise

    rpu_config = TorchInferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        modifier=WeightModifierParameter(
            rel_to_actual_wmax=True, type=WeightModifierType.ADD_NORMAL, std_dev=modifier_noise
        ),
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=ReRamCMONoiseModel(g_max=90, g_min=10,
                                                        acceptance_range=0.2,
                                                        resistor_compensator=0.0,
                                                        single_device=True),
        drift_compensation=None,
    )
    return rpu_config


def create_model(rpu_config):
    """Return Question Answering model and whether or not it was loaded from a checkpoint"""

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    if not ARGS.digital:
        model = convert_to_analog(model, rpu_config)
        model.remap_analog_weights()
        #model.load_state_dict(torch.load("/u/mvc/saved_chkpt.pth", map_location='cuda'))

    return model


# Some examples in the dataset may have contexts that exceed the maximum input length
# We can truncate the context using truncation="only_second"
def preprocess_train(dataset):
    """Preprocess the training dataset"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and padding,
    # but keep the overflows using a stride. This results
    # in one example possibly giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature, the stride being the number
    # of overlapping tokens in the overlap.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to
    # character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_dataset.pop("offset_mapping")

    # Store start and end character positions for answers in context
    tokenized_dataset["start_positions"] = []
    tokenized_dataset["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_dataset["input_ids"][i]
        cls_index = input_ids.index(TOKENIZER.cls_token_id)

        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)

        # One example can give several spans, this
        # is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = dataset["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_dataset["start_positions"].append(cls_index)
            tokenized_dataset["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Detect if the answer is out of the span
            # (in which case this feature is labeled with the CLS index).
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and
                # token_end_index to the two ends of the answer.
                # Note: we could go after the last offset
                # if the answer is the last word (edge case).
                while (
                    token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_dataset["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_dataset["end_positions"].append(token_end_index + 1)

    return tokenized_dataset


def preprocess_validation(dataset):
    """Preprocess the validation set"""
    # Some of the questions have lots of whitespace on the left,
    # which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space).
    # So we remove that
    # left whitespace
    dataset["question"] = [q.lstrip() for q in dataset["question"]]

    # Tokenize our dataset with truncation and maybe padding,
    # but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long,
    # each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_dataset = TOKENIZER(
        dataset["question"],
        dataset["context"],
        truncation="only_second",
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context,
    # we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        # Grab the sequence corresponding to that example
        # (to know what is the context and what is the question).
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1

        # One example can give several spans,
        # this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])

        # Set to None the offset_mapping that are not
        # part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_dataset["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_dataset["offset_mapping"][i])
        ]

    return tokenized_dataset


def postprocess_predictions(
    examples, features, raw_predictions, n_best_size=20, max_answer_length=30
):
    """Postprocess raw predictions"""
    features.set_format(type=features.format["type"], columns=list(features.features.keys()))
    all_start_logits, all_end_logits = raw_predictions

    # Map examples ids to index
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}

    # Create dict of lists, mapping example indices with corresponding feature indices
    features_per_example = defaultdict(list)

    for i, feature in enumerate(features):
        # For each example, take example_id, map to corresponding index
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill
    predictions = OrderedDict()

    print(
        f"Post-processing {len(examples)} example predictions "
        f"split into {len(features)} features."
    )

    # Loop over all examples
    for example_index, example in enumerate(examples):
        # Find the feature indices corresponding to the current example
        feature_indices = features_per_example[example_index]

        # Store valid answers
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            # This is what will allow us to map some the positions in our
            # logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are
                    # out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue

                    # Don't consider answers with a length
                    # that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    # Map the start token to the index of the start of that token in the context
                    # Map the end token to the index of the end of that token in the context
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]

                    # Add the answer
                    # Score is the sum of logits for the start and end position of the answer
                    # Include the text which is taken directly from the context
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char:end_char],
                        }
                    )

        # If we have valid answers, choose the best one
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Choose the best answer as the prediction for the current example
        predictions[example["id"]] = best_answer["text"]

    return predictions


def create_datasets():
    """Load the SQuAD dataset, the tokenized version, and the validation set"""
    squad = load_dataset("squad")

    # Preprocessing changes number of samples, so we need to remove some columns so
    # the data updates properly
    tokenized_data = squad.map(
        preprocess_train, batched=True, remove_columns=squad["train"].column_names
    )
    eval_data = squad["validation"].map(
        preprocess_validation, batched=True, remove_columns=squad["validation"].column_names
    )

    return squad, tokenized_data, eval_data


def create_optimizer(model):
    """Create the analog-aware optimizer"""
    optimizer = AnalogSGD(model.parameters(), lr=ARGS.learning_rate,  momentum=0.9, weight_decay=5e-4)

    optimizer.regroup_param_groups(model)

    return optimizer

class SaveModelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{int(state.epoch)}")
        kwargs["model"].save_pretrained(output_dir)
        if kwargs.get("tokenizer", None):
            kwargs["tokenizer"].save_pretrained(output_dir)
        print(f"Model saved at {output_dir}")
        return control

class EpochProgressCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n🔁 Starting epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)}")
        self.epoch_bar = tqdm(total=state.max_steps // args.num_train_epochs, desc=f"Epoch {int(state.epoch) + 1}", leave=False)
        self.epoch_loss = 0.0
        self.epoch_steps = 0

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        loss = logs.get("loss")
        if loss is not None:
            self.epoch_loss += loss
            self.epoch_steps += 1
        self.epoch_bar.update(1)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_bar.close()
        avg_loss = self.epoch_loss / self.epoch_steps if self.epoch_steps > 0 else float("nan")
        print(f"\n✅ Epoch {int(state.epoch) + 1} completed. Avg loss: {avg_loss:.4f}\n")

def make_trainer(model, optimizer, tokenized_data):
    """Create the Huggingface Trainer"""
    training_args = TrainingArguments(
        output_dir="./",
        save_strategy="no",
        evaluation_strategy="epoch",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=512,
        num_train_epochs=70,
        lr_scheduler_type='cosine',
        learning_rate = 1e-3,
        warmup_steps=100,
        no_cuda=False,
        dataloader_num_workers =32,
        report_to=[],
    )

    collator = DefaultDataCollator()

    log_dir = "logs/fit/" + ARGS.run_name
    writer = SummaryWriter(log_dir=log_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        tokenizer=TOKENIZER,
        optimizers=(optimizer, None),
        #callbacks=[TensorBoardCallback(writer)],
    )

    return trainer, writer


def do_inference(model, trainer, squad, eval_data, writer, max_inference_time=1e6, n_times=9):
    """Perform inference experiment at weight noise level specified at runtime.
    SQuAD exact match and f1 metrics are captured in Tensorboard
    """

    # Helper functions
    def predict():
        # Perform inference + evaluate metric here
        raw_predictions = trainer.predict(eval_data)
        predictions = postprocess_predictions(
            squad["validation"], eval_data, raw_predictions.predictions
        )

        # Format to list of dicts instead of a large dict
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        out_metric = metric.compute(predictions=formatted_preds, references=ground_truth)

        return out_metric["f1"], out_metric["exact_match"]

    def write_metrics(f1, exact_match, t_inference):
        # Add information to tensorboard
        writer.add_scalar("val/f1", f1, t_inference)
        writer.add_scalar("val/exact_match", exact_match, t_inference)


    model.eval()

    metric = load("squad")

    ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad["validation"]]

    t_inference_list = [1, 60*10, 3600, 3600 * 24, 3600 * 24*7, 3600 * 24 *30, 3600 * 24 *365, 3600 * 24 *365*2, 3600 * 24 *365*5, 3600 * 24 * 365 * 10] 
    #t_inference_list = [1] 

    # Get the initial metrics
    #f1, exact_match = predict()
    #write_metrics(f1, exact_match, 0.0)
    f1_times = []
    exact_match_times = []
    for t_inference in t_inference_list:
        print(t_inference)
        model.drift_analog_weights(t_inference)
        f1, exact_match = predict()
        f1_times.append(f1)
        exact_match_times.append(exact_match)
        print(f"Exact match: {exact_match: .2f}\t" f"F1: {f1: .2f}\t" f"Drift: {t_inference: .2e}")
        #write_metrics(f1, exact_match, t_inference)
    f1_times = torch.Tensor(f1_times)
    exact_match_times = torch.Tensor(exact_match_times)


def main():
    """Provide the lambda function for WandB sweep. If WandB is not used, then this
    is what is executed in the job
    """
    print("HWA:", ARGS.train_hwa)
    print("RPU config ideal:", ARGS.ideal)
    print("DIGITAL MODEL:", ARGS.digital)
    print("LOAD from checkpoint", ARGS.load)
    if ARGS.wandb:
        wandb.init()

    # Define RPU configuration and use it to create model and tokenizer
    if ARGS.ideal:
        rpu_config = create_ideal_rpu_config()
    else:
        rpu_config = create_rpu_config(modifier_noise=ARGS.noise)
    #rpu_config = StandardHWATrainingPreset()
    model = create_model(rpu_config)

    squad, tokenized_data, eval_data = create_datasets()
    #tokenized_data["train"] = tokenized_data["train"].select(range(5000))
    optimizer = create_optimizer(model)
    trainer, writer = make_trainer(model, optimizer, tokenized_data)
    if ARGS.load:
        model.load_state_dict(torch.load("/u/mvc/saved_chkpt_test_keys.pth", map_location='cuda'))

    # Do hw-aware training if in analog domain and the model isn't loaded from
    # an existing checkpoint
    if ARGS.train_hwa and not ARGS.digital and not ARGS.load:
        trainer.train()
        torch_save(model.state_dict(), "/u/mvc/bert_hwa_quantization.pth")
    #print("Starting inference")
    #do_inference(model, trainer, squad, eval_data, writer)


if ARGS.wandb:
    wandb.agent(SWEEP_ID, function=main, count=4)
else:
    main()
