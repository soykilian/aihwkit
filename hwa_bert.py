
import aihwkit
from aihwkit.simulator.presets.inference import StandardHWATrainingPreset
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from aihwkit.optim import AnalogAdam
from transformers import get_linear_schedule_with_warmup
from aihwkit.nn.conversion import convert_to_analog
import numpy as np
from collections import OrderedDict, defaultdict
from transformers import TrainerCallback
import os
from datetime import datetime
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
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
    InferenceRPUConfig,
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

def train_epoch(
    train_dataloader,
    model,
    optimizer,
    scheduler,
    current_epoch,
    logging_step_frequency,
    wandb_logging=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    early_exit_n_iters=-1,
):
    model.train()
    step = 0
    n_steps = len(train_dataloader)
    with tqdm(train_dataloader, desc="Iteration") as tepoch:
        for batch in tepoch:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad(set_to_none=True)
            if step % logging_step_frequency == 0:
                tepoch.set_postfix(loss=loss.item(), lr=scheduler.get_lr()[0])
                if wandb_logging:
                    wandb.log(
                        {
                            "step": n_steps * current_epoch + step,
                            "training_loss": loss.item(),
                        }
                    )

            if step == early_exit_n_iters:
              break

            step += 1
def evaluate(
    model,
    tokenizer,
    examples,
    features,
    eval_dataloader,
    cache_dir,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    early_exit_n_iters=-1,
):
    all_results = []
    batch_idx = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }
            feature_indices = batch[3]
            outputs = model(**inputs)

        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [to_list(output[i]) for output in outputs.to_tuple()]
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]
                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

        if batch_idx == early_exit_n_iters:
          break

        batch_idx += 1

    output_prediction_file = os.path.join(cache_dir, "predictions.json")
    output_nbest_file = os.path.join(cache_dir, "nbest_predictions.json")
    predictions = compute_predictions_logits(
        examples[0:early_exit_n_iters],
        features[0:early_exit_n_iters],
        all_results[0:early_exit_n_iters],
        20,
        30,
        True,
        output_prediction_file,
        output_nbest_file,
        None,
        False,
        False,
        0.0,
        tokenizer,
    )
    results = squad_evaluate(examples[0:early_exit_n_iters], predictions)
    return results
MODEL_NAME = "csarron/bert-base-uncased-squad-v1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LENGTH = 320
DOC_STRIDE = 128

def preprocess_train(dataset):
    dataset["question"] = [q.lstrip() for q in dataset["question"]]
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
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_dataset.pop("offset_mapping")
    tokenized_dataset["start_positions"] = []
    tokenized_dataset["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_dataset["input_ids"][i]
        cls_index = input_ids.index(TOKENIZER.cls_token_id)
        sequence_ids = tokenized_dataset.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = dataset["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_dataset["start_positions"].append(cls_index)
            tokenized_dataset["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_dataset["start_positions"].append(cls_index)
                tokenized_dataset["end_positions"].append(cls_index)
            else:
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
    dataset["question"] = [q.lstrip() for q in dataset["question"]]
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
    sample_mapping = tokenized_dataset.pop("overflow_to_sample_mapping")

    tokenized_dataset["example_id"] = []

    for i in range(len(tokenized_dataset["input_ids"])):
        sequence_ids = tokenized_dataset.sequence_ids(i)
        context_index = 1
        sample_index = sample_mapping[i]
        tokenized_dataset["example_id"].append(dataset["id"][sample_index])
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

def create_model(rpu_config):
    """Return Question Answering model and whether or not it was loaded from a checkpoint"""

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model = convert_to_analog(model, rpu_config)
    #model.remap_analog_weights()
    #model.load_state_dict(torch.load("/u/mvc/saved_chkpt.pth", map_location='cuda'))
    return model
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

        print(f"Exact match: {exact_match: .2f}\t" f"F1: {f1: .2f}\t" f"Drift: {t_inference: .2e}") 
    metric = load("squad")

    ground_truth = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad["validation"]]
    f1_times = []
    exact_match_times = []
    t_inference_list = [1, 60*10, 3600, 3600 * 24]
    for t_inference in t_inference_list:
        model.drift_analog_weights(t_inference)
        f1, exact_match = predict()
        f1_times.append(f1)
        exact_match.append(f1)
        print(t_inference)
        print(f1)
    #torch.save(f1_times, 'f1_inference.pth')
def main():
    n_reps = 5
    rpu_config = StandardHWATrainingPreset()
    max_seq_length = 320# wandb.config.max_seq_length
    logging_step_frequency = 5#wandb.config.logging_step_frequency
    batch_size_train = 16#wandb.config.batch_size_train
    batch_size_eval = 256#wandb.config.batch_size_eval
    num_training_epochs = 50#wandb.config.num_training_epochs
    learning_rate = 10 ** -4
    model_id = "csarron/mobilebert-uncased-squad-v1"
    model = create_model(rpu_config)
    print("Loading and parsing training features.")
    squad, tokenized_data, eval_data = create_datasets()
    tokenized_data.set_format(type="torch")
    train_dataset = tokenized_data["train"]
    val_dataset = tokenized_data["validation"]
    print(type(train_dataset))
    print(type(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle= True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle= False)
    optimizer = AnalogAdam(
        model.parameters(), weight_decay = 0.0005, lr=learning_rate,
    )
    t_total = len(train_loader) // num_training_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )
    model.zero_grad()
    for current_epoch in range(0, num_training_epochs):
        print("Training epoch: ", current_epoch)
        model.train()
        train_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            current_epoch=current_epoch,
            logging_step_frequency=logging_step_frequency,
            wandb_logging=False,
            early_exit_n_iters=-1,
        )
        with torch.no_grad():
            model.eval()
            do_inference(model, trainer, squad, eval_data, writer)

if __name__ == "__main__":
    main()