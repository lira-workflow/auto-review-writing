import os, sys, json

sys.path.append(".")

import gc, evaluate
import numpy as np
from tqdm import tqdm
from absl import logging
from joblib import Parallel, delayed
from typing import Dict, List, Generator
from utils.pbar_utils import tqdm_joblib
from utils.eval_metrics.eval_constants import ROUNDING

# Disable absl notifications
logging.set_verbosity(logging.ERROR)


def combine_dict_list(
    dict_list: List[Dict], average: bool = False, rounding: int = ROUNDING
) -> Dict:
    """
    Combines a list of dictionaries (assuming they all have the same keys) into a single dictionary of lists.

    Parameters:
    dict_list - the list of dictionaries to combine.
    average - whether to average the results or not.
    rounding - the rounding to use in case averaging is used.

    Returns:
    a dictionary of all the lists combined into one.
    """

    keys = list(dict_list[0].keys())
    final = {key: [] for key in keys}
    for entry in dict_list:

        for key in keys:

            final[key].append(entry[key])

    if average:
        final = {key: np.round(np.mean([final[key]]), rounding) for key in keys}

    return final


def to_generator(
    text_list: List[str], chunk_size: int = 10
) -> Generator[List[str], None, None]:
    """
    Converts a list of strings into a generator.
    Chunking is also supported.

    Parameters:
    text_list - the list of strings/text to convert.
    chunk_size - the size of the chunks returned.

    Returns:
    a generator version of the list.
    """
    for i in range(0, len(text_list), chunk_size):

        yield text_list[i : i + chunk_size]


def process_single(
    pred_answer: List[str],
    gold_answer: List[str],
    temp_dict: Dict,
    idx: int,
    temp_file: str,
) -> Dict:
    """
    Processes a single pair of predictions and gold answers to compute ROUGE scores.

    Parameters:
    pred_answer - a predicted answer in the form of a list.
    gold_answer - a golden reference answer in the form of a list.
    temp_dict - the dictionary to store the temporary results.
    idx - the entry index (for checking if the processing needs to be done or not).

    Returns:
    a dictionary containing ROUGE scores for the input pair.
    """

    if str(idx) in temp_dict:
        return temp_dict[str(idx)]

    else:
        # Load the ROUGE calculator instance
        rouge = evaluate.load("rouge")
        scores = rouge.compute(
            predictions=pred_answer,
            references=gold_answer,
            rouge_types=[
                "rouge1",
                "rouge2",
                "rougeL",
            ],  # Skip the rougeLsum calculation
            use_aggregator=True,
        )
        final = {
            "ROUGE1": scores["rouge1"],
            "ROUGE2": scores["rouge2"],
            "ROUGEL": scores["rougeL"],
        }
        temp_dict[str(idx)] = final
        with open(temp_file, "w+") as f:

            json.dump(temp_dict, f, indent=2)

        # Memory Leak Prevention
        del rouge, scores
        gc.collect()
        return final


def rouge_eval(
    gold_answers: List[str],
    pred_answers: List[str],
    rounding: int = ROUNDING,
    n_jobs: int = -1,
    temp_folder: str = "temp",
    temp_file: str = "temp.json",
) -> Dict:
    """
    Calculates the ROUGE-1, 2, and L scores between a list of gold and predicted answers using parallel processing.
    If the predicted text is too long, then an approximation of the score is used instead which is based on splitting
    the text into smaller parts first (see `approx_rouge` for more details).

    Parameters:
    gold_answers - the golden answers to use as reference for ROUGE.
    pred_answers - the answers outputted by the automated pipeline.
    chunk_size - size of chunks to process in parallel.
    rounding - the rounding for adjusting the final result output.
    n_jobs - the number of workers to use for parallel processing.
    temp - temporary folder for saving the intermediate results.

    Returns:
    a dictionary containing the ROUGE scores for all articles.
    Specifically, it has the following format: {"ROUGE[1/2/L]": [LIST_OF_RESULTS]}

    For further reference you can refer to: https://aclanthology.org/W04-1013/
    """

    # Prepare the folder
    os.makedirs(temp_folder, exist_ok=True)
    temp_file_full = os.path.join(temp_folder, temp_file)

    # Pre-check: ensuring lists are the same length
    len_gold = len(gold_answers)
    assert len_gold == len(
        pred_answers
    ), f"The number of entries do not match between lists: {len(gold_answers)} and {len(pred_answers)}!"

    # Use generators to reduce memory usage
    gold_gen = to_generator(gold_answers, 1)
    pred_gen = to_generator(pred_answers, 1)

    # Memory Leak Prevention
    del gold_answers, pred_answers
    gc.collect()

    # Temporary storage in case something happens with the process
    # (the `!= full` condition is for the unit test.)
    temp_dict = {}
    if os.path.isfile(temp_file_full) and temp_file != "full":
        with open(temp_file_full, "r") as f:

            temp_dict = json.load(f)

    if not len(temp_dict) == len_gold:
        # Process chunks in parallel
        with tqdm_joblib(
            tqdm(
                desc="Calculating ROUGE scores",
                unit="answer(s)",
                total=len_gold,
            )
        ):
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(process_single)(
                    pred_answer, gold_answer, temp_dict, idx, temp_file_full
                )
                for (pred_answer, gold_answer, idx) in zip(
                    pred_gen, gold_gen, range(len_gold)
                )
            )

    # Memory Leak Prevention
    del gold_gen, pred_gen
    gc.collect()

    # Aggregate results
    final = combine_dict_list(results, rounding=rounding)

    # Round the outputs
    for key in final.keys():

        final[key] = [round(res, rounding) for res in final[key]]

    return final
