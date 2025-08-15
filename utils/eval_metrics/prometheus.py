# Functions for evaluating the pipeline outputs using Prometheus
# (parameter choices adapted from https://github.com/stanford-oval/storm/tree/NAACL-2024-code-backup)
import gc, os, re, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from itertools import accumulate
from typing import Dict, List, Tuple
from utils.text_utils import (
    handle_entry,
    preprocess_text,
    smart_divide_lists,
    check_and_fix_lengths,
)
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

# For GPU memory utilization maximization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set this to remove the text cutoff from pandas
pd.set_option("display.max_colwidth", None)


def read_json(file_path: str) -> Dict:
    """
    Read a JSON file to a dictionary.

    Parameters:
    file_path - the path to the JSON file to open.

    Returns:
    a dictionary containing the parsed JSON data.
    """

    with open(file_path, "r") as file:

        return json.load(file)


def filter_non_alpha(text: str) -> str:
    """
    Removes non-alphanumeric characters.
    This was split from the rest of preprocessing to allow for sentence splitting
    (as the `punkt` tokenizer can't split without punctuations).

    Parameters:
    text - the raw text to be cleaned.

    Returns:
    the cleaned and formatted text string.
    """

    # Remove non alphanumeric characters
    text = re.sub(r"[^\w\s]", "", text)
    return text


def preprocess_tables(
    human_table: pd.DataFrame,
    generated_table: pd.DataFrame,
    dataset: str = "srg",
    method: str = "base",
) -> Dict:
    """
    Preprocesses the data for Prometheus evaluation.
    For the coverage and relevance, the sections have to be split to approximately comparable lengths.
    For the structure, the outline of the paper is used instead of the full text. Both these decisions
    are due to the context window limit of Prometheus (32,768).

    Parameters:
    human_table - the human written article table.
    generated_table - the AI generated article table.
    dataset - the dataset being used.
    method - the method of article generation used.

    Returns:
    a dictionary containing the sections and outlines. See the `output` variable for more information.
    """

    # Prepare the output container
    output = {
        "paper_ids": [],
        "human_sects": [],
        "generated_sects": [],
        "sect_lens": [],
        "human_outline": [],
        "generated_outline": [],
    }

    # Getting the data pairs
    gold_tmp, pred_tmp = [], []
    gold_papers = list(human_table.iterrows())
    for _, entry in tqdm(
        gold_papers,
        desc="Formatting results for Prometheus",
        total=len(gold_papers),
    ):

        # Getting the generated papers and texts
        paper_id = entry["paper_id"]
        pred_paper = generated_table[generated_table["paper_id"] == paper_id]

        # For the Title
        gold_title = entry["title"]
        pred_title = pred_paper["title"]

        # Check if it's still a series
        if isinstance(pred_title, pd.Series):
            pred_title = pred_title.item()

        # For the article sections
        gold_text = handle_entry(entry, "text", dataset)
        pred_text = handle_entry(pred_paper, "text", dataset)

        # For the structure
        gold_headings = handle_entry(entry, "section", dataset)
        pred_headings = handle_entry(pred_paper, "section", dataset)

        # Get the sections
        gold_sects = [
            gold_headings[i] + "\n\n" + gold_text[i] for i in range(len(gold_headings))
        ]
        pred_sects = [
            pred_headings[i] + "\n\n" + pred_text[i] for i in range(len(pred_headings))
        ]

        # Adding the abstract if applicable (if not "autosurvey", as it does not generate abstracts)
        gold_abs = entry["abstract"]
        gold_sects = [f"{gold_title}\n\n{gold_abs}\n\n"] + gold_sects
        if method != "auto":
            pred_abs = pred_paper["abstract"]
            pred_sects = [f"{pred_title}\n\n{pred_abs}\n\n"] + pred_sects

        else:
            pred_sects[0] = f"{pred_title}\n\n{pred_sects[0]}"

        # Clean the text
        gold_sects = [preprocess_text(sect) for sect in gold_sects]
        pred_sects = [preprocess_text(sect) for sect in pred_sects]

        # Splitting the lists
        gold_sects, pred_sects = smart_divide_lists(gold_sects, pred_sects)

        # Get the outlines
        gold_outline = "\n".join(gold_headings)
        pred_outline = "\n".join(pred_headings)

        # Add to results
        gold_tmp.append(gold_sects)
        pred_tmp.append(pred_sects)
        output["paper_ids"].append(paper_id)
        output["human_outline"].append(gold_outline)
        output["generated_outline"].append(pred_outline)

    # Check for texts that are too long
    gold_final, pred_final, len_list = check_and_fix_lengths(gold_tmp, pred_tmp)

    # Removing non-alphanumeric characters
    gold_final = [filter_non_alpha(text) for text in gold_final]
    pred_final = [filter_non_alpha(text) for text in pred_final]

    output["human_sects"] = gold_final
    output["generated_sects"] = pred_final
    output["sect_lens"] = len_list

    # Memory Leak Prevention
    del gold_papers, gold_tmp, pred_tmp, gold_final, pred_final, len_list
    gc.collect()
    return output


def aggregate_results(
    score_list: List[int], split_list: List[int]
) -> Tuple[float, float]:
    """
    Calculates the mean score for each article.

    Parameters:
    score_list - the score list obtained from Prometheus.
    split_list - the lengths to split on.

    Returns:
    the scores aggregated (averaged) per article.
    """

    # To check in case NoneType is returned
    if None in score_list:
        print("WARNING: NoneType found in scores. Replacing with ones...")
        score_list = [1 if item is None else item for item in score_list]

    return [
        np.mean(score_list[x - y : x])
        for x, y in zip(accumulate(split_list), split_list)
    ]


def prometheus_eval(
    human_table: pd.DataFrame,
    generated_table: pd.DataFrame,
    method: str,
    dataset: str,
    output_dir: str,
    overwrite: False,
    end_extension: str,
    setting_name: str,
    args: Namespace,
):
    """
    Main function for running the Prometheus evaluation.

    Parameters:
    human_table - the table of human written articles.
    generated_table - the table of AI written articles.
    method - the method of article generation used.
    dataset - the name of the dataset being used.
    output_dir - the directory to write the output files to.
    overwrite - Whether to overwrite existing results or not (for ROUGE and Recalls).
    end_extension - the extension to use for saving the file if needed for theSciReviewGen subsetting).
    setting_name - the name of the specific setting used (specific to only LiRA).
    args - a Namespace based on the `prometheus.yaml` file (check it for more details).
    """

    # Clearing space (in case of OOM errors)
    gc.collect()

    # Skip the process if all files are present already
    output_file = os.path.join(
        output_dir, f"{method}_{dataset}{setting_name}{end_extension}.json"
    )
    check_files = [
        os.path.isfile(re.sub(".json", f"_{i+1}.json", output_file))
        for i in range(args.n_reviews)
    ]
    if all(check_files):
        return

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model components (everything has to be the same length to work)
    content = preprocess_tables(human_table, generated_table, dataset, method=method)

    # The instruction was adjusted as before it was "Wikipedia editor".
    instructions_cr = [
        "You are a scientific journal editor examining a literature review article. Your task is reviewing the below article section(s)."
    ] * len(content["human_sects"])
    instructions_s = [
        "You are a scientific journal editor examining a literature review article. Your task is reviewing the below article outline."
    ] * len(content["human_outline"])

    rubric_data = read_json(args.rubric_file)
    score_rubric_c = SCORE_RUBRIC_TEMPLATE.format(**rubric_data[0])
    score_rubric_s = SCORE_RUBRIC_TEMPLATE.format(**rubric_data[1])
    score_rubric_r = SCORE_RUBRIC_TEMPLATE.format(**rubric_data[2])

    # Override the configurations
    hf_overrides = {
        "disable_sample": not args.disable_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }

    # Load the Model and Judge
    model = VLLM(
        model=args.model,
        quantization=args.quantization,
        dtype="bfloat16",
        hf_overrides=hf_overrides,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    # Run parallel evaluation for each review pass
    for i in range(args.n_reviews):

        print(f"[PASS {i+1}/{args.n_reviews}] Reviewing {method} articles...")
        outfile = re.sub(".json", f"_{i+1}.json", output_file)
        if os.path.isfile(outfile) and not overwrite:
            print(f"Reviews already done for PASS {i+1}. Skipping...")
            continue

        c_resps, c_scores = judge.absolute_grade(
            instructions=instructions_cr,
            responses=content["generated_sects"],
            rubric=score_rubric_c,
            reference_answers=content["human_sects"],
        )

        s_resps, s_scores = judge.absolute_grade(
            instructions=instructions_s,
            responses=content["generated_outline"],
            rubric=score_rubric_s,
            reference_answers=content["human_outline"],
        )

        r_resps, r_scores = judge.absolute_grade(
            instructions=instructions_cr,
            responses=content["generated_sects"],
            rubric=score_rubric_r,
            reference_answers=content["human_sects"],
        )

        # Get aggregated scores for coverage and relevance
        coverage_aggregated = aggregate_results(c_scores, content["sect_lens"])
        relevance_aggregated = aggregate_results(r_scores, content["sect_lens"])

        # Saving
        # NOTE: Response aggregation is not performed, so only a sample is taken
        coverage_full = {
            content["paper_ids"][i]: (c_resps[sl], coverage_aggregated[i])
            for i, sl in enumerate(content["sect_lens"])
        }
        structure_full = {
            content["paper_ids"][i]: (s_resps[i], s_scores[i])
            for i in range(len(s_resps))
        }
        relevance_full = {
            content["paper_ids"][i]: (r_resps[sl], relevance_aggregated[i])
            for i, sl in enumerate(content["sect_lens"])
        }
        resps = {
            "Coverage": coverage_full,
            "Structure": structure_full,
            "Relevance": relevance_full,
        }

        print(f"Saving responses to `{outfile}`...")
        with open(outfile, "w+") as f:

            json.dump(resps, f, indent=2)

    return
