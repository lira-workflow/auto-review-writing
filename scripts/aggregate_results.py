import sys

sys.path.append(".")

import gc, os, re, json, hydra
import numpy as np
import pandas as pd
from argparse import Namespace
from omegaconf import DictConfig
from typing import Dict, Optional
from collections import defaultdict
from utils.misc import load_json, save_to_json


# Set this to remove the text cutoff from pandas
pd.set_option("display.max_colwidth", None)


def create_per_subj(df: pd.DataFrame) -> Dict:
    """
    Creates a dictionary mapping from each subject area to the corresponding paper IDs.

    Parameters:
    df - the dataframe containing the metadata of the papers (mainly the subject areas).

    Returns:
    a dictionary in the following format:
    {[SUBJECT_AREA]: [LIST_OF_IDS]}
    """

    out = {}
    for _, row in df.iterrows():

        pii = row["PII"]
        subj_list = eval(row["subjareas"])
        for subj in subj_list:

            if subj not in out:
                out[subj] = [pii]

            else:
                out[subj].append(pii)

    return out


def compile_standard(
    dir: str = "results", subj_dict: Optional[Dict] = None, rounding: int = 3
) -> pd.DataFrame:
    """
    Compiles the results from the ROUGE and Recall JSON files into a table.

    Parameters:
    dir - the directory containing the ROUGE and Recall result JSON files.
    subj_dict - the dictionary of subject areas and the articles (only for if aggregation per subject is desired).
    rounding - the rounding to use for result aggregation.

    Returns:
    the table of results, where each cell contains the mean and standard deviation as a tuple.
    """

    methods = None
    results = defaultdict(float)
    for file in os.listdir(dir):

        # Loading the (correct) files
        if not file.endswith(".json"):
            continue

        fullpath = os.path.join(dir, file)
        with open(fullpath, "r") as f:

            data = json.load(f)

        components = file.strip(".json")
        samples = list(data.keys())
        methods = list(data[samples[0]].keys())  # Same keys for every entry

        tmp = {method: [] for method in methods}
        # Pull out the samples
        for sample in samples:

            for method in methods:

                tmp[method].append(data[sample][method])

        # Aggregate the results
        results[components] = {method: [] for method in methods}
        for method in methods:

            results[components][method] = (
                np.round(np.mean(tmp[method]), rounding),
                np.round(np.std(tmp[method]), rounding),
            )

    if methods is None:
        return None

    # Conversion to pandas dataframe
    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.set_axis(methods, axis=1)

    # Memory Leak Prevention
    del tmp, results
    gc.collect()
    return df


# For calculating the scaled recall and F1 score for citation quality
def scale_recall_f1(
    df: pd.DataFrame,
    citation_dir: str = "results",
    scaling_param: float = 0.01,
    rounding: int = 3,
) -> pd.DataFrame:
    """
    Scales the recall based on an exponential factor (to penalize articles with less claims) and
    uses it (and the citation precision) to calculate the citation F1 score.

    Parameters:
    df - the dataframe to process.
    citation_dir - the location of the citation quality results.
    scaling_param - the scaling factor for the exponential penalty.
    rounding - the rounding to use for result calculation.

    Returns:
    the DataFrame with the scaled recall and citation F1 scores as new columns.
    """

    # First check if the precision and recall are present
    assert (
        "recall" in df.columns and "precision" in df.columns
    ), "Please run the citation quality evaluation first!"

    # Get the number of claims per setting
    n_claims_dict = {}
    n_claims_dir = os.path.join(citation_dir, "n_claims")
    for file in os.listdir(n_claims_dir):

        n_claims_file = os.path.join(n_claims_dir, file)
        temp_dict = load_json(n_claims_file)

        setting = re.sub(".json", "", file)
        n_claims_dict[setting] = np.mean(list(temp_dict.values()))

    # Add the new scores
    df[["recall scaled", "citation F1"]] = None, None
    for idx, entry in df.iterrows():

        # Scale the recall
        n_claims = n_claims_dict[entry["Name"]]
        recall_scaled = entry["recall"][0] * (
            1 - np.power(np.e, (-scaling_param * n_claims))
        )

        # Calculate the F1 score
        precision = entry["precision"][0]
        f1 = (2 * precision * recall_scaled) / (precision + recall_scaled)

        # Put the values in the dataframe
        df.at[idx, "recall scaled"] = round(recall_scaled, rounding)
        df.at[idx, "citation F1"] = round(f1, rounding)

    # Re-sorting the columns
    if "Coverage" in df.columns:
        colnames = list(df.columns)
        sorting = colnames[:-5] + colnames[-2:] + colnames[-5:-2]
        df = df[sorting]

    return df


def aggregate_per_file_prom(
    filepath: str, subj_dict: Dict, dir: str, rounding: int = 3
) -> Dict:
    """
    Aggregates the results for one Prometheus reviewer file.

    Parameters:
    filepath - the filename of the Prometheus result JSON.
    subj_dict - the dictionary of subject areas and the articles.
    dir - the directory containing the result JSON files.
    rounding - the rounding to use for result formatting.

    Returns:
    a dictionary containing the aggregated results for the file across all samples.
    """

    with open(filepath, "r") as f:

        data = json.load(f)

    # Go over the results
    metrics = list(data.keys())
    samples = list(data[metrics[0]].keys())
    tmp = {metric: [] for metric in metrics}
    for metric in metrics:

        for sample in samples:

            score = float(data[metric][sample][1])
            tmp[metric].append(score)

    # Then aggregate over them
    out = {metric: [] for metric in metrics}
    for metric in metrics:

        out[metric] = (
            np.round(np.mean(tmp[metric]), rounding),
            np.round(np.std(tmp[metric]), rounding),
        )

    # Memory Leak Prevention
    del tmp
    gc.collect()
    return out


def compile_prometheus(dir: str, subj_dict: Dict, rounding: int = 3) -> pd.DataFrame:
    """
    Compiles the results from the Prometheus evaluations.

    Parameters:
    dir - the directory containing the Prometheus result JSON files.
    subj_dict - the dictionary of subject areas and the articles.
    rounding - the rounding to use for result formatting.

    Returns:
    the table of results, where each cell contains the mean and standard deviation as a tuple.
    """

    # Group by method
    results = defaultdict(list)
    for file in os.listdir(dir):

        if not file.endswith(".json"):
            continue

        fullpath = os.path.join(dir, file)
        components = re.sub(r"\_\d+\.json", "", file)
        tmp = aggregate_per_file_prom(fullpath, subj_dict, dir)
        results[components].append(tmp)

    # Aggregation
    for key in results.keys():

        averages = {}
        metrics = list(results[key][0].keys())
        for metric in metrics:

            means = [e[metric][0] for e in results[key]]
            stds = [e[metric][1] for e in results[key]]
            tmp = (np.mean(means), np.mean(stds))
            tmp = (round(float(tmp[0]), rounding), round(float(tmp[1]), rounding))
            averages[metric] = tmp

        results[key] = averages

    # Conversion to a pandas dataframe
    output = pd.DataFrame(
        {
            outer_key: {inner_key: value for inner_key, value in inner_dict.items()}
            for outer_key, inner_dict in results.items()
        }
    )
    output = output.T
    return output


def compile_results(results_dir: str, rounding: int, subj_dict: Optional[Dict]):
    """
    Compiles the results for all metrics into a table.
    This is then saved as a `.csv` file.

    Parameters:
    results_dir - the directory containing the result files.
    rounding - the rounding to use for result aggregation.
    subj_dict - the dictionary of subject areas and the articles.
    """

    rouge_dir = os.path.join(results_dir, "rouge")
    recall_dir = os.path.join(results_dir, "recall")
    citation_dir = os.path.join(results_dir, "citation")
    prometheus_dir = os.path.join(results_dir, "prometheus")

    # Check the directories first
    dirs = [rouge_dir, recall_dir, citation_dir, prometheus_dir]
    if any([not os.path.isdir(dir) for dir in dirs]):
        raise NotADirectoryError(
            "Please run the experiments before running this script!"
        )

    # First handle the standard results
    main_df = ""
    for dir in dirs[:3]:

        # For the main results
        if isinstance(main_df, str):
            main_df = compile_standard(dir, subj_dict=subj_dict, rounding=rounding)

        else:
            new_df = compile_standard(dir, subj_dict=subj_dict, rounding=rounding)
            if new_df is not None:
                main_df = pd.merge(main_df, new_df, left_index=True, right_index=True)

    # Then handle the Prometheus results
    prometheus = compile_prometheus(prometheus_dir, subj_dict, rounding)

    if not isinstance(main_df, str):
        final_df = pd.merge(main_df, prometheus, left_index=True, right_index=True)

        # Sorting by components
        final_df.index.name = "Name"
        final_df = final_df.reset_index()

        # Calculating the scaled citation recall and F1-score
        final_df = scale_recall_f1(final_df, citation_dir)

        final_df["Name"] = final_df["Name"].apply(
            lambda x: re.sub(r"\_subset\_\d+\_\d+", "", x)
        )  # Remove the subset for sorting

        final_df = final_df.sort_index(ascending=True)  # Sorting
        final_df.to_csv(os.path.join(results_dir, "final_results.csv"), index=False)

        # Memory Leak Prevention
        del final_df, prometheus
        gc.collect()


@hydra.main(
    version_base=None, config_path="../configs", config_name="aggregate_results"
)
def main(cfg: DictConfig):

    # Loading the config
    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)
    meta_dict = None
    compile_results(args.result_dir, args.rounding, meta_dict)
    print("All results aggregated!\n")


if __name__ == "__main__":

    main()
