import gc, os, hydra
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from hydra import compose, initialize
from typing import Dict, List, Tuple, Union
from .misc import load_json, read_file, save_to_json
from .create_db import prepare_dataset, create_database_fulltext
from utils.eval_metrics.rouge import rouge_eval
from utils.eval_metrics.recall import calc_recall_full
from utils.eval_metrics.citation_quality import citation_quality
from utils.text_utils import handle_entry, format_text_from_table


def fetch_tables(
    human_rev_path: str, generated_rev_path: str, dataset: str = "srg"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the tables containing the articles.

    Parameters:
    human_rev_path - the filename containing the human-written data.
    generated_rev_path - the location of the generated articles.
    dataset - the dataset being used.

    Returns:
    the article tables.
    """

    if dataset == "srg":
        human_reviews = pd.read_csv(human_rev_path, index_col=False)

    else:
        raise ValueError(f"Dataset `{dataset}` not supported!")

    generated_reviews = pd.read_csv(generated_rev_path, index_col=False)
    return human_reviews, generated_reviews


def format_data(
    filename: str, args: Namespace
) -> Tuple[List, List, List, List[Tuple], List[Tuple]]:
    """
    Formats the data for evaluation.

    Parameters:
    filename - the filename containing the human-written data.
    args - a Namespace containing the following variables which get read:

    n_samples - the number of samples used for selection.
    seed - the seed used for sample selection.
    method - the method of article generation used (mainly for determining how to format the articles).
    dataset - the dataset used for testing (used for file saving and some processing).
    final_papers_file - the location of the generated articles.

    Returns:
    the following data elements in order:
    the list of paper IDs
    the list of human-written articles
    the list of system-generated articles
    the fulltext pairs in this order for each tuple: (human, machine)
    the heading pairs in this order for each tuple: (human, machine)
    """

    human_reviews, generated_reviews = fetch_tables(
        filename, args.final_papers_file, args.dataset
    )

    # Getting the data pairs
    gold_papers = list(human_reviews.iterrows())
    paper_ids, gold_list, pred_list = [], [], []
    text_pairs, heading_pairs = [], []
    for _, entry in tqdm(
        gold_papers,
        desc="Formatting results for evaluation",
        total=len(gold_papers),
    ):

        # Getting the generated papers and texts
        paper_id = entry["paper_id"]
        pred_paper = []

        # Ensure the paper is retrieved (sometimes it doesn't return anything)
        count = 1
        while len(pred_paper) <= 0:
            pred_paper = generated_reviews[generated_reviews["paper_id"] == paper_id]
            if count >= 5:
                break

            count += 1

        # Extraction
        base_file = (
            f"{paper_id}_subset_{args.n_samples}_{args.seed}.txt"
            if args.dataset == "srg"
            else f"{paper_id}.txt"
        )

        gold_text = format_text_from_table(entry, args.dataset, method=args.method)
        pred_file = os.path.join(args.final_papers_dir, base_file)

        # If the paper has not been generated yet, skip it for now
        if not os.path.isfile(pred_file):
            print(f"`{pred_file}` not found. Skipping...")
            continue

        pred_text = read_file(pred_file)

        gold_headings = handle_entry(entry, "section", args.dataset)
        pred_headings = handle_entry(pred_paper, "section", args.dataset)

        paper_ids.append(paper_id)
        text_pairs.append((gold_text, pred_text))
        heading_pairs.append((gold_headings, pred_headings))

        # For the ROUGE scoring
        gold_list.append(gold_text)
        pred_list.append(pred_text)

    # Memory Leak Prevention
    del (
        gold_text,
        pred_text,
        gold_papers,
        human_reviews,
        gold_headings,
        pred_headings,
        generated_reviews,
    )
    gc.collect()
    return paper_ids, gold_list, pred_list, text_pairs, heading_pairs


def compile_and_save(paper_ids: List[Union[int, str]], results: Dict, out_file: str):
    """
    Compiles the results for a metric per article and then saves it.
    Do note that the results and IDs should all be in order.

    Parameters:
    paper_ids - the list of paper IDs.
    results - the dictionary containing results in the following format: {[METRIC]: [LIST_OF_RESULTS]}.
    out_file - the output name for saving the file.
    """

    final_dict = {}
    for idx in range(len(paper_ids)):

        final_dict[paper_ids[idx]] = {k: v[idx] for k, v in results.items()}

    save_to_json(final_dict, out_file)
    print(f"Created `{out_file}`\n")

    # Memory Leak Prevention
    del final_dict
    gc.collect()


def eval_method(filename: str, end_extension: str, args: Namespace):
    """
    Evaluates the results of the article generation compared to the baseline reviews.
    These are then saved to a JSON file in the specified directory.

    Parameters:
    filename - the location of the human written articles.
    end_extension - the extension to use for saving the file if needed for theSciReviewGen subsetting).
    args - a Namespace containing the following variables which get read:

    n_samples - the number of samples used for selection.
    seed - the seed used for sample selection.
    dataset - the dataset used for testing (used for file saving and some processing).
    final_papers_dir - the location where the generated article `.txt`'s are saved.
    eval_result_dir - the directory to where the results will be saved.
    method - the method of article generation used (here it's used for file saving).
    final_papers_file - the location of the generated articles.
    rounding - the rounding to use for result formatting.
    n_jobs - the number of workers to use for parallelization.
    do_rouge - whether not to perform evaluation using ROUGE scores.
    do_recall - whether not to perform evaluation using heading/article recall scores.
    do_citation - whether not to perform evaluation on the citation precision and recall.
    do_prometheus - whether not to perform Prometheus LLM evaluation.
    overwrite_results - Whether to overwrite existing results or not (for ROUGE and Recalls).
    data_dir - the location of the dataset files.
    db_dir - the location of the database files (for citation quality evaluation).
    embedding_model - the embedding model used for encoding (for citation quality evaluation).
    max_tokens - the maximum number of tokens the LLM can handle.
    setting_name - the name of the specific setting used (specific to only LiRA).
    editor_model - the editor model used (for in case we want to do the `_noedit` setting).
    """

    print("Starting evaluation. This may take a while...")

    # Setup the output directory
    os.makedirs(args.eval_result_dir, exist_ok=True)

    # Setting the `setting_name` (handling both present cases)
    if args.method == "lira":
        setting_name = args.setting_name

    else:
        setting_name = ""

    # For the editor setting
    setting_name += "_noedit" if args.editor_model == "none" else ""

    # Formatting the data
    if args.do_rouge or args.do_recall or args.do_citation:
        paper_ids, gold_list, pred_list, text_pairs, heading_pairs = format_data(
            filename, args
        )
        gc.collect()

    # Calculate the ROUGE scores if desired ----------------------------------------
    out_folder_rouge = os.path.join(args.eval_result_dir, "rouge")
    os.makedirs(out_folder_rouge, exist_ok=True)
    out_file_rouge = os.path.join(
        out_folder_rouge,
        f"{args.method}_{args.dataset}{setting_name}{end_extension}.json",
    )
    temp_folder = os.path.join(out_folder_rouge, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    if args.do_rouge and (not os.path.isfile(out_file_rouge) or args.overwrite_results):
        rouge_dict = rouge_eval(
            gold_list,
            pred_list,
            n_jobs=args.n_jobs,
            rounding=args.rounding,
            temp_folder=temp_folder,
            temp_file=f"{args.method}_{args.dataset}{end_extension}.json",
        )
        compile_and_save(paper_ids, rouge_dict, out_file_rouge)

        # Memory Leak Prevention
        del gold_list, pred_list
        gc.collect()

    else:
        print("Skipped ROUGE calculation")

    # Calculate the recall scores if desired ---------------------------------------
    out_folder_recall = os.path.join(args.eval_result_dir, "recall")
    os.makedirs(out_folder_recall, exist_ok=True)
    out_file_recall = os.path.join(
        out_folder_recall,
        f"{args.method}_{args.dataset}{setting_name}{end_extension}.json",
    )
    if args.do_recall and (
        not os.path.isfile(out_file_recall) or args.overwrite_results
    ):
        recall_dict = calc_recall_full(text_pairs, heading_pairs, args=args)
        compile_and_save(paper_ids, recall_dict, out_file_recall)

        # Memory Leak Prevention
        del text_pairs, heading_pairs
        gc.collect()

    else:
        print("Skipped Recall calculation")

    # Calculate the citation quality scores if desired -----------------------------
    out_folder_citation = os.path.join(args.eval_result_dir, "citation")
    os.makedirs(out_folder_citation, exist_ok=True)
    out_file_citation = os.path.join(
        out_folder_citation,
        f"{args.method}_{args.dataset}{setting_name}{end_extension}.json",
    )

    if args.do_citation and (
        not os.path.isfile(out_file_citation) or args.overwrite_results
    ):
        # First prepare the database if not made already
        _ = prepare_dataset(args)
        ref_db_path = os.path.join(args.db_dir, "refs.json")
        ref_data = load_json(ref_db_path)
        create_database_fulltext(ref_data, args)
        generated_reviews = pd.read_csv(args.final_papers_file)
        citation_dict = citation_quality(
            paper_ids,
            generated_reviews,
            args.method,
            args.dataset,
            end_extension=end_extension,
            setting_name=setting_name,
            args=args,
        )
        compile_and_save(paper_ids, citation_dict, out_file_citation)
        # Memory Leak Prevention
        del ref_data, generated_reviews
        gc.collect()

    else:
        print("Skipped Citation Quality evaluation")

    # Get the Prometheus reviews if desired ----------------------------------------
    if args.do_prometheus:
        # Import here to prevent issues in case Prometheus results won't/can't be run
        from utils.eval_metrics.prometheus import prometheus_eval

        # Get the data for Prometheus formatting before resetting the config
        human_reviews, generated_reviews = fetch_tables(
            filename, args.final_papers_file, args.dataset
        )

        # Get the method as the config will be reset
        method = args.method

        # Reset the Hydra configuration as it's already called in the main function
        hydra.core.global_hydra.GlobalHydra.instance().clear()

        # Fetch the configuration
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose(config_name="prometheus")["base"]
            prom_args = Namespace(**cfg)

        output_dir_prom = os.path.join(args.eval_result_dir, "prometheus")
        _ = prometheus_eval(
            human_reviews,
            generated_reviews,
            method=method,
            dataset=args.dataset,
            output_dir=output_dir_prom,
            overwrite=args.overwrite_results,
            end_extension=end_extension,
            setting_name=setting_name,
            args=prom_args,
        )

    else:
        print("Skipped Prometheus evaluation")

    print("\nEvaluation done!\n")
