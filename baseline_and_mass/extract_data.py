import gc, os, ast
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from argparse import Namespace
from .utils.file_utils import save_ref_as_txt


def process_entry_srg(
    title_dict: Dict,
    abs_dict: Dict,
    count: int,
) -> List:
    """
    Processes data from a SciReviewGen row and metadata to generate MASS-structured references.

    Parameters:
    title_dict - the dictionary of titles from the papers cited.
    abs_dict - the dictionary of abstracts from the papers cited.
    count - the reference number.

    Returns:
    all formatted references as a list.
    """

    references = []

    # De-duplicate entries
    corpus_ids = list(set(list(title_dict.keys())))
    for corpus_id in corpus_ids:

        # Get the abstract
        abs = abs_dict[corpus_id]

        title = title_dict[corpus_id]
        reference = {"num": f"[{count}]", "title": title, "abstract": abs}
        references.append(reference)

        # Update the counter
        count += 1

    return references, count


def process_all_entries_srg(data_file: str, args: Namespace):
    """
    Extracts all references and saves the files in `.txt` format.

    Parameters:
    data_file - the name of the SciReviewGen subset dataframe to load.
    args - a Namespace containing the following variables which get read:

    processed_dir - the target directory to save to.
    """

    # Loading the `data_file` and metadata
    df = pd.read_csv(data_file)

    # Ensuring the target directory exists
    os.makedirs(args.processed_dir, exist_ok=True)

    # Processing each file
    for _, entry in tqdm(
        df.iterrows(),
        desc="Extracting from Data Table",
        total=len(list(df.iterrows())),
        unit="file(s)",
    ):

        # Check first if the target file already exists
        paper_id = entry["paper_id"]
        out_name = os.path.split(data_file)[1]
        out_name = str(paper_id) + "_" + os.path.splitext(out_name)[0]
        output_file = os.path.join(args.processed_dir, out_name + "_references.txt")

        if os.path.isfile(output_file):
            continue

        # Get all the references and numberings
        count = 1
        references_full = []
        title_dicts = ast.literal_eval(entry["bib_titles"])
        abs_dicts = ast.literal_eval(entry["bib_abstracts"])
        for title_dict, abs_dict in zip(title_dicts, abs_dicts):

            references, count = process_entry_srg(title_dict, abs_dict, count)
            references_full.extend(references)

        # Saving
        save_ref_as_txt(references_full, output_file)
        del references_full

    # Memory Leak Prevention
    del df, entry
    gc.collect()
