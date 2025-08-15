import sys

sys.path.append(".")

import gc, os, re, ast, csv, gdown, hydra, tarfile
import pandas as pd
from tqdm import tqdm
from typing import List, Union
from argparse import Namespace
from omegaconf import DictConfig
from utils.misc import chunk_list
from joblib import Parallel, delayed
from utils.pbar_utils import tqdm_joblib
from semanticscholar import SemanticScholar


# Setup SemanticScholar fetcher object
sch = SemanticScholar()

# Data location
LOCATION = "data/scireviewgen/scireviewgen.pkl"
METADATA_LOCATION = "data/scireviewgen/srg_metadata.csv"


# Download and extract data from SciReviewGen files
def download_srg_files():
    """
    Downloads the SciReviewGen files needed for experimentation.
    """

    # Setting variables (currently summarization is unused)
    filename = f"temp.tar.gz"

    # Donwloading the file
    print(f"Handling `original_survey` data...")
    gdown.download(id="1MnjQ2fQ_fJjcqKvIwj2w7P6IGh4GszXH", output=filename)

    # Ensure `data` directory exists:
    os.makedirs("data", exist_ok=True)

    # Open and extract files
    print("Extracting to 'data'...")
    file = tarfile.open(filename)
    file.extractall("./data")

    # Move out of the `original_survey_df` folder
    os.rename(
        "data/original_survey_df/original_survey_df.pkl",
        LOCATION,
    )
    os.remove("data/original_survey_df")

    # Remove .tar files after processing
    os.remove(filename)

    # Formatting space print
    print()


# Remove temp_files in the folde
def clean_dir():
    """
    Removes 'temp.tar.gz' files from the main directory.
    """

    print("Removing temporary files...")
    files = [f for f in os.listdir(".") if os.path.isfile(f)]
    for f in files:

        if "temp.tar.gz" in f:
            os.remove(f)

    print("Done!\n")


def save_checkpoint(filename: str, rows: List[dict]):
    """
    Saves rows to a CSV file as a checkpoint.

    Parameters:
    filename - the name of the CSV file.
    rows - a list of dictionaries to be saved.
    """

    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as csvfile:
        fieldnames = rows[0].keys() if rows else []
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def request_metadata_chunk(chunk) -> List:
    """
    Fetches metadata for a chunk of paper IDs.

    Parameters:
    chunk - a list of paper IDs in the format "CorpusId:XXXX".

    Returns:
    a list of metadata dictionaries.
    """

    try:
        return sch.get_papers(chunk)

    except Exception as e:
        print(f"Error fetching chunk: {chunk}, error: {e}")
        return []


def request_metadata(
    corpus_ids: List[Union[int, str]], target_file: str = METADATA_LOCATION
) -> List:
    """
    Fetches paper metadata from Semantic Scholar using joblib parallel processing
    and saves intermediate results to a checkpoint file.

    Parameters:
    corpus_ids - a list of Semantic Scholar IDs.
    target_file - the path to the CSV file used for saving checkpoints.
    """

    # Converting corpus IDs to the required format
    paper_ids = [f"CorpusId:{corpus_id}" for corpus_id in corpus_ids]

    # Chunking the IDs (500 per API call limit)
    chunked_ids = chunk_list(paper_ids, 500)

    # Loading previously completed IDs from the checkpoint file
    completed_ids = set()
    if os.path.exists(target_file):
        with open(target_file, mode="r", encoding="utf-8") as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:

                completed_ids.add(row["corpusId"])

    # Filtering out completed chunks
    remaining_chunks = [
        chunk
        for chunk in chunked_ids
        if not any(cid.split(":")[1] in completed_ids for cid in chunk)
    ]

    # Only continue if any chunks remain
    metadata = []
    if len(remaining_chunks) > 0:
        # Using joblib for parallel processing
        with tqdm_joblib(
            tqdm(
                desc="Processing Chunks",
                unit="chunk(s)",
                total=len(remaining_chunks),
            )
        ):
            chunk_results = Parallel(n_jobs=-3)(
                delayed(request_metadata_chunk)(chunk) for chunk in remaining_chunks
            )

        # Process results
        for chunk_data in chunk_results:
            if chunk_data:
                metadata.extend(chunk_data)

                # Save the completed chunk to the checkpoint file
                rows_to_save = [
                    {"corpusId": paper["corpusId"], **paper} for paper in chunk_data
                ]
                save_checkpoint(target_file, rows_to_save)

    print("All metadata has been downloaded! \n")


def get_corpus_ids(df: pd.DataFrame) -> List:
    """
    Extracts the ID's of every reference entry in the SciReviewGen dataset table.

    Parameters:
    df - the pandas DataFrame of the SciReviewGen data.

    Returns:
    a list containing every corpus ID in the dataset.
    """

    # As every entry has a list of lists of dictionaries, we have to multi-iterate
    bib_dicts = df["bib_citing_sentences"]
    ref_ids = [key for entry in bib_dicts for d in entry for key in d.keys()]
    corpus_ids = list(set(ref_ids))

    return corpus_ids


@hydra.main(version_base=None, config_path="../configs", config_name="sample")
def main(cfg: DictConfig):

    # Loading the config
    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    # Check if the main dataset file exists already
    if not os.path.isfile(LOCATION):
        download_srg_files()

        # Check for temp files in current directory
        clean_dir()

    else:
        print("Dataset already downloaded!")

    # Get the metadata and save it to another table
    df = pd.read_pickle(LOCATION)
    print(f"Generating references table for {args.n_samples} sample(s)...")

    # Sample based on if there are enough references (minimum of 10)
    df["n_bibs_eval"] = df["n_bibs"].apply(lambda x: ast.literal_eval(str(x)))
    df = df[df["n_bibs_eval"].apply(lambda x: sum(i != 0 for i in x)) >= 10]
    df = df.drop(["n_bibs_eval"], axis=1)
    sample = df.sample(n=args.n_samples, random_state=args.seed)
    sample.to_csv(f"data/scireviewgen/subset_{args.n_samples}_{args.seed}.csv")

    # Memory Leak Prevention
    del df
    gc.collect()

    # Get the required ID's
    corpus_ids = get_corpus_ids(sample)

    # Get the Semantic Scholar Metadata
    print("Requesting the Semantic Scholar metadata...")
    target_file = re.sub(
        ".csv", f"_{args.n_samples}_{args.seed}.csv", METADATA_LOCATION
    )
    request_metadata(corpus_ids, target_file)

    print("Metadata collection finished!")
    print("Downloading finished!\n")

    # Memory Leak Prevention
    del sample, corpus_ids
    gc.collect()


if __name__ == "__main__":

    main()
