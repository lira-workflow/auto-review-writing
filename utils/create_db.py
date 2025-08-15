import gc, os, re, ast, torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from argparse import Namespace
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union
from sentence_transformers import SentenceTransformer
from utils.misc import read_file, load_json, save_to_json, merge_dict_list

# Set this to remove the text cutoff from pandas
pd.set_option("display.max_colwidth", None)


def encode_batch(embedding_model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Helper function to encode a batch of texts.

    Parameters:
    embedding_model - the embedding model instance to use.
    texts - the texts to encode.

    Returns:
    the array of embeddings.
    """

    return embedding_model.encode(texts, convert_to_numpy=True)


def prepare_dataset(args: Namespace) -> List[Dict]:
    """
    Loads a dataset as a dictionary.

    Parameters:
    api_model - the API model to use for prompting.
    args - a Namespace containing the following variables which get read (mostly in the sub-functions):

    db_dir - the location of the database.
    n_samples - the number of samples used for selection.
    seed - the seed used for sample selection.
    dataset - the dataset to use.

    Returns:
    the data loaded as a list of dictionaries.
    """

    # Setup the file directories
    os.makedirs(args.db_dir, exist_ok=True)
    if args.dataset == "srg":
        data, ref_data = prepare_srg(args)

    else:
        raise ValueError(f"The dataset is not supported: `{args.dataset}`!")

    create_database(ref_data, args)

    # Memory Leak Prevention
    del ref_data
    gc.collect()
    return data


def remove_duplicates(
    json_list: List[Dict],
    key1: Union[int, str, None] = "review_id",
    key2: Union[int, str, None] = "id",
):
    """
    Deduplicates a list of dictionaries based on two key values.
    Used for deduplicating the references.

    Parameters:
    json_list - the list of entries to deduplicate.
    key1 - the first key (value) to check for.
    key2 - the second key (value) to check for.
    """

    seen = set()
    new_list = []
    for item in json_list:

        identifier = (item[key1], item[key2])
        if identifier not in seen:
            seen.add(identifier)
            new_list.append(item)

    return new_list


def create_embeddings(
    texts: List[str],
    model_name: str,
    filename: str,
    batch_size: int = 32,
    n_jobs: int = -4,
    desc: str = "titles",
) -> np.array:
    """
    Embeds a list of texts into numpy arrays given an encoder.

    Parameters:
    texts - a list of texts to embed.
    model_name - the name of the encoder model to use.
    filename - the target file to save to/read from.
    batch_size - the batch size to use for batch encoding.
    n_jobs - the number of workers to use.
    desc - the description for the progress bar.

    Returns:
    a numpy array of all the texts embedded.
    """

    if os.path.isfile(filename):
        print(f"Embeddings file found for {desc}. Loading...")
        return np.load(filename)

    embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model.to(torch.device(device))
    print(f"Creating embeddings for {desc.capitalize()} using device: `{device}`")

    if device == "cuda":
        tmp = [
            embedding_model.encode(
                ["search_documents: " + str(text) for text in texts[i : i + batch_size]]
            )
            for i in trange(0, len(texts), batch_size)
        ]

    else:
        # Use Joblib for parallelization on CPU
        tmp = Parallel(n_jobs=n_jobs)(
            delayed(encode_batch)(
                embedding_model,
                [
                    "search_documents: " + str(text)
                    for text in texts[i : i + batch_size]
                ],
            )
            for i in trange(0, len(texts), batch_size)
        )

    out = np.concatenate(tmp, axis=0)
    np.save(filename, out)

    # Memory Leak Prevention
    del tmp, embedding_model
    gc.collect()
    return out


def prepare_srg(args: Namespace) -> List[Dict]:
    """
    Creates a database containing the title, abstracts, and topics of the SciReviewGen dataset.

    Parameters:
    api_model - the API model to use for prompting.
    args - a Namespace containing the following variables which get read:

    db_dir - the location of the database.
    n_samples - the number of samples used for selection.
    seed - the seed used for sample selection.
    dataset - the dataset to use.

    Returns:
    the data loaded as a list of dictionaries.
    Specifically, the following formats are used:

    For the main data:
    [
        {
            "id": [REVIEW_PAPER_ID],
            "title": [REFERENCE_PAPER_TITLE],
        }
    ]

    For the reference data:
    [
        {
            "review_id": [REVIEW_PAPER_ID],
            "id": [REFERENCE_ID],
            "title": [REFERENCE_PAPER_TITLE],
            "abs": [REFERENCE_PAPER_ABSTRACT],
        }
    ]
    """

    # First check if the processed data already exists
    main_file = os.path.join(args.db_dir, "main.json")
    ref_file = os.path.join(args.db_dir, "refs.json")
    if os.path.isfile(main_file):
        print("Data already formatted. Loading...")
        return load_json(main_file), load_json(ref_file)

    # Loading the data
    papers = pd.read_csv(
        os.path.join(args.data_dir, f"subset_{args.n_samples}_{args.seed}.csv")
    )
    references = pd.read_csv(
        os.path.join(
            args.data_dir,
            f"srg_metadata_{args.n_samples}_{args.seed}.csv",
        )
    )

    # Formatting the data
    data, ref_data = [], []
    for _, paper in tqdm(
        papers.iterrows(),
        desc="Formatting Paper Information",
        unit="file(s)",
        total=len(list(papers.iterrows())),
    ):

        # Setup the base container
        entry = {"id": paper["paper_id"], "title": paper["title"]}

        # Get the data for the references
        ref_titles = ast.literal_eval(paper["bib_titles"])
        ref_abstracts = ast.literal_eval(paper["bib_abstracts"])

        ref_titles_full = merge_dict_list(ref_titles)
        ref_abs_full = merge_dict_list(ref_abstracts)

        ref_ids = list(ref_titles_full.keys())
        ref_list = []
        for ref in ref_ids:

            ref_paper = references[references["corpusId"] == int(ref)]
            if len(ref_paper) > 0:
                # Check the abstracts from the table first
                if ref in ref_abs_full:
                    abs = ref_abs_full[ref]

                elif ref_paper["abstract"] is not None:
                    abs = ref_paper["abstract"]

                ref_entry = {
                    # To allow for exact reference fetching
                    "review_id": paper["paper_id"],
                    "id": int(ref_paper["corpusId"].iloc[0]),
                    "title": ref_paper["title"].to_string(index=False),
                    "abs": abs,
                }

            else:
                ref_entry = {
                    # To allow for exact reference fetching
                    "review_id": paper["paper_id"],
                    "id": int(ref),
                    "title": ref_titles_full[ref],
                    "abs": ref_abs_full[ref],
                }

            ref_list.append(ref_entry)

        ref_data.extend(ref_list)
        data.append(entry)

    # Memory Leak prevention by deleting variables
    del papers, references
    gc.collect()

    # De-duplication
    ref_data = remove_duplicates(ref_data)

    # Save the contents
    save_to_json(data, main_file)
    save_to_json(ref_data, ref_file)
    return data, ref_data


def create_database(data: List[Tuple], args: Namespace):
    """
    Creates the additional database files for a given dataset.
    This includes the title/abstract embeddings.

    Parameters:
    data - the data to complete/use.
    args - a Namespace containing the following variables which get read:

    db_dir - the location to store the database files.
    embedding_model - the embedding model instance to use.
    batch_size - the batch size to use for batch encoding.
    n_jobs - the number of workers to use.
    """

    # First check if the processed data already exists
    target_file = os.path.join(args.db_dir, "db.json")
    embed_file_title = os.path.join(args.db_dir, "title_emb.npy")
    embed_file_abs = os.path.join(args.db_dir, "abs_emb.npy")
    if os.path.isfile(target_file):
        print("Database with embeddings already created. Loading...")
        ref_data = ""

    else:
        # We continue with creating the embeddings for all titles and abstracts
        print("Creating embeddings for titles and abstracts, this may take a while...")
        title_embeds = create_embeddings(
            [entry["title"] for entry in data],
            args.embedding_model,
            embed_file_title,
            args.batch_size,
            args.n_jobs,
            desc="titles",
        )
        abs_embeds = create_embeddings(
            [entry["abs"] for entry in data],
            args.embedding_model,
            embed_file_abs,
            args.batch_size,
            args.n_jobs,
            desc="abstracts",
        )

        # Format the reference data
        ref_data = {}
        for idx, ref in enumerate(data):

            # Casting to list is required for JSON
            ref["title_embed"] = title_embeds[idx]
            ref["abs_embed"] = abs_embeds[idx]
            ref_data[str(idx + 1)] = ref

        ref_data = {"_default": ref_data}
        save_to_json(ref_data, target_file, use_numpy_encoder=True)

    # Memory Leak Prevention
    del ref_data
    gc.collect()


def get_fulltext(
    entry: Dict, df: pd.DataFrame, data_folder: str, dataset: str = "srg"
) -> List[str]:
    """
    Loads the fulltext of a reference entry.

    Parameters:
    entry - the dictionary entry of the reference paper.
    df - the dataframe for loading (currently unused).
    data_folder - the folder containing the data.
    dataset - the dataset to use.

    Returns:
    the text as a string.
    """

    id = entry["id"]
    if dataset == "srg":
        data_file = os.path.join(data_folder, "ref_fulltext", f"{id}.txt")
        if os.path.isfile(data_file):
            text = read_file(data_file)

        else:  # Use the abstract
            text = entry["abs"]

    return text
