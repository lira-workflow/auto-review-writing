# File preprocessing utilities
import gc, os, ast, itertools
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from src.utils.constants import ALIASES, DATA_FOLDER
from src.utils.file_utils import load_json, save_to_json


def merge_dict_list(dict_list: List[Dict]) -> Dict:
    """
    Merges a list of dictionaries into a dictionary.
    This is needed due to how SciReviewGen formats the references.

    Parameters:
    dict_list - the list of dictionaries.

    Returns:
    the combined entries as one dictionary.
    """

    out = {}
    for d in dict_list:

        out.update(d)

    return out


def get_papers(
    dataset: str = "srg",
    location: str = DATA_FOLDER,
    end_extension: str = "",
    fulltext: bool = False,
) -> List[Dict]:
    """
    Loads the reference papers for a dataset.

    Parameters:
    dataset - the dataset to load.
    location - the folere containing the dataset subfolders.
    end_extension - the extension to use for saving the file if needed (for the SciReviewGen subsetting).
    fulltext - whether to use the fulltexts for writing or not.

    Returns:
    the papers as a list of dictionary entries in the following format:
    [
        {
            "id": [REVIEW_ID],
            "topic": [REVIEW_TITLE],
            "references": [
                {
                    "id": [REFERENCE_ID],
                    "title": [REFERENCE_TITLE],
                    "content": [REFERENCE_CONTENT]
                },
            ]
        }
    ]
    """

    if dataset == "srg":
        file_base = end_extension[1:]
        data = prepare_srg(f"{file_base}.csv", location=location, fulltext=fulltext)

    else:
        raise ValueError(f"Dataset `{dataset}` not supported!")

    return data


def prepare_srg(
    target_file: str, location: str = "data", fulltext: bool = False
) -> List[Dict]:
    """
    Creates and prepares the data for the SciReviewGen dataset.
    This data is then saved to prevent repeat processing in consequent runs.

    Parameters:
    target_file - the `.csv` file containing the review data.
    location - the location to load and save the data.
    fulltext - whether to use the fulltexts for writing or not.

    Returns:
    the data as a list of dictionaries.
    """

    # First check if the dataset file has been made already
    folder = os.path.join(location, ALIASES["srg"].lower())
    filename = "full_data"
    filename += "_ft.json" if fulltext else "_abs.json"
    dataset_store_file = os.path.join(folder, filename)
    if os.path.isfile(dataset_store_file):
        print("SciReviewGen data already preprocessed! Loading...")
        papers_final = load_json(dataset_store_file)
        return papers_final

    # If not then get the fulltexts
    if fulltext:
        fulltext_file = os.path.join(folder, "fulltext_final.json")
        fulltext_dict = load_json(fulltext_file)

    else:
        fulltext_dict = {}

    # Then the database file (for the abstracts and titles)
    file_dir = os.path.join(folder, target_file)
    papers = pd.read_csv(file_dir)

    # Process the SciReviewGen reference papers
    papers_final = []
    for _, paper in tqdm(
        papers.iterrows(),
        desc="Formatting Paper Information",
        unit="file(s)",
        total=len(list(papers.iterrows())),
    ):

        # Setup the base container
        review_id = paper["paper_id"]
        entry = {"id": review_id, "topic": paper["title"], "references": []}

        # Get the data for the references
        ref_titles = ast.literal_eval(paper["bib_titles"])
        ref_titles_full = merge_dict_list(ref_titles)
        ref_ids = [list(r.keys()) for r in ref_titles]
        ref_ids = list(itertools.chain.from_iterable(ref_ids))

        # Deduplication
        ref_ids = list(set(ref_ids))
        num = 1
        for ref_id in ref_ids:

            ref_entry = {
                "num": num,
                "id": int(ref_id),
                "title": ref_titles_full[ref_id],
            }
            if fulltext and str(ref_id) in fulltext_dict:
                fulltext = fulltext_dict[str(ref_id)]

            else:
                # We then use the abstracts instead
                ref_abstracts = ast.literal_eval(paper["bib_abstracts"])
                ref_abs_full = merge_dict_list(ref_abstracts)
                fulltext = ref_abs_full[ref_id]

            ref_entry["content"] = fulltext
            entry["references"].append(ref_entry)
            num += 1

        papers_final.append(entry)

    # Memory Leak Prevention
    del (
        papers,
        fulltext,
        fulltext_dict,
        ref_abstracts,
        ref_abs_full,
        ref_titles,
        ref_titles_full,
    )
    gc.collect()

    # Save for later use
    save_to_json(papers_final, dataset_store_file)
    return papers_final
