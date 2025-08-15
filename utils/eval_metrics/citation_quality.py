# Functions for evaluating the citation quality
# (Code adapted from https://github.com/AutoSurveys/AutoSurvey)
import sys

sys.path.append(".")

import gc, os, re, ast, time, logging, itertools, threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from typing import Dict, List, Tuple, Union
from autosurvey.src.model import APIModel
from autosurvey.src.prompt import NLI_PROMPT
from autosurvey.src.database import database
from utils.constants import MAX_TOKENS, ALIASES
from utils.text_utils import handle_entry, count_tokens
from utils.misc import read_file, load_json, save_to_json

# Set this to remove the text cutoff from pandas
pd.set_option("display.max_colwidth", None)

# Disabling the HTTPS info notifications
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def generate_prompt(template: str, paras: Dict) -> str:
    """
    Formats a prompt with a dictionary of parameters.

    Parameters:
    template - the prompt template.
    paras - the parameters.

    Returns:
    the formatted prompt.
    """

    prompt = template
    for k in paras.keys():

        prompt = prompt.replace(f"[{k}]", paras[k])

    return prompt


def split_prompt(
    sources: List[str], claim: str, max_tokens: int = MAX_TOKENS
) -> List[str]:
    """
    Splits the sources for a prompt if the total is too long.

    Parameters:
    sources - the list of sources to check for.
    claim - the claim to check for.
    max_tokens - the maximum number of tokens the LLM can handle.

    Returns:
    the list of prompts.
    """

    # First check the prompt
    content_paras = {"SOURCE": "\n".join(sources), "CLAIM": claim}
    prompt = generate_prompt(NLI_PROMPT, content_paras)

    # Then split if needed
    if count_tokens(prompt) >= max_tokens:
        print("Prompt too long. Splitting...")
        sources_split = []
        for source in sources:

            sources_split.extend(source.split("\n\n\n"))

        # Then prompt the model
        prompts = []
        for split in sources_split:

            content_paras = {"SOURCE": "\n".join(split), "CLAIM": claim}
            prompts.append(generate_prompt(NLI_PROMPT, content_paras))

    # Otherwise just return the actual prompt as a list
    else:
        prompts = [prompt]

    return prompts


def nli(
    sources: List[str],
    claim: str,
    res_l: List,
    idx: int,
    api_model: APIModel,
    max_tokens: int = MAX_TOKENS,
) -> int:
    """
    Determines the entailment of a claim given a list of sources (Natural Language Inference).

    Parameters:
    sources - the list of sources to check for.
    claim - the claim to check for.
    res_l - the result list to save to.
    idx - the location to store the output in for `res_l`.
    api_model - the API model to use for prompting.
    max_tokens - the maximum number of tokens the LLM can handle.

    Returns:
    the entailment score of the article (either 0 or 1).
    """

    prompt_list = split_prompt(sources, claim, max_tokens=max_tokens)
    res_list = []
    for prompt in prompt_list:

        res = api_model.chat(prompt, temperature=0.0)
        entailed = "yes" in res.lower() if isinstance(res, str) else False
        res_list.append(entailed)

        # Stop early if a yes is present
        if entailed:
            break

    result = 1 if any(res_list) else 0
    res_l[idx] += result
    return result


def relevant(
    sources: List[str],
    com_sources: List[str],
    claim: str,
    res_l: List,
    idx: int,
    api_model: APIModel,
    max_tokens: int = MAX_TOKENS,
) -> int:
    """
    Determines the relevance of a claim given a list of sources.

    Parameters:
    sources - the list of sources to check for.
    com_sources - the list of other sources used by the review which do not support the claim.
    claim - the claim to check for.
    res_l - the result list to save to.
    idx - the location to store the output in for `res_l`.
    api_model - the API model to use for prompting.
    max_tokens - the maximum number of tokens the LLM can handle.

    Returns:
    the relevance score of the article (either 0 or 1).
    """

    prompt_list = split_prompt(sources, claim, max_tokens=max_tokens)
    res_list = []
    for prompt in prompt_list:

        res = api_model.chat(prompt, temperature=0.0)
        entailed = "yes" in res.lower() if isinstance(res, str) else False
        res_list.append(entailed)

        # Stop early if a yes is present
        if entailed:
            break

    if any(res_list):
        res_l[idx] += 1
        return 1

    else:
        com_prompts = split_prompt(com_sources, claim, max_tokens=max_tokens)
        com_res = []
        for com_prompt in com_prompts:

            res = api_model.chat(com_prompt, temperature=0.0)
            com_res.append(
                "yes" not in res.lower()
            )  # Reversed because if it's supported here, then it conflicts with the reference

        final_com_res = 0 if any(com_res) else 1
        res_l[idx] += final_com_res
        return final_com_res


def extract_num(string: str) -> int:
    """
    Extracts the numbers in a string.

    Parameters:
    string - the string to parse.

    Returns:
    the number as an integer.
    """

    numbers = re.findall(r"\d+", string)
    if len(numbers) == 0:
        return ""

    # Adjustment: remove numbers which don't make sense to use for referencing (i.e., leading zeros, below or equal to zero, too large)
    if re.search(r"^00\d+", numbers[0]) is not None:
        return ""

    output = eval(numbers[0])
    if output <= 0 or output >= 5000:
        return ""

    return output


def preprocess_ref_text(ref_list_text: str) -> Dict:
    """
    Converts the preprocessed references (used for baseline and MASS generation) into a dictionary for
    later use.

    Parameters:
    ref_list_text - the processed references file as used by the baseline and MASS generation methods.

    Returns:
    a dictionary in the following format: {[INDEX_NUMBER]: [REFERENCE_TITLE]}
    """

    final_out = {}
    ref_list = re.split("\n\n", ref_list_text)
    for entry in ref_list:

        parts = re.split("\n", entry)
        # If there are too little parts: Most likely it's just an empty space
        if len(parts) < 3:
            continue

        num = int(re.sub(r"Number: \[(\d+)\]", r"\1", parts[0]))
        title = re.sub("Title: ", "", parts[1])
        final_out[num] = title

    return final_out


def get_id_by_titles(
    df: pd.DataFrame, review_id: Union[int, str], dataset: str = "srg"
) -> Dict:
    """
    Fetches the reference ID and titles from a dataframe and returns them as a dictionary.

    Parameters:
    df - the dataframe to process.
    review_id - the ID of the original review paper used as the basis (for filtering references).
    dataset - the dataset to use.

    Returns:
    a dictionary in the form of {[REFERENCE_TITLE]: [REFERENCE_ID]}
    """

    if dataset == "srg":
        review = df[df["paper_id"] == review_id]
        ref_list = handle_entry(review, "bib_titles", dataset)
        ref_ids = [list(r.items()) for r in ref_list]
        ref_ids = list(itertools.chain.from_iterable(ref_ids))
        ref_dict = {entry[1]: int(entry[0]) for entry in ref_ids}

        # Memory Leak Prevention
        del review, ref_list, ref_ids
        gc.collect()

    else:
        raise ValueError(f"Dataset `{dataset}` not supported!")

    return ref_dict


def preprocess_table(
    paper_ids: List[Union[int, str]],
    generated_table: pd.DataFrame,
    dataset: str = "srg",
    method: str = "base",
    data_dir: str = "data",
    ref_dir: str = "./articles",
    end_extension: str = "",
) -> Tuple[List[List[str]], List[Dict]]:
    """
    Preprocesses the data for citation quality evaluation.

    Parameters:
    paper_ids - the paper IDs to use to ensure ordering is preserved.
    generated_table - the AI generated article table.
    dataset - the dataset being used.
    method - the method of article generation used.
    data_dir - the location of the dataset files (used if method is not `auto`).
    ref_dir - the location where the reference JSON file is located (only used if method is `auto`).
    end_extension - the file extension to use for the subset extension used for SciReviewGen
    (i.e., `_subset_[n_samples]_[seed]`)).

    Returns:
    the lists of surveys split by section and references by index.
    """

    # The paper file (if AutoSurvey is not used)
    if method != "auto":
        data_file = (
            os.path.join(data_dir, f"{end_extension.strip('_')}.csv")
            if dataset == "srg"
            else os.path.join(data_dir, "ref_metadata.csv")
        )  # The subset file is used as the metadata may be incomplete due to missing SemanticScholar data
        paper_data = pd.read_csv(data_file)

    # Check the sorting to ensure the ordering is preserved
    gen_sorted = generated_table.loc[
        pd.Index(paper_ids).get_indexer(generated_table["paper_id"]).argsort()
    ]

    # Memory Leak Prevention
    del generated_table
    gc.collect()

    # Getting the data pairs
    surveys_split, reference_dicts = [], []
    paper_list = list(gen_sorted.iterrows())

    # Preparing the table for LiRA
    if method == "lira":
        ref_json = load_json(os.path.join(data_dir, "full_data_abs.json"))

    for _, entry in tqdm(
        paper_list,
        desc="Formatting results for Citation Quality Evaluation",
        total=len(paper_list),
    ):

        if dataset == "srg":
            text = handle_entry(entry, "text", dataset)

        else:
            raise ValueError(f"Dataset `{dataset}` not supported!")

        paper_id = entry["paper_id"]
        if method in ["base", "mass"]:
            # Fetch the text from the processed references files
            processed_ref_file = os.path.join(
                data_dir,
                "processed",
                f"{paper_id}{end_extension}_references.txt",
            )
            processed_ref_text = read_file(processed_ref_file)
            processed_ref_dict = preprocess_ref_text(processed_ref_text)
            title_to_id = get_id_by_titles(paper_data, paper_id, dataset)
            references = {
                entry[0]: title_to_id[entry[1]] for entry in processed_ref_dict.items()
            }

        elif method == "auto":
            # Then retrieve the references JSON created by AutoSurvey
            ref_path = os.path.join(ref_dir, f"{paper_id}_ref{end_extension}.json")
            references = load_json(ref_path)
            references = {int(k): v for k, v in references.items()}

        elif method == "lira":
            # Then retrieve the references by ID's for LiRA
            ref_main = [entry for entry in ref_json if entry["id"] == paper_id][0]
            references = {entry["num"]: entry["id"] for entry in ref_main["references"]}

        else:
            raise ValueError(f"Method `{method}` not supported!")

        surveys_split.append(text)
        reference_dicts.append(references)

    return surveys_split, reference_dicts


def extract_claims_sources(survey_sections: List[str]) -> Tuple[List, List]:
    """
    Extracts all claims and source ID's from a list of sections.

    Parameters:
    survey_sections - the survey split by section.

    Returns:
    a list of all claims and a list of source ID's.
    """

    citation_pattern = re.compile(r"[^.!?]*\[[^\]]+\][^.!?]*[.!?]")
    sentences = []
    for content in survey_sections:

        sentences += citation_pattern.findall(content)

    claims, sources_ids = [], []
    for s in sentences:

        sources = re.findall(r"\[(.*?)\]", s)
        if len(sources) > 0:
            source_ids = set()

            for ref in sources:

                for num in ref.split(";"):

                    number = extract_num(num)
                    if number != "":
                        source_ids.add(number)

            if len(source_ids) > 0:
                claims.append(re.sub(r"\[(.*?)\]", "", s))
                sources_ids.append(list(source_ids))

    return claims, sources_ids


def citation_quality_single(
    survey_sections: List[str],
    references: Dict,
    db: database,
    api_model: APIModel,
    max_tokens: int = MAX_TOKENS,
) -> Tuple[float, float, int]:
    """
    Evaluates the citation quality for a single review and counts the number of claims made.

    Parameters:
    survey_sections - the survey split by section.
    references - the dictionary of references in the form of: {[INDEX_IN_PAPER]: [PAPER_ID]}.
    db - the database containing all the reference data (created by running `run_autosurvey.py`).
    api_model - the API model to use for prompting.
    max_tokens - the maximum number of tokens the LLM can handle.

    Returns:
    the recall, precision, and number of claims of the article as a tuple, in the order mentioned here.
    """

    # Extract the claims and source ID's
    claims, sources_ids = extract_claims_sources(survey_sections)

    # Get the paper infos
    paper_infos = db.get_paper_info_from_ids(list(references.values()))
    ids_to_paper = {p["id"]: p["abs"] for p in paper_infos}
    index_to_paper = {
        int(index): ids_to_paper[idx] for index, idx in references.items()
    }

    # Processing the fulltexts
    thread_l = []
    scores = [0] * len(claims)

    # If the results are NaN (i.e., because the method did not cite papers properly, then return zero)
    if len(scores) <= 0:
        print("Zero claims found. Returning zero...")
        return 0.0, 0.0, len(claims)

    for i in range(len(claims)):

        sources = [
            index_to_paper[index] for index in sources_ids[i] if index in index_to_paper
        ]  # Make sure that the index actually exists
        thread = threading.Thread(
            target=nli, args=(sources, claims[i], scores, i, api_model, max_tokens)
        )
        thread_l.append(thread)
        thread.start()

    for thread in thread_l:

        thread.join()

    citation_num, thread_l = 0, []
    precisions = [0] * len(claims)
    for j, claim, source_ids in zip(range(len(claims)), claims, sources_ids):

        citation_num += len(source_ids)
        if scores[j] == 1:
            for index in source_ids:

                sources = [index_to_paper[index]]
                com_sources = [index_to_paper[_] for _ in source_ids if not _ == index]
                thread = threading.Thread(
                    target=relevant,
                    args=(
                        sources,
                        com_sources,
                        claim,
                        precisions,
                        j,
                        api_model,
                        max_tokens,
                    ),
                )
                thread_l.append(thread)
                thread.start()

    for thread in thread_l:

        thread.join()

    recalls = np.array(scores).mean()
    precisions = np.array(precisions).sum() / citation_num
    return recalls, precisions, len(claims)


def citation_quality(
    paper_ids: List[Union[int, str]],
    generated_table: pd.DataFrame,
    method: str,
    dataset: str,
    end_extension: str,
    setting_name: str,
    args: Namespace,
) -> Dict:
    """
    Main function for running the citation quality (recall and precision) evaluation.

    Parameters:
    paper_ids - the paper IDs to use to ensure ordering is preserved.
    generated_table - the table of AI written articles.
    method - the method of article generation used.
    dataset - the name of the dataset being used.
    overwrite - Whether to overwrite existing results or not.
    end_extension - the extension to use for saving the file if needed for theSciReviewGen subsetting).
    setting_name - the name of the specific setting used (specific to only LiRA).
    args - a Namespace based on the main experiment script being used (i.e., `baseline_mass.yaml`).

    Returns:
    a dictionary containing the recalls and precisions for all articles in the dataset in the following format:

    {"recall": [LIST_OF_RECALLS], "precision": [LIST_OF_PRECISIONS]}
    """

    # Clearing space (in case of OOM errors)
    gc.collect()

    # Initialize the fulltext database
    db = database(db_dir=args.db_dir, embedding_model=args.embedding_model)

    # Initialize paper data
    surveys, reference_lists = preprocess_table(
        paper_ids,
        generated_table,
        dataset,
        method=method,
        data_dir=args.data_dir,
        ref_dir=args.final_papers_dir,  # Where the references are saved
        end_extension=end_extension,
    )

    # Load the model
    api_model = APIModel(args.model)

    # Temporary file for saving in case issues arise with the API
    temp_dir = os.path.join("results", "citation", "temp")
    temp_file = os.path.join(
        temp_dir, f"{method}_{dataset}{setting_name}{end_extension}.json"
    )
    os.makedirs(temp_dir, exist_ok=True)
    temp_dict = load_json(temp_file) if os.path.isfile(temp_file) else {}

    # File for saving the number of claims
    n_dir = os.path.join("results", "citation", "n_claims")
    n_file = os.path.join(
        n_dir, f"{method}_{dataset}{setting_name}{end_extension}.json"
    )
    os.makedirs(n_dir, exist_ok=True)
    n_dict = load_json(n_file) if os.path.isfile(n_file) else {}

    # Calculate the results (also saves the article lengths)
    recall, precision = [], []
    for survey, references, id in tqdm(
        zip(surveys, reference_lists, paper_ids),
        desc="Calculating Citation Quality",
        total=len(surveys),
        unit="file(s)",
    ):

        if str(id) not in temp_dict:
            r, p, num_c = citation_quality_single(
                survey_sections=survey,
                references=references,
                db=db,
                api_model=api_model,
                max_tokens=args.max_tokens,
            )
            temp_dict[id] = [r, p]
            save_to_json(temp_dict, temp_file)
            time.sleep(2)  # To slow down requests

        else:
            r, p = (
                temp_dict[str(id)][0],
                temp_dict[str(id)][1],
            )
            c, _ = extract_claims_sources(survey)
            num_c = len(c)

        recall.append(r)
        precision.append(p)
        n_dict[str(id)] = num_c
        save_to_json(n_dict, n_file)

    # Memory Leak Prevention
    del db, api_model, surveys, reference_lists
    gc.collect()
    return {"recall": recall, "precision": precision}
