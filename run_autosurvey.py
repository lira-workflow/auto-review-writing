import gc, os, re, hydra, logging
import numpy as np
import pandas as pd
from typing import List, Union
from argparse import Namespace
from omegaconf import DictConfig
from utils.constants import ALIASES
from utils.evaluator import eval_method
from utils.create_db import prepare_dataset
from autosurvey.src.database import database
from autosurvey.src.agents.writer import subsectionWriter
from autosurvey.src.agents.outline_writer import outlineWriter
from autosurvey.src.utils import load_txt, load_json, save_to_json

# Disabling the HTTPS info notifications
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Getting the AzureOpenAI variables
BASE_URL = os.getenv("AZURE_OPENAI_URL_BASE")
API_KEY = os.getenv("OPENAI_ORGANIZATION_KEY")


def remove_descriptions(text: str) -> str:
    """
    Removes the descriptions in an outline.

    Parameters:
    text - the outline to clean.

    Returns:
    the outline with the lines starting with the word "Description" removed.
    """

    lines = text.split("\n")
    filtered_lines = [
        line for line in lines if not line.strip().startswith("Description")
    ]
    result = "\n".join(filtered_lines)
    return result


def format_to_table(end_extension: str, args: Namespace) -> str:
    """
    Reads all the text files in the specified directory and formats them into a table.
    This is then saved to the output folder.

    Parameters:
    end_extension - the extension to use for saving the file if needed (for the SciReviewGen subsetting).
    args - a Namespace containing the following variables which get read:

    dataset - the dataset to use.
    method - the method (should just be "auto" here).
    final_papers_dir - the location where the files will be saved to.

    Returns:
    the name of the `.csv` table containing all the paper information.
    """

    files = os.listdir(args.final_papers_dir)
    table = {"paper_id": [], "title": [], "section": [], "text": []}
    for file in files:

        if file.endswith(".txt"):
            full_path = os.path.join(args.final_papers_dir, file)
            data = load_txt(full_path, use_truncation=False)
            full_text = data.split("## References")[0]  # Remove references
            data_list = re.split(r"\#{1,3} \d+(?:\.\d+)? .+\n", full_text)

            # Get the values
            paper_id = file.split(".txt")[0]
            if args.dataset == "srg":
                paper_id = re.sub(r"\_subset\_\d+\_\d+", "", paper_id)

            title = data_list[0]

            # Text cleaning
            text = [
                re.sub(r"\#{0,3} \d+(?:\.\d+)? .+\n\n", "", sect)
                for sect in data_list[1:]
            ]
            text = [re.sub(r"\d+ .+\n\n", "", sect).strip() for sect in data_list[1:]]

            # Add empty strings to ensure consistent length with the number of sections
            section_titles = re.findall(r"\# \d+ .+\n", full_text)
            sections = re.findall(r"\#{1,3} \d+(?:\.\d+)? .+\n", full_text)
            sections = [
                re.sub(r"\#{1,3} \d+(?:\.\d+)? ", "", sect).strip() for sect in sections
            ]
            for idx, section in enumerate(sections):

                if section in section_titles:
                    text.insert(idx, "")

            table["paper_id"].append(paper_id)
            table["title"].append(title)
            table["section"].append(sections)
            table["text"].append(text)

    df = pd.DataFrame.from_dict(table)
    df_name = os.path.join(
        args.final_papers_dir, "..", f"papers_{args.dataset}{end_extension}.csv"
    )
    df.to_csv(df_name, index=False)

    # Memory Leak Prevention
    del df, files, table
    return df_name


def fix_formatting_review(filedir: str, id_list: List[Union[int, str]]):
    """
    Fixes the formatting for some reviews which do not have the headers written properly (due to missing numbers).
    It automatically finds these problem articles and adjusts the numbering automatically.

    Parameters:
    filedir - the directory containing all reviews to fix.
    id_list - the article ID list of all faulty reviews.
    """

    for file in os.listdir(filedir):

        if not file.endswith(".txt") or not any(
            [file.find(str(id)) != -1 for id in id_list]
        ):
            continue

        fullpath = os.path.join(filedir, file)
        with open(fullpath, "r") as f:

            data = f.read()

        main_headers = re.findall(r"\#{1} .+\n", data)
        all_headers = re.findall(r"\#{1,2} .+\n", data)

        h_num, s_num = 0, 0
        for header in all_headers:

            if "## References" in header:
                continue

            if header in main_headers:
                h_num += 1
                s_num = 0
                new_header = re.sub("#", f"# {h_num}", header)

            else:
                s_num += 1
                new_header = re.sub("##", f"## {h_num}.{s_num}", header)

            data = data.replace(header, new_header)

        with open(fullpath, "w+") as f:

            f.write(data)


def format_and_fix(end_extension: str, args: Namespace) -> str:
    """
    Formats and fixes the entry data.

    Parameters:
    end_extension - the extension to use for saving the file if needed for theSciReviewGen subsetting).
    args - a Namespace containing the following variables which get read:

    dataset - the dataset to use.
    method - the method (should just be "auto" here).
    final_papers_dir - the location where the files will be saved to.

    Returns:
    the name of the `.csv` table containing all the paper information.
    """

    # First format the data initially
    df_name = format_to_table(end_extension, args)

    # Then check the file for empty sections
    df = pd.read_csv(df_name)
    to_fix = df[df["section"] == "[]"]
    id_list = to_fix["paper_id"].to_list()

    # If problematic entries exist, fix them
    if len(id_list) > 0:
        fix_formatting_review(args.final_papers_dir, id_list)

        # Continue with reformatting the data
        df_name = format_to_table(end_extension, args)

    return df_name


# Writing the outline
def write_outline(
    topic: str,
    model: outlineWriter,
    review_id: Union[int, str, None],
    section_num: int,
    save_dir: str,
    end_extension: str,
):

    outline = model.draft_outline(
        topic,
        review_id=review_id,
        chunk_size=30_000,
        section_num=section_num,
        save_dir=save_dir,
        end_extension=end_extension,
    )

    return outline, remove_descriptions(outline)


# Writing the subsection
def write_subsection(
    topic: str,
    model: subsectionWriter,
    outline: str,
    review_id: Union[int, str, None],
    subsection_len: int,
    rag_num: int,
    save_dir: str,
    final_dir: str,
    refinement: bool = True,
    end_extension: str = "",
):

    final_survey = model.write(
        topic,
        outline,
        review_id=review_id,
        subsection_len=subsection_len,
        rag_num=rag_num,
        refining=refinement,
        save_dir=save_dir,
        final_dir=final_dir,
        end_extension=end_extension,
    )
    return final_survey


def write_survey(
    topic: str,
    review_id: Union[int, str, None],
    outline_writer: outlineWriter,
    subsection_writer: subsectionWriter,
    end_extension: str,
    args: Namespace,
):

    # Writing the outline
    outline_with_description, _ = write_outline(
        topic,
        outline_writer,
        review_id,
        section_num=args.section_num,
        save_dir=args.outline_dir,
        end_extension=end_extension,
    )

    survey = write_subsection(
        topic,
        subsection_writer,
        outline_with_description,
        review_id,
        args.subsection_len,
        args.rag_num,
        save_dir=args.survey_dir,
        final_dir=args.final_papers_dir,
        end_extension=end_extension,
    )

    # Saves to "articles/auto" by default
    if survey is not None:
        txt_path = os.path.join(
            args.final_papers_dir, f"{review_id}{end_extension}.txt"
        )
        with open(txt_path, "w+") as f:

            f.write(survey)


@hydra.main(version_base=None, config_path="configs", config_name="autosurvey")
def main(cfg: DictConfig):

    # Loading the config
    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    # Merge the directories for organization
    dataset_name = ALIASES[args.dataset].lower()
    args.data_dir = os.path.join(args.data_dir, dataset_name)
    args.outline_dir = os.path.join(args.component_dir, args.outline_dir, args.dataset)
    args.survey_dir = os.path.join(args.component_dir, args.survey_dir, args.dataset)
    args.cost_dir = os.path.join(args.component_dir, args.cost_dir)

    print(f"Running for the {ALIASES[args.dataset]} dataset...")

    # Prepare the data and paths
    args.db_dir = os.path.join(args.db_dir, args.dataset)  # Adjust to save in folders
    args.method = "auto"  # The method for AutoSurvey == "auto"
    data = prepare_dataset(args)

    # Load the database
    db = database(args.db_dir, embedding_model=args.embedding_model)

    # Make the required directories
    args.final_papers_dir = os.path.join(
        args.final_papers_dir, args.method, args.dataset
    )
    os.makedirs(args.final_papers_dir, exist_ok=True)

    # Defining the models in the beginning to reduce overhead
    print("Loading the models...")
    outline_writer = outlineWriter(
        model=args.model, api_key=API_KEY, api_url=BASE_URL, database=db
    )
    subsection_writer = subsectionWriter(
        model=args.model, api_key=API_KEY, api_url=BASE_URL, database=db
    )

    # For the SciReviewGen subset naming
    end_extension = (
        f"_subset_{args.n_samples}_{args.seed}" if args.dataset == "srg" else ""
    )

    # Writing the surveys
    cost_file = os.path.join(args.cost_dir, f"{args.dataset}{end_extension}.json")
    costs = {}
    if os.path.isfile(cost_file):
        costs = load_json(cost_file)

    for entry in data:
        # We assume the title is the topic
        write_survey(
            entry["title"],
            entry["id"],
            outline_writer=outline_writer,
            subsection_writer=subsection_writer,
            end_extension=end_extension,
            args=args,
        )

        # Store the Price
        # (JSON stores the ID as a string, hence the conversion)
        if str(entry["id"]) not in costs:
            costs[entry["id"]] = (
                outline_writer.compute_price() + subsection_writer.compute_price()
            )

        outline_writer.reset_token_usage()
        subsection_writer.reset_token_usage()

    # Get final cost
    cost_list = [float(v) for v in costs.values()]
    cost_out = f"Total Cost: {sum(cost_list)}\nAverage Cost: {np.mean(cost_list)}"
    print(cost_out, "\n")

    # Save costs
    os.makedirs(args.cost_dir, exist_ok=True)
    save_to_json(costs, cost_file)

    # Format the results to match the evaluation pipeline
    args.final_papers_file = format_and_fix(end_extension, args)

    # Memory Leak Prevention
    del db, costs, cost_list, outline_writer, subsection_writer
    gc.collect()

    # Evaluate the articles
    if args.dataset == "srg":
        content_file = os.path.join(
            args.data_dir,
            f"subset_{args.n_samples}_{args.seed}.csv",
        )

    else:
        raise ValueError(f"Dataset `{args.dataset}` not supported!")

    # Evaluation
    eval_method(content_file, end_extension, args)


if __name__ == "__main__":

    main()
