import gc, os, re, time, uuid, hydra
from argparse import Namespace
from omegaconf import DictConfig
from utils.evaluator import eval_method
from typing import List, Dict, Tuple, Union
from src.retriever import query_api
from src.builder import build_research_workflow
from src.utils.preprocessing import get_papers
from src.utils.file_utils import read_file, load_json, save_to_file, save_to_json
from src.utils.post_processing import adjust_review, convert_save_to_table
from src.utils.constants import (
    LLM_SEED,
    ALIASES,
    TEMP_DIR,
    MODEL_VER,
    BEGIN_PROMPT,
    SECTION_LEN,
    SETTING_NAME,
    CONTEXT_SIZE,
    NUM_SECTIONS,
    NUM_SUBSECTIONS,
)


# Write a review
def run_research_workflow(
    papers: List[Dict[str, str]],
    topic: str,
    review_id: Union[int, str],
    dataset: str,
    seed: int = LLM_SEED,
    max_revisions: int = 3,
    researcher_model: str = MODEL_VER,
    writer_model: str = MODEL_VER,
    editor_model: str = MODEL_VER,
    reviewer_model: str = MODEL_VER,
    temp_dir: str = TEMP_DIR,
    display_stream: bool = True,
    n_jobs: int = -4,
    fulltext: bool = False,
    n_sections: int = NUM_SECTIONS,
    n_subsections: int = NUM_SUBSECTIONS,
    section_len: int = SECTION_LEN,
    context_sizes: Dict = {"gpt": CONTEXT_SIZE, "other": CONTEXT_SIZE},
    setting_name: str = SETTING_NAME,
    overwrite_responses: bool = False,
    use_retriever: bool = False,
) -> Tuple[str, Dict]:

    # Prepare the configuration
    config = {"configurable": {"thread_id": uuid.uuid4()}, "recursion_limit": 60}

    # Build the workflow
    app = build_research_workflow(
        papers,
        topic=topic,
        review_id=review_id,
        seed=seed,
        dataset=dataset,
        max_revisions=max_revisions,
        researcher_model=researcher_model,
        writer_model=writer_model,
        editor_model=editor_model,
        reviewer_model=reviewer_model,
        temp_dir=temp_dir,
        n_jobs=n_jobs,
        fulltext=fulltext,
        n_sections=n_sections,
        n_subsections=n_subsections,
        section_len=section_len,
        setting_name=setting_name,
        context_sizes=context_sizes,
        overwrite_responses=overwrite_responses,
        use_retriever=use_retriever,
    )

    # This is just to start the system (does NOT get read)
    input_message = BEGIN_PROMPT.format(REVIEW_ID=review_id)

    # Running and displaying the workflow
    # (set subgraphs to `True` for the researcher group updates)
    count = 0
    for event in app.stream(
        {"messages": [input_message]},
        config,
        stream_mode="values",
        subgraphs=True,
    ):

        # Skip the first message
        if count <= 0:
            count += 1
            continue

        # Determining what to print
        to_print = event[1]["messages"][-1]
        if display_stream:
            to_print.pretty_print()
            print()

        count += 1

    # return the final paper and print the workflow end
    print(
        "================================================================================\n"
    )
    print(f"Execution finished with {count} events!\n")

    if editor_model.lower() != "none":
        final_review = app.get_state(config)[0]["final_review"]

    else:
        final_review = app.get_state(config)[0]["draft_review"]

    # Memory Leak Prevention
    del app
    gc.collect()
    adjusted_review, references = adjust_review(final_review, papers)
    return adjusted_review, references


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):

    # Loading the config
    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    # Pre-adjusting certain arguments
    end_extension = (
        f"_subset_{args.n_samples}_{args.seed}" if args.dataset == "srg" else ""
    )
    no_editor = "_noedit" if args.editor_model == "none" else ""
    args.method = "lira"
    args.final_papers_file = re.sub(
        ".csv", f"_{args.dataset}{end_extension}.csv", args.final_papers_file
    )

    # Define the setting name
    ft = "ft" if args.fulltext else "abs"

    # Checking if every model specified is the same
    model_list = [args.researcher_model, args.writer_model, args.reviewer_model]
    model_list = [entry for entry in model_list if entry != "none"]
    models = "same" if len(set(model_list)) == 1 else "mix"

    setting_name = f"_{ft}_{models}"
    setting_name += "_ret" if args.use_retriever else ""
    setting_name += "_noresearch" if args.researcher_model == "none" else ""
    args.setting_name = setting_name

    # Ensure that the section length is acceptable
    assert args.section_len > 0, "Please enter a `section_len` larger than 0!"

    # Loading the data
    papers_list = get_papers(
        args.dataset, args.data_dir, end_extension, fulltext=args.fulltext
    )

    # Setting up the context size dictionary
    context_sizes = {"gpt": args.max_tokens_gpt, "other": args.max_tokens_other}

    # Setup the directory
    save_folder = os.path.join(
        args.final_papers_dir, f"{args.method}{setting_name}{no_editor}"
    )  # For the `.csv` file
    save_dir = os.path.join(save_folder, args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    # The extension for file handling
    end_extension = (
        f"_subset_{args.n_samples}_{args.seed}" if args.dataset == "srg" else ""
    )

    # Check if the results are already present
    full_save_path = os.path.join(save_folder, args.final_papers_file)
    if os.path.isfile(full_save_path) and not args.overwrite_responses:
        print("All files generated. Continuing to evaluation...")

    else:
        # Retrieval if used
        if args.use_retriever:
            papers_list = query_api(
                papers_list,
                data_dir=args.data_dir,
                dataset=args.dataset,
                seed=args.llm_seed,
                temperature=args.temperature,
            )

        # Run for all papers
        temp_dict = {}
        print("Running the worfklow for all papers. This may take a long time...\n")
        for paper in papers_list:

            # First check if the review has been made already
            id = paper["id"]
            out_name = os.path.join(save_dir, f"{id}{end_extension}.txt")
            json_out_name = os.path.join(save_dir, f"{id}{end_extension}_ref.json")
            if os.path.isfile(out_name) and not args.overwrite_responses:
                print(f"Review `{id}` already generated! Skipping...")
                review = read_file(out_name)
                references = load_json(json_out_name)

            else:
                review, references = run_research_workflow(
                    papers=paper["references"],
                    topic=paper["topic"],
                    review_id=paper["id"],
                    seed=args.llm_seed,
                    dataset=args.dataset,
                    max_revisions=args.n_revisions,
                    researcher_model=args.researcher_model,
                    writer_model=args.writer_model,
                    editor_model=args.editor_model,
                    reviewer_model=args.reviewer_model,
                    n_jobs=args.n_jobs,
                    context_sizes=context_sizes,
                    fulltext=args.fulltext,
                    n_sections=args.n_sections,
                    n_subsections=args.n_subsections,
                    section_len=args.section_len,
                    setting_name=setting_name[1:],  # To skip the initial underscore
                    overwrite_responses=args.overwrite_responses,
                    use_retriever=args.use_retriever,
                    temp_dir=args.temp_dir,
                )

                # Save the paper and reference dictionary
                save_to_file(review, out_name)
                save_to_json(references, json_out_name)
                time.sleep(2)

            # Store for processing into a table
            temp_dict[id] = (review, references)

        # Process all the entries into a table and evaluate the system
        convert_save_to_table(temp_dict, full_save_path)

    # Finally, evaluate the method
    args.data_dir = os.path.join(args.data_dir, ALIASES[args.dataset].lower())
    if args.dataset == "srg":
        content_file = os.path.join(
            args.data_dir,
            f"subset_{args.n_samples}_{args.seed}.csv",
        )

    else:
        raise ValueError(f"Dataset `{args.dataset}` not supported!")

    # Adjustment for the evaluator
    args.db_dir = os.path.join(args.db_dir, args.dataset)
    args.final_papers_file = os.path.join(save_folder, args.final_papers_file)
    args.final_papers_dir = os.path.join(save_folder, args.dataset)
    eval_method(content_file, end_extension, args)


if __name__ == "__main__":

    main()
