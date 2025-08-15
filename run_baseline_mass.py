import os, re, hydra
from argparse import Namespace
from omegaconf import DictConfig
from utils.constants import ALIASES
from utils.evaluator import eval_method
from baseline_and_mass.baseline import generate_all_baselines
from baseline_and_mass.extract_data import (
    process_all_entries_sd,
    process_all_entries_srg,
)
from baseline_and_mass.agents import (
    generate_csv,
    generate_txt,
    analyze_all_references,
    generate_all_conclusions,
    generate_all_chapter_contents,
    generate_all_titles_and_abstracts,
)


@hydra.main(version_base=None, config_path="configs", config_name="baseline_mass")
def main(cfg: DictConfig):

    # Loading the config
    h_args = cfg.get("base")
    args = Namespace(**h_args)

    # Setting the location for the processed files
    dataset_name = ALIASES[args.dataset].lower()
    args.data_dir = os.path.join(args.data_dir, dataset_name)
    args.db_dir = os.path.join(args.db_dir, args.dataset)  # Adjust to save in folders
    args.processed_dir = os.path.join(args.data_dir, "processed")

    # Adjusting filenames based on method and dataset
    args.base_dir = os.path.join(args.base_dir, args.method, args.dataset)
    args.structured_ref_dir = os.path.join(args.base_dir, args.structured_ref_dir)
    args.title_abstract_dir = os.path.join(args.base_dir, args.title_abstract_dir)
    args.chapter_contents_dir = os.path.join(args.base_dir, args.chapter_contents_dir)
    args.conclusion_dir = os.path.join(args.base_dir, args.conclusion_dir)

    # Extension if the dataset is SciReviewGen
    extension = f"_subset_{args.n_samples}_{args.seed}" if args.dataset == "srg" else ""
    args.final_papers_file = os.path.join(
        args.final_papers_dir, args.method, args.final_papers_file
    )
    args.final_papers_file = re.sub(
        ".csv", f"_{args.dataset}{extension}.csv", args.final_papers_file
    )
    args.final_papers_dir = os.path.join(
        args.final_papers_dir, args.method, args.dataset
    )  # Saves to the `articles` folder

    # Begin Pipeline
    print(
        f"Generating {args.method.upper()} articles for the '{ALIASES[args.dataset]}' dataset...\n"
    )

    # Data pre-processing
    if args.dataset == "srg":
        content_file = os.path.join(
            args.data_dir, f"subset_{args.n_samples}_{args.seed}.csv"
        )
        process_all_entries_srg(content_file, args)

    else:
        raise ValueError(f"Dataset `{args.dataset}` not supported!")

    # Content Generation
    if args.method == "base":
        generate_all_baselines(args)

    elif args.method == "mass":
        analyze_all_references(args)
        generate_all_titles_and_abstracts(args)
        generate_all_chapter_contents(args)
        generate_all_conclusions(args)
        generate_csv(args)
        generate_txt(args)

    else:
        raise ValueError(f"Dataset `{args.dataset}` not supported!")

    # Evaluation
    eval_method(content_file, extension, args)


if __name__ == "__main__":

    main()
