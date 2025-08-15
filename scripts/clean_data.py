import gc, os, re, hydra
import pandas as pd
from tqdm import tqdm
from typing import List
from argparse import Namespace
from omegaconf import DictConfig


# Set this to remove the text cutoff from pandas
pd.set_option("display.max_colwidth", None)

# Terms to use for filtering out sections (as they only add noise)
filter_terms = [
    "competing interest",
    "contribution statement",
    "funding",
    "consent for publication",
    "ethics approval",
    "assurance",
    "author contributions",
    "conflict of interest",
]

# For chemical formula finding
form_pattern = (
    r"[A-Z][A-Za-z]?[ ]?\d{0,2}(?:[ ][A-Z][A-Za-z]?[ ]?\d{0,2}){0,5}[ ]?(?:\+|\-)?"
)


def clean_abs(args: Namespace):
    """
    Removes the word `Abstract` and its variations from the beginning of an abstract text.
    This function then overwrites the existing `.csv` files.

    Parameters:
    args - a Namespace containing the following variables which get read:

    n_samples - the number of samples to take.
    seed - the seed used for sampling.
    """

    # Setting up the regex pattern
    pattern = r"^Abstract\s?\:?\.?"

    # Define the filenames
    srg_dir = f"data/scireviewgen/subset_{args.n_samples}_{args.seed}.csv"
    srg_ref_dir = f"data/scireviewgen/srg_metadata_{args.n_samples}_{args.seed}.csv"

    # Load the files
    srg = pd.read_csv(srg_dir, index_col=False)
    srg_ref = pd.read_csv(srg_ref_dir, index_col=False)

    # Clean the abstracts
    srg["abstract"] = srg["abstract"].str.replace(pattern, "", regex=True, case=False)
    srg_ref["abstract"] = srg_ref["abstract"].str.replace(
        pattern, "", regex=True, case=False
    )

    # Save the changes
    srg.to_csv(srg_dir, index=False)
    srg_ref.to_csv(srg_ref_dir, index=False)

    # Memory Leak Prevention
    del srg, srg_ref, srg_dir, srg_ref_dir
    gc.collect()


@hydra.main(version_base=None, config_path="../configs", config_name="sample")
def main(cfg: DictConfig):

    # Loading the config
    hydra_args = cfg.get("base")
    args = Namespace(**hydra_args)

    # Removing the word `abstract`
    print("Cleaning abstracts...")
    clean_abs(args)

    print("Done!\n")


if __name__ == "__main__":

    main()
