import re, json
from typing import Dict, List


def read_file(filename: str) -> str:
    """
    Tries to read a `.txt` file and return its contents.

    Parameters:
    filename - the filename (directory) to read from.

    Returns:
    the file contents as text if present, else returns an error.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:

            content = f.read()

        return content

    except FileNotFoundError:
        print(f"Error: The file `{filename}` was not found.")


def save_to_file(response: str, output_file: str, verbose: bool = False):
    """
    Saves a file using the desired name in a similar directory.

    Parameters:
    response - the content to save.
    output_file - the target file name.
    verbose - whether to print the filenames saved or not.

    Returns:
    the filename of the new output file generated.
    """

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(response)

    if verbose:
        print(f"Successfully created `{output_file}`!")

    return output_file


def load_json(input_file: str) -> Dict:
    """
    Reads a baseline review file and returns it as its corresponding type.

    Parameters:
    input_file - the file containing the data.

    Returns:
    the contents as a dictionary.
    """

    with open(input_file, "r") as f:

        data = json.load(f)

    return dict(data)


def save_to_json(content: Dict, output_file: str):
    """
    Saves a baseline review file as a JSON for quick loading.

    Parameters:
    content - the dictionary of data to save.
    output_file - the file to output to.
    """

    with open(output_file, "w+") as f:

        json.dump(content, f, indent=2)


def save_ref_as_txt(references: List[dict], output_file: str):
    """
    Writes reference list to an output .txt file.

    Parameters:
    references - list as outputted by the "extract_references" function above.
    output_file - the target file name for saving the "references" list.
    """

    with open(output_file, "w+", encoding="utf-8") as f:

        for ref in references:

            f.write(f'Number: {ref["num"]}\n')
            f.write(f'Title: {ref["title"]}\n')
            f.write(f'Abstract: {ref["abstract"]}\n')
            f.write("\n")


def pattern_selector(
    dataset: str, input_suffix: str, extension: str = "txt", use_group: bool = False
) -> str:
    """
    Adjusts the regex pattern to use based on the dataset selected.

    Parameters:
    dataset - the dataset abbreviation.
    input_suffix - the suffix to use for processing.
    extension - the file extension to use.
    use_group - whether not to add the first subgroup in the regex string.

    Returns:
    the formatted regex string for filename matching.
    """

    if dataset == "srg":
        id = "\d+"
        body = "_subset_\d+_\d+"

    else:
        raise ValueError(f"`{dataset}` not supported!")

    if use_group:
        id = f"({id})"

    return rf"{id}{body}{input_suffix}\.{extension}$"


def check_paper_in_subset(
    filenames: List, target_n_samples: int, target_seed: int
) -> List:
    """
    Utility function to check if a paper within a folder is actually part of the
    desired subset.

    Parameters:
    filename - the filename to check (must contain the paper corpus ID or other
    identifier in the start).
    target_n_samples - the number of samples in the target subset.
    target_seed - the seed of the target subset sampler.

    Returns:
    a list of files which are in the subset table.
    """

    # Initialize a list to hold the full filenames of the matching entries
    matching_filenames = []

    # Loop over the filenames and extract the corpusId from each
    for filename in filenames:

        match = re.search(r"\d+_subset_(\d+)_(\d+)", filename)
        if match:
            # Select only files that match the group
            n_samples = int(match.group(1))
            seed = int(match.group(2))

            if n_samples == target_n_samples and seed == target_seed:
                matching_filenames.append(filename)

    return matching_filenames
