import json
from itertools import islice
from typing import Dict, List, Union
from numpyencoder import NumpyEncoder


def chunk_list(lst: List[str], n: int) -> List:
    """
    Splits a list into chunks of size `n`.

    Parameters:
    lst - the list of strings.
    n - the chunk size.

    Returns:
    a list containing the chunked lists.
    """

    return [list(islice(lst, i, i + n)) for i in range(0, len(lst), n)]


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


def read_file(filename: str) -> str:
    """
    Reads a baseline review file and returns it as text.

    Parameters:
    filename - the file containing the data.

    Returns:
    the contents as a string.
    """

    with open(filename, "r") as f:

        content = f.read()

    return content


# This function is the same as in `baseline_and_mass/utils/file_utils.py`
def load_json(filename: str) -> Union[Dict, List]:
    """
    Reads a baseline review file and returns it as its corresponding type.

    Parameters:
    input_file - the file containing the data.

    Returns:
    the contents as a list of dictionaries or a dictionary, depending on the
    formatting used.
    """

    with open(filename, "r") as f:

        data = json.load(f)

    return data


def save_to_json(
    content: Union[Dict, List], output_file: str, use_numpy_encoder: bool = False
):
    """
    Saves a dictionary to a JSON file.

    Parameters:
    content - the dictionary of data to save.
    output_file - the file to output to.
    use_numpy_encoder - whether to use `NumpyEncoder` for saving numpy arrays to JSON.
    (used for creating the custom database objects).
    """

    cls = NumpyEncoder if use_numpy_encoder else None
    with open(output_file, "w+") as f:

        json.dump(content, f, indent=2, cls=cls)
