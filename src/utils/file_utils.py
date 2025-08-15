# These functions are mostly the same as in `utils/misc`
import json
from typing import Dict, List, Union


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


def save_to_file(response: str, output_file: str):
    """
    Saves a file using the desired name in a similar directory.

    Parameters:
    response - the content to save.
    output_file - the target file name.

    Returns:
    the filename of the new output file generated.
    """

    with open(output_file, "w", encoding="utf-8") as f:

        f.write(response)


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

    with open(output_file, "w+") as f:

        json.dump(content, f, indent=2)
