import json, tiktoken
from typing import Dict, List, Union
from numpyencoder import NumpyEncoder


class tokenCounter:

    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        # The pricing has been adjusted for every one million tokens
        # See https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/ for more info.
        self.model_price = {
            "gpt-4o-2024": [2.37948, 9.5180],
            "gpt4o-mini-240718": [0.14277, 0.5711],
        }

    def num_tokens_from_string(self, string: str) -> int:
        return len(self.encoding.encode(string))

    def num_tokens_from_list_string(self, list_of_string: List[str]) -> int:
        num = 0
        for s in list_of_string:

            num += len(self.encoding.encode(s))

        return num

    def compute_price(self, input_tokens: int, output_tokens: int, model: str) -> float:
        # The pricing has been adjusted for every one million tokens
        return (input_tokens / 1_000_000) * self.model_price[model][0] + (
            output_tokens / 1_000_000
        ) * self.model_price[model][1]

    def text_truncation(self, text: str, max_len: int = 1000) -> str:
        encoded_id = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(encoded_id[: min(max_len, len(encoded_id))])


# Same function as used in `baseline_and_mass/utils/text_utils.py`
def check_length(
    string: str,
    max_len: int = 1_048_576,
    verbose: bool = False,
) -> bool:
    """
    Checks if the number of tokens exceeds a certain pre-defined amount.

    Parameters:
    string - the string to count the length of.
    token_limit - the string length to use as a limit.
    The default value is the input context window of ChatGPT 4o (and 4o-mini).
    verbose - whether to print if the limit is exceeded.

    Returns:
    True if the `max_len` is exceeded, else False.
    """

    if len(string) > max_len:
        if verbose:
            print(f"The string length exceeds {max_len}!")

        return True

    return False


def load_txt(file: str, max_len: int = 1_500, use_truncation: bool = True) -> str:
    """
    Loads a txt file up to a given length.

    Parameters:
    file - the file to parse.
    max_len - the maximum number of tokens to return.
    use_truncation - whether to truncate the output text.

    Return:
    the contents of the file parsed up to the `max_len`.
    """

    counter = tokenCounter()
    with open(file, "r") as f:
        # Remove trailing whitespaces
        text = f.read().strip()

    if use_truncation:
        return counter.text_truncation(text, max_len)

    else:
        return text


def save_to_txt(response: str, output_file: str, verbose: bool = False):
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


# Below functions are the same as in `baseline_and_mass/utils/file_utils.py`
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
    Saves a baseline review file as a JSON for quick loading.

    Parameters:
    content - the dictionary of data to save.
    output_file - the file to output to.
    use_numpy_encoder - whether to use `NumpyEncoder` for saving numpy arrays to JSON.
    (used for creating the custom database objects).
    """

    cls = NumpyEncoder if use_numpy_encoder else None
    with open(output_file, "w+") as f:

        json.dump(content, f, indent=2, cls=cls)
