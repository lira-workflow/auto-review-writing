import gc, re, nltk, tiktoken
import pandas as pd
from typing import List, Tuple


# Tokenizer for handling too long texts
try:
    nltk.data.find("tokenizers/punkt")

except LookupError:
    nltk.download("punkt")


def handle_entry(entry_row, key: str, dataset: str = "srg") -> List:
    """
    Extra step to cover for if the entry is still a Series.

    Parameters:
    entry_row - a pandas dataframe row.
    key - the column name to get the data from.
    dataset - the dataset currently being used.

    Returns:
    the entry's string parsed as a list.
    """

    if dataset == "srg":
        if not isinstance(entry_row[key], pd.Series):
            out_list = eval(entry_row[key])

        else:
            out_list = eval(entry_row[key].values[0])

    else:
        raise ValueError(f"Dataset `{dataset}` not supported!")

    return out_list


def format_text_from_table(
    entry_row: pd.Series, dataset: str = "srg", method: str = "base"
) -> str:
    """
    Formats the components from a dataset table into a legible text.

    Parameters:
    entry_row - a pandas dataframe row.
    dataset - the dataset currently being used.
    method - the method of article generation used.

    Returns:
    the text formatted for ROUGE evaluation.
    """

    # Pre-checks (there should only be one generated review for every human one)
    # Also checking for in case the data is for some reason loaded differently
    titles = handle_entry(entry_row, "section", dataset)
    text_parts = handle_entry(entry_row, "text", dataset)

    assert len(titles) == len(
        text_parts
    ), f"Number of titles and sections do not match: {len(titles)}, {len(text_parts)}!"

    # Getting the text
    text = f"""{entry_row["title"]}"""
    text += f"""\n\n{entry_row["abstract"]}""" if method == "auto" else ""

    # Removing the "tab"
    text = re.sub("\t", "", text)
    text = re.sub(r"[ ]{3,6}", "", text)

    for idx in range(len(titles)):

        text += "\n\n\n" + titles[idx] + "\n\n" + text_parts[idx]

    return text


# Utilities for the LLM evaluator (due to the context window limit)
def count_tokens(text: str, encoding_name: str = "gpt-3.5-turbo") -> int:
    """
    Returns the number of tokens in a given text using `tiktoken`.

    Parameters:
    text - the text to count the length of.
    encoding_name - the encoding model to use.

    Returns:
    the token length of the text.
    """

    encoding = tiktoken.encoding_for_model(encoding_name)
    return len(encoding.encode(text))


def preprocess_text(text: str) -> str:
    """
    Clean up text by removing references, URLs, and non-ASCII characters.

    Parameters:
    text - the raw text to be cleaned.

    Returns:
    the cleaned and formatted text string.
    """

    # Remove empty lines
    paragraphs = [i for i in text.split("\n") if len(i) > 0]

    # Clean section titles and remove reference section
    cleaned_paragraphs = []
    for i in paragraphs:

        if i == "# References":
            break

        if i.startswith("#"):
            i = "section: " + i.replace("#", "").strip()

        cleaned_paragraphs.append(i)

    text = "\n".join(cleaned_paragraphs)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove non-ascii characters
    text = re.sub(r"[^\x00-\x7F]+", "", text)

    # Remove citation brackets (e.g. [10])
    text = re.sub(r"\[\d+\]", "", text)

    return text


def divide_text_half(text: str) -> List[str]:
    """
    Divides a text into two parts based on punctuation (i.e., the middle-most mark ['.']).

    Parameters:
    text - the text to split.

    Returns:
    the list of texts post-splitting.
    """

    sents = nltk.tokenize.sent_tokenize(text)
    # No splitting if there's only one sentence
    # (but add another sentence as padding)
    if len(sents) <= 1:
        return [text, ""]

    mid_index = len(sents) // 2
    first = " ".join(sents[:mid_index])
    second = " ".join(sents[mid_index:])
    return [first, second]


def smart_divide_lists(
    A: List[str], B: List[str], more_space: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Maps elements from the shorter string list to the longer list as evenly as possible.

    Parameters:
    A - a list.
    B - a list.
    more_space - whether to join the sections with three newlines or two.

    Returns:
    both lists as a tuple, with the longer one having the contents merged based on the
    length of the shorter list. Note that a one-to-one mapping occurs if both lists have
    the same length.

    For example:
    A = ["a", "b", "c", "d", "d"]
    B = ["f", "g", "h", "i", "j", "k"]

    Result = (["a", "b", "c", "d", "e"], ["fg", "h", "i", "j", "k"])
    """
    if not A or not B:
        return ([], [])

    # If A and B are of the same length, no need to map
    if len(A) == len(B):
        return (A, B)

    # Determine the shorter and longer list
    short_list, long_list = (A, B) if len(A) < len(B) else (B, A)
    short_len, long_len = len(short_list), len(long_list)

    # Determine chunk sizes
    base_size, remainder = divmod(long_len, short_len)

    # Create mapping using direct slicing
    sect_map, start = [], 0
    spacing = "\n\n\n\n" if more_space else "\n\n"
    for i in range(short_len):

        chunk_size = base_size + (1 if i < remainder else 0)
        sect_map.append(spacing.join(long_list[start : start + chunk_size]))
        start += chunk_size

    # To indicate which is A and B for later
    return (short_list, sect_map) if len(A) < len(B) else (sect_map, short_list)


def split_special(short_text: str, long_text: str) -> Tuple[List, List]:
    """
    Splits the texts based on their type (either per sentence or by paragraph).

    Parameters:
    short_text - the shorter text.
    long_text - the longer text.

    Returns:
    the sentences split into a tuple of lists.
    """

    short_list = nltk.tokenize.sent_tokenize(short_text)
    long_list = long_text.split("\n\n\n\n")

    return short_list, long_list


# Lowered here because too long samples were still found
def divide_elms_in_half(
    texts_1: List[str], texts_2: List[str], max_len: int = 25_000
) -> Tuple[List[str], List[str]]:
    """
    Divides a text in half if possible and needed.

    ParametersL
    texts_1 - the first list of texts.
    texts_2 - the second list of texts.
    max_len - The maximum token length allowed for any given pair of texts from both lists.

    Returns:
    the list of texts with the too long texts split in order.
    """

    result_1, result_2 = [], []
    for text_1, text_2 in zip(texts_1, texts_2):

        if count_tokens(text_1 + " " + text_2) >= max_len:
            print("Too long text still detected. Splitting further...")
            splits_1 = divide_text_half(text_1)
            splits_2 = divide_text_half(text_2)

            assert len(splits_1) == len(splits_2)

            result_1.extend(splits_1)
            result_2.extend(splits_2)

        else:
            result_1.append(text_1)
            result_2.append(text_2)

    return result_1, result_2


def check_and_fix_lengths_texts(
    gold_list: List[str],
    pred_list: List[str],
    max_len: int = 30_000,
    gold_is_long: bool = False,
) -> Tuple[List, List]:
    """
    Splits the lists even further (involving sentences).

    Parameters:
    gold_list - the list of gold texts.
    pred_list - the list of generated texts.
    max_len - The maximum token length allowed for any given pair of texts from both lists.
    gold_is_long - whether the gold list was the one that was longer.

    Returns:
    the lists further split.
    """

    tmp_gold, tmp_pred = [], []
    for gold_text, pred_text in zip(gold_list, pred_list):

        comb_len = count_tokens(gold_text + " " + pred_text)
        if comb_len >= max_len:
            print("Too long text pair detected. Splitting further...")
            if not gold_is_long:
                short_text, long_text = gold_text, pred_text

            else:
                short_text, long_text = pred_text, gold_text

            short_split, long_split = split_special(short_text, long_text)
            short_list, long_list = smart_divide_lists(
                short_split, long_split, more_space=False
            )
            if not gold_is_long:
                gold_split, pred_split = short_list, long_list

            else:
                gold_split, pred_split = long_list, short_list

            # If the text is still too long, split even further
            gold_split, pred_split = divide_elms_in_half(gold_split, pred_split)

            tmp_gold.extend(gold_split)
            tmp_pred.extend(pred_split)

        else:
            tmp_gold.append(gold_text)
            tmp_pred.append(pred_text)

    return tmp_gold, tmp_pred


def check_and_fix_lengths(
    gold_lists: List[List[str]], pred_lists: List[List[str]], max_len: int = 30_000
) -> Tuple[List[str], List[str], List[int]]:
    """
    Additional checking if the text lengths are too long after smart division.

    Parameters:
    gold_lists - the list of gold text section lists, which are split by review article.
    pred_lists -the list of generated text section lists, which are split by review article.
    max_len - The maximum token length allowed for any given pair of texts from both lists
    (i.e., tok_len(A[0]) + tok_len(B[0])).

    Returns:
    the list of texts, controlled to be below the specified `max_len`.
    """

    final_out_gold, final_out_pred, len_list = [], [], []

    # Initial splitting
    for gold_list, pred_list in zip(gold_lists, pred_lists):

        # Up until here it's fine
        gold_is_long = any(["\n\n\n\n" in text for text in gold_list])
        tmp_gold, tmp_pred = check_and_fix_lengths_texts(
            gold_list,
            pred_list,
            max_len=max_len,
            gold_is_long=gold_is_long,
        )

        final_out_gold.extend(tmp_gold)
        final_out_pred.extend(tmp_pred)
        len_list.append(len(tmp_gold))

    # Memory Leak Prevention
    del tmp_gold, tmp_pred
    gc.collect()
    return final_out_gold, final_out_pred, len_list
