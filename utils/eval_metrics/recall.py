# Functions for evaluating the pipeline outputs using heading soft recall and heading entity recall
# (adapted from https://github.com/stanford-oval/storm/tree/NAACL-2024-code-backup)
import sys

sys.path.append(".")

import gc, spacy
from re import split
from tqdm import tqdm
from argparse import Namespace
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from utils.pbar_utils import tqdm_joblib
from utils.eval_metrics.eval_constants import ROUNDING
from sklearn.metrics.pairwise import cosine_similarity

# Loading the base classifiers and encoder
spacy_tagger = spacy.load(
    "en_core_sci_lg"
)  # Required for the life and health (maybe physical) scientific fields
encoder = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def card(s: set) -> float:
    """
    Function to calculate the cardinality of a set 's'.
    Uses the embedding cosine similarities, as defined in https://aclanthology.org/2024.naacl-long.347/.

    Parameters:
    s - the set to calculate the cardinality over.

    Returns:
    the soft count as a float.
    """

    encoded_s = encoder.encode(list(s))
    cosine_sim = cosine_similarity(encoded_s)
    soft_count = 1 / cosine_sim.sum(axis=1)
    result = float(soft_count.sum())

    # Memory Leak Prevention
    del encoded_s, cosine_sim, soft_count
    gc.collect()
    return result


def heading_soft_recall(
    gold_headings: List[str], pred_headings: List[str], rounding: int = ROUNDING
) -> float:
    """
    Computes the soft heading recall for a set of predicted headings.

    Parameters:
    gold_headings - the golden headings to use as reference.
    pred_headings - the heading predictions outputted by the automated pipeline.
    rounding - the rounding for adjusting the final result output.

    Returns:
    the soft heading recall as a float.
    """

    g, p = set(gold_headings), set(pred_headings)
    if len(p) == 0:
        return 0.0

    card_g, card_p = card(g), card(p)
    card_intersection = card_g + card_p - card(g.union(p))
    output = round(card_intersection / card_g, rounding)

    # Memory Leak Prevention
    del g, p, card_g, card_p
    gc.collect()
    return output


def extract_entities_from_list(l: List[str]) -> List[str]:
    """
    Extracts the entities found in a list of headings.

    Parameters:
    l - the list of headings to extract from.

    Returns:
    the list of all people names (strings), lowercased.
    """

    entities = []
    for sent in spacy_tagger.pipe(
        l,
        disable=[
            "tok2vec",
            "tagger",
            "parser",
            "attribute_ruler",
            "lemmatizer",
        ],  # Disable the other components for speed
    ):

        if len(sent) == 0:
            continue

        entities.extend([e.text for e in sent.ents])

    output = list(set([e.lower() for e in entities]))

    # Memory Leak Prevention
    del entities
    gc.collect()
    return output


def heading_entity_recall(
    gold_entities: Optional[List[str]] = None,
    pred_entities: Optional[List[str]] = None,
    gold_headings: Optional[List[str]] = None,
    pred_headings: Optional[List[str]] = None,
    rounding: int = ROUNDING,
):
    """
    Computes the heading entity recall for a list of predicted headings.

    Parameters:
    gold_entities - list of strings or None; if None, extract from gold_headings.
    pred_entities - list of strings or None; if None, extract from pred_headings.
    gold_headings - list of strings or None.
    pred_headings - list of strings or None.
    rounding - the rounding for adjusting the final result output.

    Returns:
    the soft heading recall as a float.
    """

    if gold_entities is None:
        assert (
            gold_headings is not None
        ), "gold_headings and gold_entities cannot both be None."
        gold_entities = extract_entities_from_list(gold_headings)

    if pred_entities is None:
        assert (
            pred_headings is not None
        ), "pred_headings and pred_entities cannot both be None."
        pred_entities = extract_entities_from_list(pred_headings)

    g, p = set(gold_entities), set(pred_entities)
    out = 1.0 if (len(g) == 0) else round(len(g.intersection(p)) / len(g), rounding)

    # Memory Leak Prevention
    del g, p
    gc.collect()
    return out


def article_entity_recall(
    gold_entities: Optional[List[str]] = None,
    pred_entities: Optional[List[str]] = None,
    gold_article: Optional[str] = None,
    pred_article: Optional[str] = None,
    rounding: int = ROUNDING,
):
    """
    Computes the article entity recall for a list of predicted entities.

    Parameters:
    gold_entities - list of strings or None; if None, extract from gold_article.
    pred_entities - list of strings or None; if None, extract from pred_article.
    gold_article - string or None.
    pred_article - string or None.
    rounding - the rounding for adjusting the final result output.

    Returns:
    the article entity recall as a float.
    """

    if gold_entities is None:
        assert (
            gold_article is not None
        ), "gold_article and gold_entities cannot both be None."
        sentences = split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", gold_article)
        gold_entities = extract_entities_from_list(sentences)

    if pred_entities is None:
        assert (
            pred_article is not None
        ), "pred_article and pred_entities cannot both be None."
        sentences = split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", pred_article)
        pred_entities = extract_entities_from_list(sentences)

    g, p = set(gold_entities), set(pred_entities)
    out = 1.0 if (len(g) == 0) else round(len(g.intersection(p)) / len(g), rounding)

    # Memory Leak Prevention
    del g, p
    gc.collect()
    return out


def calc_recall(
    t_entry: Tuple, h_entry: Tuple, rounding: int = 3
) -> Tuple[float, float, float]:
    """
    Helper function to calculate the recall metrics for singular text and heading pairs.

    Parameters:

    t_entry - a tuple containing the texts for an entry (gold first, prediction second).
    h_entry - a tuple containing the header(s) for an entry (gold first, prediction second).
    Note that these entries should correspond to the same article.
    rounding - the rounding to use for result formatting.

    Returns:
    the heading soft recall, heading entity recall, and article entity recall for the entry.
    """

    gold_text, pred_text = t_entry
    gold_headings, pred_headings = h_entry
    hsr = heading_soft_recall(
        gold_headings=gold_headings,
        pred_headings=pred_headings,
        rounding=rounding,
    )
    her = heading_entity_recall(
        gold_headings=gold_headings,
        pred_headings=pred_headings,
        rounding=rounding,
    )
    aer = article_entity_recall(
        gold_article=gold_text, pred_article=pred_text, rounding=rounding
    )

    # Memory Leak Prevention
    del gold_text, pred_text, gold_headings, pred_headings
    return hsr, her, aer


def calc_recall_full(
    text_pairs: List[Tuple],
    heading_pairs: List[Tuple],
    args: Namespace,
) -> Dict:
    """
    The function for calculating the heading-related recall metrics for all text and heading pair sets.


    Parameters:
    text_pairs - a tuple containing the texts for an entry (gold first, prediction second).
    heading_pairs - a tuple containing the header(s) for an entry (gold first, prediction second).
    Note that these entries should correspond to the same article.
    args - a Namespace containing the following variables which get read:

    rounding - the rounding to use for result formatting.
    n_jobs - the number of workers to use for parallelization.

    Returns:
    the heading soft recall, heading entity recall, and article entity recall result lists.
    See the final line for more info on how the output looks.
    """

    # Result placeholders
    hsr_list, her_list, aer_list = [], [], []

    # Parallelize heading recall calculations
    calc_heading_pairs = list(zip(text_pairs, heading_pairs))
    with tqdm_joblib(
        tqdm(
            desc="Calculating Recall Metrics",
            unit="file pair(s)",
            total=len(calc_heading_pairs),
        )
    ):
        heading_results = Parallel(args.n_jobs, verbose=0)(
            delayed(calc_recall)(t_entry, h_entry, args.rounding)
            for t_entry, h_entry in calc_heading_pairs
        )

    # Aggregate the heading recall results
    for hsr, her, aer in heading_results:

        # Check the heading soft recall in case of mistakes
        # (It should always return 0-1)
        if hsr > 1.000:
            hsr = 1.000

        elif hsr < 0.000:
            hsr = 0.000

        hsr_list.append(hsr)
        her_list.append(her)
        aer_list.append(aer)

    # Memory Leak Prevention
    del text_pairs, heading_pairs
    gc.collect()

    # Return this dictionary if `return_all`
    return {"hsr": hsr_list, "her": her_list, "aer": aer_list}
