import os, json, time
from tqdm import tqdm
from typing import Dict, List
from src.researchers import ResearcherAgent
from src.utils.file_utils import load_json, save_to_json


class APIClass:
    """A client for the Internal API.
    Used for the retrieval setting (details removed for anonymization)."""

    def __init__(self):
        self.name = "DocumentRetriever"

    def simple_query(
        self,
        query,
        amount,
    ):

        final_out = (
            """A list of documents retrieved using embedding search via an API."""
        )
        return final_out


def query_api(
    papers_list: List[Dict],
    data_dir: str,
    dataset: str,
    seed: int = 42,
    temperature: float = 0.0,
) -> List[Dict]:
    """
    Queries the API for paper titles and abstracts.
    This was included only for completeness and does not work for SciReviewGen.

    Parameters:
    papers_list - the list of papers and references from the original dataset.
    data_dir - the location to load/save the results.
    dataset - the dataset being used.
    seed - the seed used for the LLM (if supported).
    temperature - the output temperature (higher value results in more response variance).

    Returns:
    the papers retrieved by the API in a similar format
    """

    # Create the folder
    retrieval_folder = os.path.join(data_dir, "retrieval")
    os.makedirs(retrieval_folder, exist_ok=True)
    retrieval_file = os.path.join(retrieval_folder, f"{dataset}.json")
    if os.path.isfile(retrieval_file):
        new_papers_list = load_json(retrieval_file)

    else:
        doc_api = APIClass()
        new_papers_list = []
        for idx, paper in tqdm(
            enumerate(papers_list),
            total=len(papers_list),
            desc="Retrieving documents...",
        ):

            topic = paper["topic"]
            amount = len(paper["references"])
            amount = (
                amount if amount <= 499 else 499
            )  # Because of an article limit with the retriever
            # Enrich the query with relevant terms
            researcher_agent = ResearcherAgent(
                "researcher",
                review_id=paper["id"],
                review_topic=topic,
                temp_dir="temp_analysis/query_enrichment",
                dataset=dataset,
                seed=seed,
                temperature=temperature,
            )
            enriched_query = researcher_agent.add_terms()
            results = doc_api.simple_query(query=enriched_query, amount=amount)

            # Remove the title from the results
            results = list(
                filter(lambda x: x["title"][0].lower() != topic.lower(), results)
            )
            new_papers_list.append(
                {
                    "num": idx + 1,
                    "id": paper["id"],
                    "topic": paper["topic"],
                    "references": results,
                }
            )

        save_to_json(new_papers_list, retrieval_file)

    return new_papers_list
