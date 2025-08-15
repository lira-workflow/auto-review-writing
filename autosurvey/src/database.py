import faiss, torch, operator
import numpy as np
from tinydb import TinyDB, Query
from typing import Dict, List, Tuple, Union
from autosurvey.src.utils import tokenCounter
from sentence_transformers import SentenceTransformer


class database:

    def __init__(
        self,
        db_dir: str,
        embedding_model: str = "nomic-ai/nomic-embed-text-v1",
        fulltext: bool = False,
    ):
        self.embedding_model = SentenceTransformer(
            embedding_model, trust_remote_code=True
        )

        # Automate device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model.to(torch.device(self.device))

        db_loc = f"{db_dir}/db.json" if not fulltext else f"{db_dir}/db_fulltext.json"
        self.db = TinyDB(db_loc)
        self.table = self.db.table("_default")

        self.User = Query()
        self.token_counter = tokenCounter()

    def get_embeddings(self, batch_text: List[str], mode: str = "query") -> np.array:
        """
        Generates the embeddings for a list of texts.

        Parameters:
        batch_text - the texts to embed.
        mode - the "mode" to use for searching.
        It is used for determining how the embeddings should appear
        (i.e., more geared towards encoding as a "query" or "document". These are the possible values).

        Returns:
        the embedding as a numpy array.
        """

        batch_text = [f"search_{mode.lower()}: " + _ for _ in batch_text]
        embeddings = self.embedding_model.encode(batch_text)
        return embeddings

    def batch_search(
        self,
        query_vectors: np.array,
        review_id: Union[int, str, None],
        top_k: int = 1,
        title: bool = False,
    ) -> List[List]:
        """
        Searches the database using multiple query vectors at the same time.
        This is mainly used for searching the title of a reference given the citing sentence/citation.

        Parameters:
        query_vectors - the query vectors returned by the `get_embeddings` function for a list of terms.
        review_id - the ID of the original review paper used as the basis (for filtering references).
        top_k - the number of entries to return per citation entry (if too large then defaults to around 25% of the
        total documents after filtering).
        title - whether to do a vector search based on the title or the abstracts.

        Returns:
        the list of IDs for each citation sentence.
        """

        # First get the vectors and mappings
        vectors, id_mappings = self.format_vector_and_mapping(review_id, title)

        # Then determine how many to query based on amount
        # (between either the number specified or around 25% of the documents needed)
        top_n = min(top_k, len(id_mappings) // 4)
        top_n = 2 if top_n <= 0 else top_n  # in case the `top_n` is zero

        # Build the temporary FAISS index
        index = self.build_faiss_index(vectors)

        # Now we search
        query_vectors = np.array(query_vectors).astype("float32")
        _, indices = index.search(query_vectors, top_n)  # Get top_n results

        # ...and obtain the indexes! (this returns a tuple, hence the list casting later.)
        results = []
        for i in range(len(query_vectors)):

            result = [
                id_mappings[idx]
                for idx in indices[i]
                if idx != -1 and idx < len(id_mappings)
            ]
            results.append(result)

        # Memory Leak Prevention
        del index, indices, vectors, id_mappings, query_vectors
        return results

    def search(
        self,
        query_vector: np.array,
        review_id: Union[int, str, None] = None,
        top_k: int = 1,
        title: bool = False,
    ) -> List:
        """
        Searches the database using a single query vector.

        Parameters:
        query_vector - the query vector returned by the `get_embeddings` function for one search term.
        review_id - the ID of the original review paper used as the basis (for filtering references).
        top_k - the number of entries to return per citation entry (if too large then defaults to around 25% of the
        total documents after filtering).
        title - whether to do a vector search based on the title or the abstracts.

        Returns:
        the list of relevant document IDs for the query vector.
        """

        # First get the vectors and mappings
        vectors, id_mappings = self.format_vector_and_mapping(review_id, title)

        # Then determine how many to query based on amount
        # (between either the number specified or around 25% of the documents needed)
        top_n = min(top_k, len(id_mappings) // 4)
        top_n = 2 if top_n <= 0 else top_n  # in case the `top_n` is zero

        # Build the temporary FAISS index
        index = self.build_faiss_index(vectors)

        # Now we search
        query_vector = np.array([query_vector]).astype("float32")
        _, indices = index.search(query_vector, top_n)  # Get top_n results

        # ...and obtain the indexes! (this returns a tuple, hence the list casting later.)
        results = operator.itemgetter(*indices[0])(id_mappings)

        # Memory Leak Prevention
        del index, indices, vectors, id_mappings, query_vector
        results = [results] if isinstance(results, int) else list(results)
        return results

    def format_vector_and_mapping(
        self, review_id: Union[int, str, None], title: bool = False
    ) -> Tuple[List, List]:
        """
        Filters the documents in the database and then creates the vector and ID mapping lists based on them.

        Parameters:
        review_id - the ID of the original review paper used as the basis (for filtering references).
        title - whether to do a vector search based on the title or the abstracts.
        Otherwise returns all the documents.

        Returns:
        the lists of vectors and ID mappings.
        """

        # First obtain the (filtered) documents
        docs = (
            self.db.search(self.User.review_id == review_id)
            if review_id is not None
            else self.db.all()
        )

        # Then get the vectors and id_mappings (as a link to the original file)
        vectors, id_mappings = [], []
        for doc in docs:
            vectors.append(doc["title_embed"] if title else doc["abs_embed"])
            id_mappings.append(doc["id"])

        # Memory Leak Prevention
        del docs

        return vectors, id_mappings

    def get_ids_from_query(
        self, query: str, review_id: Union[int, str, None] = None, num: int = 5
    ) -> List[Union[int, str]]:
        """
        Gets the relevant document ID based on a query.

        Parameters:
        query - the search query (i.e., topic).
        review_id - the ID of the original review paper used as the basis (for filtering references).
        num - the number of results to return.

        Returns:
        a list of IDs for the documents relevant to the query.
        """

        q = self.get_embeddings([query])[0]
        return self.search(q, review_id=review_id, top_k=num)

    def get_titles_from_citations(
        self, citations: List[str], review_id: Union[int, str, None]
    ) -> List[Union[int, str]]:
        """
        Gets the relevant document IDs based on a list of citing sentences/citations.

        Parameters:
        citations - the list of citations.
        review_id - the ID of the original review paper used as the basis (for filtering references).

        Returns:
        a list of IDs belonging to the original documents of the citations.
        """

        q = self.get_embeddings(citations, mode="document")
        ids = self.batch_search(q, review_id, top_k=1, title=True)

        # Memory Leak Prevention
        del q
        return [id[0] for id in ids]

    def get_paper_info_from_ids(self, ids: List[Union[int, str]]) -> List[Dict]:
        """
        Gets the paper information given a set of IDs.

        Parameters:
        ids - the list of paper IDs to retrieve.

        Returns:
        the list of papers corresponding to the IDs.
        """

        return self.table.search(self.User.id.one_of(ids))

    def build_faiss_index(self, vectors: List[List]):
        """
        Creates a FAISS index given a list of vectors.

        Parameters:
        vectors - the list of vectors to convert.

        Returns:
        a FAISS index containing the vectors as entries.
        """

        d = len(vectors[0])
        index = faiss.IndexFlatL2(d)
        vectors_np = np.array(vectors, dtype=np.float32)
        index.add(vectors_np)

        # Memory Leak Prevention
        del vectors_np
        return index
