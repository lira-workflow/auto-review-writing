import re, torch
from typing import Dict, List, Optional
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.base_agent import BaseAgent
from src.utils.communication import LiRAState
from src.utils.constants import (
    LLM_SEED,
    TEMP_DIR,
    MODEL_VER,
    MAX_ITEMS,
    CONTEXT_SIZE,
    TEMPERATURE,
    NUM_SECTIONS,
    NUM_SUBSECTIONS,
    SECTION_LEN,
    SETTING_NAME,
    ENCODING_NAME,
)
from src.prompts.write import (
    WRITER_PROMPT,
    DRAFT_OUTLINE_PROMPT,
    MERGE_OUTLINES_PROMPT,
    FIX_OUTLINE_PROMPT,
    WRITE_TITLE_ABSTRACT_PROMPT,
    WRITE_CONTENT_PROMPT,
    WRITE_CONCLUSION_PROMPT,
    FIX_TITLE_ABSTRACT_PROMPT,
    FIX_CONTENT_PROMPT,
    FIX_CONCLUSION_PROMPT,
)


# RegEx patterns
type_pattern = r"## TYPE\n(.+)\n"
reason_pattern = r"## REASON\n(.+\n?(?:\n.+){1,7})"
citation_pattern = r"\[(.+)\]"
chain_of_thought_pattern = r"## THOUGHTS\n(?:.+\n){1,10}\n"


class OutlineDrafterAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        review_id: str,
        seed: int = LLM_SEED,
        model_name: str = MODEL_VER,
        context_size: int = CONTEXT_SIZE,
        temperature: float = TEMPERATURE,
        max_memory_items: int = MAX_ITEMS,
        review_topic: str = "Computer Science",  # The default topic for SciReviewGen
        dataset: str = "srg",
        temp_dir: str = TEMP_DIR,
        encoding_name: str = ENCODING_NAME,
        setting_name: str = SETTING_NAME,
        overwrite_responses: bool = False,
    ):

        super().__init__(
            name=name,
            model_name=model_name,
            context_size=context_size,
            seed=seed,
            temperature=temperature,
            max_memory_items=max_memory_items,
            system_prompt=WRITER_PROMPT,
            dataset=dataset,
            temp_dir=temp_dir,
            review_id=review_id,
            encoding_name=encoding_name,
            setting_name=setting_name,
            overwrite_response=overwrite_responses,
        )
        self.review_topic = review_topic

    def format_response(self, group_analysis: Dict) -> Dict:
        """
        Formats a group analysis for easier processing.

        Parameters:
        group_analysis - the group analysis response from the researcher group.

        Returns:
        a dictionary containing the type and reasoning for the selected group.
        """

        content = group_analysis[-1].content
        type_search = re.search(type_pattern, content)
        reason_search = re.search(reason_pattern, content)

        return {"type": type_search.group(1), "reason": reason_search.group(1)}

    def create_outline_single(
        self,
        paper_group: Dict,
        n_sections: int = NUM_SECTIONS,
        n_subsections: int = NUM_SUBSECTIONS,
    ) -> List:
        """
        Creates a paper outline based on the analyses and type provided.
        As multiple paper analyses groups are possible, this is performed on each group using
        separate calls.

        Parameters:
        paper_group - the paper analyses in a group.
        n_sections - the number of sections to include (functions more as a suggestion).
        n_subsections - the number of subsections to include within each section (functions more as a suggestion).

        Returns:
        the outline as an `AIMessage` in a list.
        """

        # Format the data
        group_id, paper_analyses = paper_group["id"], paper_group["content"]

        # Draft the outline
        prompt = DRAFT_OUTLINE_PROMPT.format(
            REVIEW_TOPIC=self.review_topic,
            NUM_SECTIONS=n_sections,
            NUM_SUBSECTIONS=n_subsections,
            PAPER_ANALYSES=paper_analyses,
        )
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"outline_{group_id}.json",
            folder_name="outline",
        )
        return response

    def format_outlines(self, draft_outlines: List[Dict]) -> str:
        """
        Formats a list of outlines into a string.

        Parameters:
        draft_outlines - the list of draft outline responses.

        Returns:
        the outlines combined into a string.
        """

        text = ""
        for outline in draft_outlines:

            text += outline[-1].content + "\n\n"

        return text.strip()

    def merge_outlines(
        self,
        draft_outlines: List[Dict],
        n_sections: int = NUM_SECTIONS,
        n_subsections: int = NUM_SUBSECTIONS,
        revision_count: int = 1,
    ) -> List:
        """
        Combines a collection of draft outlines

        Parameters:
        draft_outlines - the list of draft outline responses.
        n_sections - the number of sections to include (functions more as a suggestion).
        n_subsections - the number of subsections to include within each section (functions more as a suggestion).
        revision_count - the outline revision number.

        Returns:
        the outline as an `AIMessage` in a list.
        """

        # Format the data
        outlines_merged = self.format_outlines(draft_outlines)

        # Merge the outlines
        prompt = MERGE_OUTLINES_PROMPT.format(
            REVIEW_TOPIC=self.review_topic,
            NUM_SECTIONS=n_sections,
            NUM_SUBSECTIONS=n_subsections,
            OUTLINES=outlines_merged,
        )

        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"merged_outline_{revision_count}.json",
            folder_name="outline",
        )
        return response

    def fix_outline(
        self,
        messages: List,
        draft_outlines: List[Dict],
        n_sections: int = NUM_SECTIONS,
        n_subsections: int = NUM_SUBSECTIONS,
        revision_count: int = 1,
    ) -> List:
        """
        Fixes an outline using the review history.

        Parameters:
        messages - the messages for the current discussion (as its own list for ease of processing).
        draft_outlines - the list of draft outline responses.
        n_sections - the number of sections to include (functions more as a suggestion).
        n_subsections - the number of subsections to include within each section (functions more as a suggestion).
        revision_count - the outline revision number.

        Returns:
        the new response containing the list of messages which includes the updated outline.
        """

        # Format the data
        outlines_merged = self.format_outlines(draft_outlines)

        prompt = FIX_OUTLINE_PROMPT.format(
            REVIEW_TOPIC=self.review_topic,
            NUM_SECTIONS=n_sections,
            NUM_SUBSECTIONS=n_subsections,
            OUTLINES=outlines_merged,
        )
        response = self.call_model(
            prompt=prompt,
            messages=messages,
            response_only=False,
            file_name=f"merged_outline_{revision_count}.json",
            folder_name="outline",
        )
        return response


class ContentWriterAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        review_id: str,
        seed: int = LLM_SEED,
        model_name: str = MODEL_VER,
        context_size: int = CONTEXT_SIZE,
        temperature: float = TEMPERATURE,
        max_memory_items: int = MAX_ITEMS,
        review_topic: str = "Computer Science",  # The default topic for SciReviewGen
        dataset: str = "srg",
        temp_dir: str = TEMP_DIR,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        setting_name: str = SETTING_NAME,
        overwrite_responses: bool = False,
    ):

        super().__init__(
            name=name,
            model_name=model_name,
            context_size=context_size,
            seed=seed,
            temperature=temperature,
            max_memory_items=max_memory_items,
            system_prompt=WRITER_PROMPT,
            dataset=dataset,
            temp_dir=temp_dir,
            review_id=review_id,
            setting_name=setting_name,
            overwrite_response=overwrite_responses,
        )
        self.review_topic = review_topic
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
        )
        self.vectorstore = None

    def parse_outline(self, outline: str) -> Dict:
        """
        Parses an outline into sections and descriptions.

        Parameters:
        outline - the outline to parse.

        Returns:
        a dictionary containing the lists of headers, descriptions, the present references, and
        the name of the section containing the subsection if applicable.
        """

        # Remove the Chain-of-Thought section
        outline = re.sub(chain_of_thought_pattern, "", outline)

        # Split the sections
        parts = re.split(r"#{1,2} ", outline.strip())

        # Get the subsection headers
        headers = re.findall(r"## .+\n", outline.strip())
        headers = [re.sub("## ", "", h.strip()) for h in headers]

        # Parse each section
        out = {
            "headers": [],
            "descriptions": [],
            "references": [],
            "header_higher_level": [],
        }
        for part in parts:

            # Skip empty parts in case
            if len(part) <= 5:
                continue

            # Get tßhe header and content
            splitted = part.split("\n")
            header, content = splitted[0], splitted[1]
            content = re.sub(r"[ ]?\[.+\]", "", content)  # Remove references
            references = re.findall(citation_pattern, content)

            new_references = []
            # Split the references for when two are used in one
            for reference in references:

                processed_ref = re.sub(r"\[|\]", "", reference)
                processed_ref = re.split(r"\|", processed_ref)
                new_references.extend(processed_ref)

            out["headers"].append(header.strip())
            out["descriptions"].append(content.strip())
            out["references"].append(new_references)

            # Append the main section header name
            if header not in headers:
                header_higher = header
                out["header_higher_level"].append("None")

            else:
                out["header_higher_level"].append(header_higher)

        return out

    def create_db(self, state: LiRAState):
        """
        Creates a database for paper analysis retrieval based.

        Parameters:
        state - the worfklow's current state.
        """

        analyses = state["researcher_analyses"]
        documents = []
        for analysis in analyses:

            content = (
                "TITLE:"
                + analysis["title"]
                + "\n\nCONTENT:"
                + analysis["content"][-1].content
            )
            documents.append(
                Document(page_content=content, metadata={"id": analysis["id"]})
            )

        self.vectorstore = FAISS.from_documents(
            documents, embedding=self.embedding_model
        )

    def get_references(
        self, state: LiRAState, description: str, references: List
    ) -> str:
        """
        Retrieves the appropriate references for a section.

        Parameters:
        state - the worfklow's current state.
        description - the description of the section without references.
        references - the list of references listed for the subsection to use.

        Returns:
        the references as a string.
        """

        # Count how many references are needed
        len_references_total = len(state["researcher_analyses"])
        n_rag = len_references_total // 4 if len_references_total // 4 > 3 else 3

        # In rare cases where there are too many references, we cutoff at 150
        n_rag = n_rag if n_rag <= 150 else 150
        reference_analyses = []
        if len(references) > 0:
            for reference in references:

                analysis_single = self.vectorstore.similarity_search(
                    reference,
                    k=1,
                )
                reference_analyses.append(analysis_single[0].page_content)

        # Add more for extra context
        other_results = self.vectorstore.similarity_search(
            description,
            k=n_rag,
        )

        idx = 0
        while len(reference_analyses) < n_rag:

            try:
                content = other_results[idx].page_content
                reference_analyses.append(content)

            except IndexError:
                break

            idx += 1

        # Format the references
        ref_descriptions = "\n\n".join(reference_analyses)
        return ref_descriptions

    def write_content_section(
        self,
        state: LiRAState,
        header: str,
        description: str,
        references: List[str],
        header_high_level: str = "None",
        revision_count: int = 1,
        section_len: int = SECTION_LEN,
    ) -> Optional[str]:
        """
        Writes the review content for one section based on the outline header and analysis contents.

        Parameters:
        state - the worfklow's current state.
        header - the (sub)section's title.
        description - the description of the section without references.
        references - the list of references listed for the subsection to use.
        header_high_level - the subsection's main section header if this is a subsection.
        revision_count - the content revision number.
        section_len - the number of words to include in each section.

        Returns:
        the section content as an `AIMessage` in a list.
        """

        # Set the save name
        save_name = re.sub(" ", "_", header.lower())

        # Skip if the section is a conclusion
        if "conclusion" in header.lower() and "in" not in header.lower():
            return None

        # Get all the references listed
        ref_descriptions = self.get_references(state, description, references)

        # Finally, prompt the system
        prompt = WRITE_CONTENT_PROMPT.format(
            SECTION_TITLE=header,
            SECTION_DESCRIPTION=description,
            HEADER_HIGH_LEVEL=header_high_level,
            PAPER_ANALYSES=ref_descriptions,
            SECTION_LEN=section_len + 100,  # The 100 functions as a buffer
        )

        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"{save_name}.json",
            folder_name=f"section_{revision_count}",
        )
        return response

    def revise_content_section(
        self,
        state: LiRAState,
        header: str,
        description: str,
        references: List[str],
        header_high_level: str = "None",
        revision_count: int = 2,
        section_len: int = SECTION_LEN,
    ) -> List:
        """
        Writes the review content for one section based on the outline header and analysis contents.

        Parameters:
        state - the worfklow's current state.
        header - the (sub)section's title.
        description - the description of the section without references.
        references - the list of references listed for the subsection to use.
        header_high_level - the subsection's main section header if this is a subsection.
        revision_count - the content revision number.
        section_len - the number of words to include in each section.

        Returns:
        the section content as an `AIMessage` in a list.
        """

        # Set the save name
        save_name = re.sub(" ", "_", header.lower())

        # Skip if the section is a conclusion (this is done later)
        if "conclusion" in header.lower() and "in" not in header.lower():
            return None

        # Get all the references listed
        ref_descriptions = self.get_references(state, description, references)
        discussion = state["content_discussion"][header]

        # Finally, prompt the system
        prompt = FIX_CONTENT_PROMPT.format(
            SECTION_TITLE=header,
            SECTION_DESCRIPTION=description,
            HEADER_HIGH_LEVEL=header_high_level,
            PAPER_ANALYSES=ref_descriptions,
            SECTION_LEN=section_len + 100,  # The 100 functions as a buffer
        )

        response = self.call_model(
            prompt=prompt,
            messages=discussion,
            response_only=True,
            file_name=f"{save_name}.json",
            folder_name=f"section_{revision_count}",
        )

        # Remove the Chain-of-Thought part
        response[-1].content = re.sub(
            chain_of_thought_pattern, "", response[-1].content
        )
        return response

    def write_title_abstract(
        self,
        paper_body: str,
        revision_count: int = 1,
        feedback: bool = False,
    ) -> str:
        """
        Writes a literature review abstract based on the contents of the main paper.

        Parameters:
        paper_body - the main content of the literature review.
        revision_count - the content revision number.
        feedback - whether feedback for the section has already been provided.

        Returns:
        the title and abstract as an `AIMessage` in a list.
        """

        # Use the feedback if available
        if feedback:
            prompt = FIX_TITLE_ABSTRACT_PROMPT.format(REVIEW_CONTENT=paper_body)
            response = self.call_model(
                prompt=prompt,
                response_only=True,
                file_name="title_abstract.json",
                folder_name=f"section_{revision_count}",
            )
            response[-1].content = re.sub(
                chain_of_thought_pattern, "", response[-1].content
            )
            return response

        prompt = WRITE_TITLE_ABSTRACT_PROMPT.format(REVIEW_CONTENT=paper_body)
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name="title_abstract.json",
            folder_name=f"section_{revision_count}",
        )
        return response

    def write_conclusion(
        self,
        state: LiRAState,
        paper_body: str,
        conc_title: str = "Conclusion",
        revision_count: int = 1,
        feedback: bool = False,
    ) -> List:
        """
        Writes a literature review conclusion based on the contents of the main paper.

        Parameters:
        state - the worfklow's current state.
        paper_body - the main content of the literature review.
        conc_title - the conclusion's title (perhaps it's a variation such as "Conclusion and Future Work").
        revision_count - the content revision number.
        feedback - whether feedback for the section has already been provided.

        Returns:
        the conclusion as an `AIMessage` in a list.
        """

        # Use the feedback if available
        if feedback:
            prompt = FIX_CONCLUSION_PROMPT.format(
                REVIEW_CONTENT=paper_body, CONCLUSION_TITLE=conc_title
            )
            response = self.call_model(
                prompt=prompt,
                messages=state["content_discussion"][conc_title],
                response_only=True,
                file_name=f"{conc_title}.json",
                folder_name=f"section_{revision_count}",
            )
            response[-1].content = re.sub(
                chain_of_thought_pattern, "", response[-1].content
            )
            return response

        prompt = WRITE_CONCLUSION_PROMPT.format(REVIEW_CONTENT=paper_body)
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"{conc_title}.json",
            folder_name=f"section_{revision_count}",
        )
        return response
