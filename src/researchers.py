import os
from typing import Dict, List, Union
from utils.misc import chunk_list
from src.base_agent import BaseAgent
from src.utils.constants import (
    LLM_SEED,
    N_JOBS,
    TEMP_DIR,
    MODEL_VER,
    MAX_ITEMS,
    GROUP_SIZE,
    TEMPERATURE,
    CONTEXT_SIZE,
    SETTING_NAME,
    ENCODING_NAME,
)
from src.prompts.research import (
    RESEARCHER_PROMPT,
    ABSTRACT_ANALYSIS_PROMPT,
    SECTION_ANALYSIS_PROMPT,
    PAPER_ANALYSIS_PROMPT,
    RETRIEVAL_PROMPT,
)

# RegEx patterns
type_regex = r"## TYPE\n(.+\n?)"
reason_regex = r"## REASON\n(.+\n?(?:\n.+){1,7})"


class ResearcherAgent(BaseAgent):
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
        split_threshold: int = 10_000,
        n_jobs: int = N_JOBS,
        setting_name: str = SETTING_NAME,
    ):

        super().__init__(
            name=name,
            model_name=model_name,
            context_size=context_size,
            seed=seed,
            temperature=temperature,
            max_memory_items=max_memory_items,
            system_prompt=RESEARCHER_PROMPT.format(REVIEW_TOPIC=review_topic),
            dataset=dataset,
            temp_dir=temp_dir,
            review_id=review_id,
            encoding_name=encoding_name,
            setting_name=setting_name,
        )
        self.split_threshold = split_threshold
        self.n_jobs = n_jobs

    def split_content(self, text: str) -> List[str]:
        """
        Splits a text fullstring into sections for later processing.

        Parameters:
        text - the text to process.

        Returns:
        a list of paper sections.
        """

        # First check if triple newlines exist (the most common pre-defined way
        # to divide sections in this repository)
        if "\n\n\n" in text:
            return text.strip().split("\n\n\n")

        # Otherwise, roughly split the content (fortunately not too many papers have this issue)
        else:
            splits = text.split("\n\n")
            splits = [split for split in splits if len(split) > 0]
            chunks = chunk_list(splits, 3)
            final = ["\n\n".join(chunk) for chunk in chunks]
            return final

    def analyze_section(
        self,
        section: str,
        paper_id: Union[int, str],
        section_num: int = 1,
    ) -> List:
        """
        Analyzes a single paper section (if not already done).

        Parameters:
        section - the section to analyze.
        paper_id - the ID of the paperto analyze.
        section_num - the number of the section.

        Returns:
        the analysis as an `AIMessage` in a list.
        """

        # Analyze a paper
        prompt = SECTION_ANALYSIS_PROMPT.format(SECTION_CONTENT=section)
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            folder_name=paper_id,
            file_name=f"{section_num}.json",
        )
        return response

    def analyze_paper(self, paper: Dict) -> List:
        """
        Analyzes a single paper/abstract (if not already done).

        Parameters:
        paper - the paper dictionary to analyze (contains the paper's ID, title, and content).

        Returns:
        the analysis as an `AIMessage` in a list.
        """

        # Format the data
        paper_id = paper["id"]
        title = paper["title"] if isinstance(paper["title"], str) else "No title"
        content = (
            paper["content"] if isinstance(paper["content"], str) else "No content"
        )
        paper_content = title + "\n\n" + content

        # Check if document splitting is needed (for better factual consistency)
        if self.count_tokens(paper_content) > self.split_threshold:
            sections = self.split_content(paper_content)
            analyses_per_section = []
            # Check if the final analysis is already present
            full_analysis_file = os.path.join(self.temp_dir, f"{paper_id}.json")
            if not os.path.isfile(full_analysis_file):
                for idx, section in enumerate(sections):

                    temp = self.analyze_section(
                        section=section,
                        paper_id=paper_id,
                        section_num=idx,
                    )
                    analyses_per_section.extend(temp)

            # Now combine into one analysis
            prompt = PAPER_ANALYSIS_PROMPT
            response = self.call_model(
                prompt=prompt,
                messages=analyses_per_section,
                response_only=True,
                file_name=f"{paper_id}.json",
            )
            return response

        else:
            # Analyze a paper
            prompt = ABSTRACT_ANALYSIS_PROMPT.format(PAPER_CONTENT=paper_content)
            response = self.call_model(
                prompt=prompt,
                response_only=True,
                file_name=f"{paper_id}.json",
            )
            return response

    def format_analyses(
        self, analyses: List[Dict], group_size: int = GROUP_SIZE
    ) -> List[Dict]:
        """
        Formats a list of papers into a text for processing.

        Parameters:
        analyses - the list of analyses provided by the researcher agent.
        group_size - the number of papers allowed in one paper group.

        Returns:
        a list of all the analyses formatted as texts.
        """

        groups = chunk_list(analyses, group_size)
        formatted = []
        for idx, analysis_group in enumerate(groups):

            out = ""
            for analysis in analysis_group:

                title, content = analysis["title"], analysis["content"]
                out += f"# TITLE\n{title}\n\n{content}\n\n\n"

            tmp = {"id": idx + 1, "content": out.strip()}
            formatted.append(tmp)

        return formatted

    # For retrieval if neded
    def add_terms(self) -> str:
        """
        Adds query terms for retrieval based on the current topic.

        Returns:
        a list of all the analyses formatted as texts.
        """

        prompt = RETRIEVAL_PROMPT
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"query.json",
        )

        # Format the response
        full_query = response[0].content.split("## TERMS\n")[-1]
        return full_query
