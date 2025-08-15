import gc, os
from tqdm import trange
from typing import Dict, List, Tuple, Union
from autosurvey.src.model import APIModel
from autosurvey.src.database import database
from autosurvey.src.constants import DEFAULT_SAVEDIR
from autosurvey.src.utils import tokenCounter, load_txt, save_to_txt
from autosurvey.src.prompt import (
    ROUGH_OUTLINE_PROMPT,
    MERGING_OUTLINE_PROMPT,
    SUBSECTION_OUTLINE_PROMPT,
    EDIT_FINAL_OUTLINE_PROMPT,
)


class outlineWriter:

    def __init__(self, model: str, api_key: str, api_url: str, database: database):
        self.model, self.api_key, self.api_url = model, api_key, api_url
        self.api_model = APIModel(self.model, self.api_key, self.api_url)

        self.db = database
        self.token_counter = tokenCounter()
        self.input_token_usage, self.output_token_usage = 0, 0

    def reset_token_usage(self):
        self.input_token_usage, self.output_token_usage = 0, 0

    def draft_outline(
        self,
        topic: str,
        review_id: Union[int, str, None],
        chunk_size: int = 30000,
        section_num: int = 6,
        save_dir: str = DEFAULT_SAVEDIR,
        end_extension: str = "",
    ) -> str:
        """
        Creates the survey outline and saves it.

        Parameters:
        topic - the topic of the literature survey to write.
        review_id - the ID of the original review paper used as the basis (for filtering references).
        chunk_size - the size of the chunks.
        section_num - the number of sections to include.
        save_dir - the location where to save the resulting outlines.
        end_extension - the extension to use for saving the file if needed for theSciReviewGen subsetting).

        Returns:
        the refined outline as a string.
        """

        # First check if the outline file already exists
        out_name = os.path.join(save_dir, f"{review_id}{end_extension}.txt")
        if os.path.isfile(out_name):
            print(
                f"Outline for survey `{review_id}` already written. Skipping outline creation..."
            )
            final_outline = load_txt(out_name, use_truncation=False)

            # Add length to token counter (note: this would not include how many tokens were used for the writing process)
            self.output_token_usage += self.token_counter.num_tokens_from_string(
                final_outline
            )
            return final_outline

        # Get the references (passing the paper ID ensures only the paper's references get used)
        references_idx = self.db.get_ids_from_query(topic, review_id)
        references_infos = self.db.get_paper_info_from_ids(references_idx)
        references_titles = [r["title"] for r in references_infos]
        references_abs = [r["abs"] for r in references_infos]

        # Chunk the data
        abs_chunks, titles_chunks = self.chunking(
            references_abs, references_titles, chunk_size=chunk_size
        )

        # Memory Leak Prevention
        del references_abs, references_infos, references_titles
        gc.collect()

        # Generate rough section-level outlines
        outlines = self.generate_rough_outlines(
            topic=topic,
            papers_chunks=abs_chunks,
            titles_chunks=titles_chunks,
            section_num=section_num,
        )

        # Merge the outlines
        section_outline = self.merge_outlines(topic=topic, outlines=outlines)

        # Generate subsection-level outlines
        subsection_outlines = self.generate_subsection_outlines(
            topic=topic,
            section_outline=section_outline,
            review_id=review_id,
            rag_num=50,
        )

        # Memory Leak Prevention
        del outlines, titles_chunks, abs_chunks
        gc.collect()

        merged_outline = self.process_outlines(section_outline, subsection_outlines)

        # Edit final outline
        final_outline = self.edit_final_outline(merged_outline)

        # Memory Leak Prevention
        del section_outline, subsection_outlines, merged_outline
        gc.collect()

        # Save the final outline for evaluation later
        os.makedirs(save_dir, exist_ok=True)
        save_to_txt(final_outline, out_name)
        return final_outline

    def compute_price(self):
        return self.token_counter.compute_price(
            input_tokens=self.input_token_usage,
            output_tokens=self.output_token_usage,
            model=self.model,
        )

    def generate_rough_outlines(
        self,
        topic: str,
        papers_chunks: List[List],
        titles_chunks: List[List],
        section_num: int = 8,
    ) -> List[str]:
        """
        Creates a rough outline based on the references provided.

        Parameters:
        topic - the topic of the literature survey to write.
        papers_chunks - the paper abstract chunks created from the `chunk_list` process.
        papers_chunks - the title chunks created from the `chunk_list` process.
        section_num - the number of sections to include.

        Returns:
        the list of rough outlines generated by the model.
        """

        prompts = []
        for i in trange(len(papers_chunks)):

            titles, papers = titles_chunks[i], papers_chunks[i]
            paper_texts = ""
            for i, t, p in zip(range(len(papers)), titles, papers):

                paper_texts += f"---\npaper_title: {t}\n\npaper_content:\n\n{p}\n"

            paper_texts += "---\n"

            prompt = self.__generate_prompt(
                ROUGH_OUTLINE_PROMPT,
                paras={
                    "PAPER LIST": paper_texts,
                    "TOPIC": topic,
                    "SECTION NUM": str(section_num),
                },
            )
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(
            prompts
        )
        outlines = self.api_model.batch_chat(text_batch=prompts, temperature=0.0)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(
            outlines
        )

        # Memory Leak Prevention
        del prompt, prompts, paper_texts
        gc.collect()
        return outlines

    def merge_outlines(self, topic: str, outlines: List[str]) -> str:
        """
        Merges the outlines generated.

        Parameters:
        topic - the topic of the literature survey to write.
        outlines - the list of outlines returned by the `generate_rough_outlines` function.

        Returns:
        the merged outline.
        """

        outline_texts = ""
        for i, o in zip(range(len(outlines)), outlines):

            outline_texts += f"---\noutline_id: {i}\n\noutline_content:\n\n{o}\n"

        outline_texts += "---\n"
        prompt = self.__generate_prompt(
            MERGING_OUTLINE_PROMPT,
            paras={"OUTLINE LIST": outline_texts, "TOPIC": topic},
        )
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        outline = self.api_model.chat(prompt, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_string(outline)

        # Memory Leak Prevention
        del prompt, outline_texts

        return outline

    def generate_subsection_outlines(
        self,
        topic: str,
        section_outline: str,
        review_id: Union[int, str, None],
        rag_num: int,
    ) -> List[str]:
        """
        Creates the outlines for each subsection.

        Parameters:
        topic - the topic of the literature survey to write.
        section_outline - the overall outline generated by the `merge_outlines` function.
        review_id - the ID of the original review paper used as the basis (for filtering references).
        rag_num - the number of articles to retrieve from the database for each subsection.

        Returns:
        the list of subsection outlines generated by the model.
        """

        _, survey_sections, survey_section_descriptions = (
            self.extract_title_sections_descriptions(section_outline)
        )

        prompts = []
        for section_name, section_description in zip(
            survey_sections, survey_section_descriptions
        ):

            references_idx = self.db.get_ids_from_query(
                section_description, review_id=review_id, num=rag_num
            )
            references_infos = self.db.get_paper_info_from_ids(references_idx)
            references_titles = [r["title"] for r in references_infos]
            references_papers = [r["abs"] for r in references_infos]
            paper_texts = ""
            for _, t, p in zip(
                range(len(references_papers)), references_titles, references_papers
            ):

                paper_texts += f"---\npaper_title: {t}\n\npaper_content:\n\n{p}\n"

            paper_texts += "---\n"
            prompt = self.__generate_prompt(
                SUBSECTION_OUTLINE_PROMPT,
                paras={
                    "OVERALL OUTLINE": section_outline,
                    "SECTION NAME": section_name,
                    "SECTION DESCRIPTION": section_description,
                    "TOPIC": topic,
                    "PAPER LIST": paper_texts,
                },
            )
            prompts.append(prompt)

        self.input_token_usage += self.token_counter.num_tokens_from_list_string(
            prompts
        )
        sub_outlines = self.api_model.batch_chat(prompts, temperature=0.0)
        self.output_token_usage += self.token_counter.num_tokens_from_list_string(
            sub_outlines
        )

        # Memory Leak Prevention
        del prompt, prompts, survey_sections, survey_section_descriptions
        gc.collect()
        return sub_outlines

    def process_outlines(self, section_outline: str, sub_outlines: List[str]) -> str:
        """
        Combines the outlines and the sub-outlines generated.

        Parameters:
        section_outline - the main outline to be parsed.
        sub_outlines - the subsection outlines corresponding to the main outline.

        Returns:
        the full outline.
        """

        res = ""
        survey_title, survey_sections, survey_section_descriptions = (
            self.extract_title_sections_descriptions(outline=section_outline)
        )
        res += f"# {survey_title}\n\n"
        for i in range(len(survey_sections)):

            section = survey_sections[i]
            res += (
                f"## {i+1} {section}\nDescription: {survey_section_descriptions[i]}\n\n"
            )
            subsections, subsection_descriptions = (
                self.extract_subsections_subdescriptions(sub_outlines[i])
            )
            for j in range(len(subsections)):

                subsection = subsections[j]
                res += f"### {i+1}.{j+1} {subsection}\nDescription: {subsection_descriptions[j]}\n\n"

        # Memory Leak Prevention
        del survey_title, survey_sections, survey_section_descriptions
        gc.collect()
        return res

    def edit_final_outline(self, outline: str) -> str:
        """
        Creates the final outline by refining/editing the merged outline.

        Parameters:
        outline - the drafted outline thus far.

        Returns:
        the final outline, fully edited by the model.
        """

        prompt = self.__generate_prompt(
            EDIT_FINAL_OUTLINE_PROMPT, paras={"OVERALL OUTLINE": outline}
        )
        self.input_token_usage += self.token_counter.num_tokens_from_string(prompt)
        outline = self.api_model.chat(prompt, temperature=1)
        self.output_token_usage += self.token_counter.num_tokens_from_string(outline)

        # Memory Leak Prevention
        del prompt
        gc.collect()
        return outline.replace("<format>\n", "").replace("</format>", "")

    def __generate_prompt(self, template: str, paras: Dict) -> str:
        """
        Formats parameters into a prompt template.

        Parameters:
        template - the string template to use.
        paras - the parameters to input into the template.

        Returns:
        the formatted prompt.
        """
        prompt = template
        for k in paras.keys():

            prompt = prompt.replace(f"[{k}]", paras[k])

        return prompt

    def extract_title_sections_descriptions(
        self, outline: str
    ) -> Tuple[str, List[str], List[str]]:
        """
        Parses an outline into its parts.

        Parameters:
        outline - the outline to parse.

        Returns:
        the title, sections, and descriptions of the outline.
        """

        title = outline.split("Title: ")[1].split("\n")[0]
        sections, descriptions = [], []
        for i in range(100):

            if f"Section {i+1}" in outline:
                sections.append(outline.split(f"Section {i+1}: ")[1].split("\n")[0])
                descriptions.append(
                    outline.split(f"Description {i+1}: ")[1].split("\n")[0]
                )

        return title, sections, descriptions

    def extract_subsections_subdescriptions(
        self, outline: str
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts the subsection descriptions from an outline.

        Parameters:
        outline - the outline to parse.

        Returns:
        the subsections and the descriptions.
        """

        subsections, subdescriptions = [], []
        for i in range(100):

            if f"Subsection {i+1}" in outline:
                subsections.append(
                    outline.split(f"Subsection {i+1}: ")[1].split("\n")[0]
                )
                subdescriptions.append(
                    outline.split(f"Description {i+1}: ")[1].split("\n")[0]
                )

        return subsections, subdescriptions

    def chunking(
        self, papers: List[str], titles: List[str], chunk_size: int = 14000
    ) -> Tuple[List[List], List[List]]:
        """
        Chunks the papers based on the token size window.

        Parameters:
        papers - the paper abstracts.
        titles - the paper titles.
        chunk_size - the size of the resulting chunk (in tokens).

        Returns:
        the paper and title chunks.
        """

        paper_chunks, title_chunks = [], []
        total_length = self.token_counter.num_tokens_from_list_string(papers)
        num_of_chunks = int(total_length / chunk_size) + 1
        avg_len = int(total_length / num_of_chunks) + 1
        split_points = []
        l = 0
        for j in range(len(papers)):

            l += self.token_counter.num_tokens_from_string(papers[j])
            if l > avg_len:
                l = 0
                split_points.append(j)
                continue

        start = 0
        for point in split_points:

            paper_chunks.append(papers[start:point])
            title_chunks.append(titles[start:point])
            start = point

        paper_chunks.append(papers[start:])
        title_chunks.append(titles[start:])

        return paper_chunks, title_chunks
