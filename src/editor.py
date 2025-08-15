import re
from typing import List
from langchain.schema import AIMessage
from src.base_agent import BaseAgent
from src.utils.constants import (
    LLM_SEED,
    TEMP_DIR,
    MODEL_VER,
    MAX_ITEMS,
    TEMPERATURE,
    SETTING_NAME,
    CONTEXT_SIZE,
)
from src.prompts.edit import (
    EDITOR_PROMPT,
    PAPER_EDIT_PROMPT,
    PAPER_EDIT_REVISE_PROMPT,
    PAPER_COMPLETE_EDIT_PROMPT,
    PAPER_COMPLETE_REVISE_PROMPT,
)

# RegEx patterns
type_regex = r"## TYPE\n(.+\n?)"
reason_regex = r"## REASON\n(.+\n?(?:\n.+){1,7})"
chain_of_thought_pattern = r"## THOUGHTS\n(?:.+\n){1,10}\n"


class EditorAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        review_id: str,
        seed: int = LLM_SEED,
        model_name: str = MODEL_VER,
        context_size: int = CONTEXT_SIZE,
        temperature: float = TEMPERATURE,
        max_memory_items: int = MAX_ITEMS,
        dataset: str = "srg",
        temp_dir: str = TEMP_DIR,
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
            system_prompt=EDITOR_PROMPT,
            dataset=dataset,
            temp_dir=temp_dir,
            review_id=review_id,
            setting_name=setting_name,
            overwrite_response=overwrite_responses,
        )

    def edit_paper(self, paper: str, revision_count: int = 1) -> List:
        """
        Edits a draft paper into a final refined paper.

        Parameters:
        paper - the draft review paper text.
        revision_count - the paper's draft (iteration) number.

        Returns:
        the edited literature review as an `AIMessage` in a list.
        """

        prompt = PAPER_EDIT_PROMPT.format(DRAFT_PAPER=paper)
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"final_{revision_count}.json",
            folder_name="final",
        )

        # To check if the output was cutoff due to the output window limit
        # TODO: Improve based on present headers and not just period
        if (
            response[-1].response_metadata["token_usage"]["completion_tokens"] >= 16_384
            and response[-1].content[-1] != "."
        ):
            prompt_complete = PAPER_COMPLETE_EDIT_PROMPT.format(
                EDITED_PAPER=response[-1].content, DRAFT_PAPER=paper
            )
            response_complete = self.call_model(
                prompt=prompt_complete,
                response_only=True,
                file_name=f"final_{revision_count}_completion.json",
                folder_name="final",
            )

            # Remove the "MISSING SECTIONS list"
            response_complete[-1].content = response_complete[-1].content.split(
                "===\n\n", 1
            )[1]

            # Adjust the response based on the completion requirement
            content = (
                response[-1].content.rsplit("===\n\n", 1)[0]
                + "===\n\n"
                + response_complete[-1].content
            )
            response_final = [AIMessage(content=content)]

        else:
            response_final = response

        return response_final

    def revise_edit(
        self, paper: str, discussion: List, revision_count: int = 2
    ) -> List:
        """
        Edits a draft paper into a final refined paper with revision.

        Parameters:
        paper - the draft review paper text.
        discussion - the paper edit discussion.
        revision_count - the paper's draft (iteration) number.

        Returns:
        the revised edited literature review as an `AIMessage` in a list.
        """

        prompt = PAPER_EDIT_REVISE_PROMPT.format(DRAFT_PAPER=paper)
        response = self.call_model(
            prompt=prompt,
            messages=discussion,
            response_only=True,
            file_name=f"final_{revision_count}.json",
            folder_name="final",
        )
        # Remove any potential leftover formatting
        response[-1].content = re.sub(r"<\/?FORMAT>\n{0,2}", "", response[-1].content)

        # Remove the Chain-of-Thought part
        response[-1].content = re.sub(
            chain_of_thought_pattern, "", response[-1].content
        )

        # To check if the output was cutoff due to the output window limit
        if (
            response[-1].response_metadata["token_usage"]["completion_tokens"] >= 16_384
            and response[-1].content[-1] != "."
        ):
            prompt_complete = PAPER_COMPLETE_REVISE_PROMPT.format(
                EDITED_PAPER=response[-1].content, DRAFT_PAPER=paper
            )
            response_complete = self.call_model(
                prompt=prompt_complete,
                response_only=True,
                file_name=f"final_{revision_count}_completion.json",
                folder_name="final",
            )

            # Remove the "MISSING SECTIONS list"
            response_complete[-1].content = response_complete[-1].content.split(
                "===\n\n", 1
            )[1]

            # Adjust the response based on the completion requirement
            content = (
                response[-1].content.rsplit("===\n\n", 1)[0]
                + "===\n\n"
                + response_complete[-1].content
            )
            response_final = [AIMessage(content=content)]

        else:
            response_final = response

        return response_final
