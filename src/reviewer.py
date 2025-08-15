import re
from typing import Dict, List, Tuple
from src.base_agent import BaseAgent
from src.utils.communication import LiRAState
from src.utils.constants import (
    LLM_SEED,
    TEMP_DIR,
    MODEL_VER,
    MAX_ITEMS,
    CONTEXT_SIZE,
    TEMPERATURE,
    SETTING_NAME,
    ENCODING_NAME,
)
from src.prompts.review import (
    REVIEW_PROMPT,
    REVIEW_EDIT,
    REVIEW_DRAFT,
    REVIEW_ABSTRACT,
    REVIEW_OUTLINE,
)


# RegEx patterns
stop_pattern = r"#{1,3} SUFFICIENT\s+(.{2,3})"


class ReviewerAgent(BaseAgent):
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
        fulltext: bool = False,
        setting_name: str = SETTING_NAME,
        overwrite_responses: bool = False,
    ):

        prompt = (
            REVIEW_PROMPT + "\n" + REVIEW_ABSTRACT if not fulltext else REVIEW_PROMPT
        )
        super().__init__(
            name=name,
            model_name=model_name,
            context_size=context_size,
            seed=seed,
            temperature=temperature,
            max_memory_items=max_memory_items,
            system_prompt=prompt,
            dataset=dataset,
            temp_dir=temp_dir,
            review_id=review_id,
            encoding_name=encoding_name,
            setting_name=setting_name,
            overwrite_response=overwrite_responses,
        )
        self.review_topic = review_topic
        self.review_templates = {
            "outline": REVIEW_OUTLINE,
            "draft review": REVIEW_DRAFT,
            "edited review": REVIEW_EDIT,
        }

    def parse_review(self, review: List) -> bool:
        """
        Creates a paper outline based on the analyses and type provided.
        As multiple paper analyses groups are possible, this is performed on each group using
        separate calls.

        Parameters:
        review - the review response to parse.

        Returns:
        `True` if the review indicates that no further refinement is necessary, else `False`.
        """

        content = review[-1].content
        match = re.search(stop_pattern, content)
        if match is None:
            return False

        output = match.group(1).lower() == "yes"
        return output

    def review_component(
        self,
        state: LiRAState,
        content: str,
        review_number: int,
        component: str = "outline",
    ) -> Tuple[Dict, bool]:
        """
        Creates a paper outline based on the analyses and type provided.
        As multiple paper analyses groups are possible, this is performed on each group using
        separate calls.

        Parameters:
        state - the worfklow's current state.
        content - the content to review.
        review_number - the current review iteration number.
        component - the component type to review.

        Returns:
        a dictionary containing the outline, alongside a boolean on if the review is positive or not.
        """

        # Get the components
        component_name = state["to_review_now"]
        component_name = re.sub(" ", "_", component_name)

        # Draft the outline
        prompt = self.review_templates[component].format(
            REVIEW_TOPIC=self.review_topic, CONTENT=content
        )
        response = self.call_model(
            prompt=prompt,
            response_only=True,
            file_name=f"{self.review_id}_review_{component_name}_{review_number}.json",
            folder_name="review",
        )

        # In case a prompt artifact is leftover
        response[-1].content = re.sub(
            " (INCLUDE THIS ONLY IF YOUR ANSWER IS NOT YES)", "", response[-1].content
        )

        # Parse the review
        verdict = self.parse_review(response)
        return response, verdict
