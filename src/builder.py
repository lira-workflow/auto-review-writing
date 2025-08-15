import gc, os, re, time
from tqdm import tqdm
from typing import List, Dict, Union
from joblib import Parallel, delayed
from langgraph.types import Command
from langchain.schema import AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.editor import EditorAgent
from src.reviewer import ReviewerAgent
from src.researchers import ResearcherAgent
from src.writers import OutlineDrafterAgent, ContentWriterAgent
from src.utils.pbar_utils import tqdm_joblib
from src.utils.communication import LiRAState
from src.utils.constants import (
    BAR_FORMAT,
    LLM_SEED,
    N_JOBS,
    SETTING_NAME,
    TEMP_DIR,
    MODEL_VER,
    GROUP_SIZE,
    NUM_SECTIONS,
    NUM_SUBSECTIONS,
    SECTION_LEN,
    SETUP_STATE,
    CONTEXT_SIZE,
    ANALYZE_ALL_PAPERS,
    PREPARE_GROUPS,
    RESEARCHER_GROUP,
    DRAFT_OUTLINE,
    REVIEW_CONTENT,
    WRITE_CONTENT,
    EDIT_CONTENT,
)

# Disable parallelism for HuggingFace
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Research team node builder
def build_research_team(
    papers: List[Dict[str, str]],
    topic: str,
    review_id: Union[int, str],
    memory: MemorySaver,
    seed: int = LLM_SEED,
    researcher_model: str = MODEL_VER,
    dataset: str = "srg",
    temp_dir: str = TEMP_DIR,
    group_size: int = GROUP_SIZE,
    n_jobs: int = N_JOBS,
    context_sizes: Dict = {"gpt": CONTEXT_SIZE, "other": CONTEXT_SIZE},
    setting_name: str = SETTING_NAME,
) -> CompiledStateGraph:
    """
    Builds the research group node for one review paper.

    Parameters:
    papers - the list of references (see `src.utils.preprocessing.get_papers` for more info).
    topic - the topic (title) of the review.
    review_id - the ID of the original review paper used as the basis (for filtering references).
    memory - a memory object for LangGraph checkpointing.
    researcher_model - the model type for the researchers.
    dataset - the dataset to use.
    temp_dir - the location to save the intermediate responses to (to prevent repeat API calls).
    group_size - the number of papers allowed in one paper group.
    n_jobs - the number of workers to use.
    context_sizes - the dictionary containing the context window size for each model type.
    setting_name - the experimental setting being used.

    Returns:
    a researcher team as a `CompiledStateGraph` object.
    """

    # First define the LLM agents
    # The researcher agents ==================================================================================
    researcher_model_code = "gpt" if "gpt" in researcher_model.lower() else "other"
    researcher_agent = ResearcherAgent(
        "researcher",
        review_id,
        seed=seed,
        model_name=researcher_model,
        context_size=context_sizes[researcher_model_code],
        dataset=dataset,
        temp_dir=temp_dir,
        review_topic=topic,
        setting_name=setting_name,
    )

    # Then define the calls to make them compatible with the LangGraph API
    # Loading the papers =====================================================================================
    def setup_state(state: LiRAState) -> Command:

        # Setup the workflow
        new_messages = state["messages"] + [
            AIMessage(
                content=f"UPDATE: Workflow set-up!",
            )
        ]

        return Command(
            goto=ANALYZE_ALL_PAPERS,
            update={
                "messages": new_messages,
                "topic": topic,
                "papers": papers,
                "draft_outlines": [],
                "revision_count_outline": 0,
                "revision_count_writing": 0,
                "revision_count_editing": 0,
                "outline_discussion": [],
                "content_discussion": {},
                "editing_discussion": [],
            },
        )

    # Analyzing each paper ===================================================================================
    def analyze_all_papers(
        state: LiRAState,
    ) -> Command:

        # Perform the analysis on all papers (if the researcher model is defined)
        papers = state["papers"]
        if researcher_model.lower() != "none":
            with tqdm_joblib(
                tqdm(
                    desc="Analyzing all references",
                    unit="file(s)",
                    total=len(papers),
                    bar_format=BAR_FORMAT,
                )
            ):
                results = Parallel(n_jobs=n_jobs, verbose=0, backend="threading")(
                    delayed(researcher_agent.analyze_paper)(paper) for paper in papers
                )

        # Otherwise, just use the paper contents
        else:

            # Apparently, the content is sometimes a float
            results = [[AIMessage(content=str(paper["content"]))] for paper in papers]

        # Formatting print
        print()

        # Format the list
        analysis_list = []
        for paper, result in zip(papers, results):

            to_append = {
                "id": paper["id"],
                "title": paper["title"],
                "content": result,
            }
            analysis_list.append(to_append)

        # Memory Leak Prevention
        del results
        gc.collect()

        # Update the state
        if researcher_model.lower() == "none":
            new_content = AIMessage(
                content="INFO: Research phase will be skipped.",
                additional_kwargs={"agent": "none"},
            )

        else:
            new_content = AIMessage(
                content="UPDATE: All articles have been analyzed!",
                additional_kwargs={"agent": researcher_agent.name},
            )

        new_messages = state["messages"] + [new_content]
        return Command(
            update={
                "messages": new_messages,
                "researcher_analyses": analysis_list,
            },
            goto=PREPARE_GROUPS,
        )

    # Analyzing the type =====================================================================================
    def prepare_groups(
        state: LiRAState,
    ) -> Command:

        # This is the only part which involves the agent if the model name is set to "none"
        paper_groups = researcher_agent.format_analyses(
            state["researcher_analyses"], group_size
        )

        # Update the state
        if researcher_model.lower() == "none":
            new_content = AIMessage(
                content="INFO: Research phase will be skipped.",
                additional_kwargs={"agent": "none"},
            )

        else:
            new_content = AIMessage(
                content="UPDATE: All articles have been analyzed!",
                additional_kwargs={"agent": researcher_agent.name},
            )

        new_messages = state["messages"] + [new_content]

        return Command(
            update={
                "messages": new_messages,
                "paper_groups": paper_groups,
            },
        )

    # Build the graph ========================================================================================
    workflow = StateGraph(LiRAState)
    workflow.add_node(SETUP_STATE, setup_state)
    workflow.add_node(ANALYZE_ALL_PAPERS, analyze_all_papers)
    workflow.add_node(PREPARE_GROUPS, prepare_groups)
    workflow.set_entry_point(SETUP_STATE)

    # Compile the workflow
    return workflow.compile(checkpointer=memory)


# Main workflow builder
def build_research_workflow(
    papers: List[Dict[str, str]],
    topic: str,
    review_id: Union[int, str],
    seed: int = LLM_SEED,
    researcher_model: str = MODEL_VER,
    writer_model: str = MODEL_VER,
    editor_model: str = MODEL_VER,
    reviewer_model: str = MODEL_VER,
    max_revisions: int = 3,
    dataset: str = "srg",
    temp_dir: str = TEMP_DIR,
    group_size: int = GROUP_SIZE,
    context_sizes: Dict = {"gpt": CONTEXT_SIZE, "other": CONTEXT_SIZE},
    n_jobs: int = N_JOBS,
    n_sections: int = NUM_SECTIONS,
    n_subsections: int = NUM_SUBSECTIONS,
    fulltext: bool = False,
    section_len: int = SECTION_LEN,
    setting_name: str = SETTING_NAME,
    overwrite_responses: bool = False,
    use_retriever: bool = False,
) -> CompiledStateGraph:
    """
    Builds the research workflow for one review paper.

    Parameters:
    papers - the list of references (see `src.utils.preprocessing.get_papers` for more info).
    topic - the topic (title) of the review.
    review_id - the ID of the original review paper used as the basis (for filtering references).
    seed - the seed to use for the LLM (if supported).
    researcher_model - the model type for the researchers.
    writer_model - the model type for the writers.
    editor_model - the model type for the editor.
    reviewer_model - the model type for the reviewer.
    max_revisions - the maximum number of revisions allowed for any given component.
    dataset - the dataset to use.
    temp_dir - the directory to save the response caches (to prevsent redoing API calls).
    group_size - the number of papers allowed in one paper group.
    context_sizes - the dictionary containing the context window size for each model type.
    n_jobs - the number of workers to use.
    n_sections - the number of sections to include in the review.
    n_subsections - the number of subsections to include within a section header.
    fulltext - whether the fulltexts are being used.
    section_len - the (rough) length of each section in words.
    setting_name - the experimental setting being used.
    overwrite_responses - whether to overwrite the LLM cache responses. This is for the components other than the
    researchers.
    use_retriever - whether the retriever is being used.

    Returns:
    a compiled research workflow for one review paper.
    """

    # Review ID conversion in case it's a number
    review_id = str(review_id)

    # First define the LLM agents and memory
    # The researcher agents ==================================================================================
    memory = MemorySaver()
    researcher_group = build_research_team(
        papers,
        topic,
        review_id,
        memory=memory,
        seed=seed,
        researcher_model=researcher_model,
        dataset=dataset,
        temp_dir=temp_dir,
        group_size=group_size,
        n_jobs=n_jobs,
        context_sizes=context_sizes,
        setting_name=setting_name,  # As the analyses would be the same across settings, this would increase result validity and reduce costs.
    )

    # The outline drafter agent ==============================================================================
    writer_model_code = "gpt" if "gpt" in writer_model else "other"
    outline_drafter_agent = OutlineDrafterAgent(
        "outline_drafter",
        review_id,
        seed=seed,
        model_name=writer_model,
        context_size=context_sizes[writer_model_code],
        dataset=dataset,
        temp_dir=temp_dir,
        review_topic=topic,
        setting_name=setting_name,
        overwrite_responses=overwrite_responses,
    )

    # The content writer agent ===============================================================================
    content_writer_agent = ContentWriterAgent(
        "content_writer",
        review_id,
        writer_model,
        context_size=context_sizes[writer_model_code],
        dataset=dataset,
        temp_dir=temp_dir,
        review_topic=topic,
        setting_name=setting_name,
        overwrite_responses=overwrite_responses,
    )

    # The editor agent =======================================================================================
    editor_model_code = "gpt" if "gpt" in writer_model else "other"
    if editor_model.lower() != "none":
        editor_agent = EditorAgent(
            "editor",
            review_id,
            editor_model,
            context_size=context_sizes[editor_model_code],
            dataset=dataset,
            temp_dir=temp_dir,
            setting_name=setting_name,
            overwrite_responses=overwrite_responses,
        )

    # The reviewer agent =====================================================================================
    reviewer_model_code = "gpt" if "gpt" in reviewer_model else "other"
    reviewer_agent = ReviewerAgent(
        "reviewer",
        review_id,
        seed=seed,
        model_name=reviewer_model,
        context_size=context_sizes[reviewer_model_code],
        dataset=dataset,
        temp_dir=temp_dir,
        review_topic=topic,
        fulltext=fulltext,
        setting_name=setting_name,
        overwrite_responses=overwrite_responses,
    )

    # Creating the outline ===================================================================================
    def draft_outline(
        state: LiRAState,
    ) -> Command:

        # Get the revision count
        revision_count = state["revision_count_outline"] + 1

        # If this is the first iteration, create the initial outline
        if revision_count <= 1:
            # Create an outline for each group of papers
            draft_outlines = []
            for paper_group in state["paper_groups"]:

                draft_outline = outline_drafter_agent.create_outline_single(
                    paper_group, n_sections, n_subsections
                )
                draft_outlines.append(draft_outline)

            # Then combine them all
            final_draft_outline = outline_drafter_agent.merge_outlines(
                draft_outlines,
                n_sections,
                n_subsections,
                revision_count=revision_count,
            )

            state["draft_outlines"] = draft_outlines
            new_outline_discuss = state["outline_discussion"] + final_draft_outline

        else:
            # Otherwise fix the outline
            final_draft_outline = outline_drafter_agent.fix_outline(
                state["outline_discussion"],
                state["draft_outlines"],
                n_sections,
                n_subsections,
                revision_count=revision_count,
            )  # This is a full discussion list. See the function for more
            new_outline_discuss = final_draft_outline

        # Update the state
        new_messages = state["messages"] + [
            AIMessage(
                content="UPDATE: An outline draft has been created!",
                additional_kwargs={"agent": outline_drafter_agent.name},
            )
        ]

        return Command(
            update={
                "messages": new_messages,
                "revision_count_outline": state["revision_count_outline"] + 1,
                "outline_discussion": new_outline_discuss,
                "to_review_now": "outline",
            },
            goto=REVIEW_CONTENT,
        )

    # Writing the content ====================================================================================
    def write_content(
        state: LiRAState,
    ) -> Command:

        # Get the revision count
        revision_count = state["revision_count_writing"] + 1

        # Create the writer database
        if content_writer_agent.vectorstore is None:
            content_writer_agent.create_db(state)

        # Get the current outline
        parsed = content_writer_agent.parse_outline(
            state["outline_discussion"][-2].content
        )

        # Determine if the current step is initial writing or revision
        if revision_count <= 1:
            section_discussions = {}
            desc = "Writing Sections"
            function = content_writer_agent.write_content_section
            feedback = False

        else:
            section_discussions = state["content_discussion"]
            desc = "Revising Sections"
            function = content_writer_agent.revise_content_section
            feedback = True

        # Then write each (sub)section
        with tqdm_joblib(
            tqdm(
                desc=desc,
                unit="part(s)",
                total=len(parsed["headers"]),
                bar_format=BAR_FORMAT,
            )
        ):
            sections = Parallel(n_jobs, verbose=0, backend="threading")(
                delayed(function)(
                    state,
                    header,
                    description,
                    reference,
                    higher_header,
                    revision_count,
                    section_len,
                )
                for header, description, reference, higher_header in zip(
                    parsed["headers"],
                    parsed["descriptions"],
                    parsed["references"],
                    parsed["header_higher_level"],
                )
            )

        # Create an entry for each component for later discussion
        for h, s in zip(parsed["headers"], sections):

            if revision_count <= 1:
                section_discussions[h] = s

            else:
                if s is not None:
                    section_discussions[h].extend(s)

        # Get the abstract
        review_joined = "\n\n===\n\n".join(
            [s[-1].content for s in sections if s is not None]
        )

        title_abstract = content_writer_agent.write_title_abstract(
            review_joined, revision_count, feedback
        )

        # Get the conclusion (and the title defined by the agent)
        conc_title = (
            list(
                filter(
                    lambda t: "conclusion" in t.lower() and "in" not in t.lower(),
                    parsed["headers"],
                )
            )[0]
            if any(
                [
                    "conclusion" in t.lower() and "in" not in t.lower()
                    for t in parsed["headers"]
                ]
            )
            else "Conclusion"
        )

        conclusion = content_writer_agent.write_conclusion(
            state, review_joined, conc_title, revision_count, feedback
        )

        if revision_count <= 1:
            section_discussions["Title and Abstract"] = title_abstract
            section_discussions[conc_title] = conclusion

        else:
            section_discussions["Title and Abstract"].extend(title_abstract)
            section_discussions[conc_title].extend(conclusion)

        # Formatting print
        print()

        # Get the title, abstract, and conclusion text
        title_abstract_f = re.sub("# Abstract\n", "", title_abstract[-1].content)
        conclusion_f = conclusion[-1].content

        # Combining the text
        review_draft_full = (
            title_abstract_f
            + "\n\n===\n\n"
            + review_joined
            + "\n\n===\n\n"
            + conclusion_f
        )

        # Update the state
        new_messages = state["messages"] + [
            AIMessage(
                content=f"UPDATE: Draft review pass {revision_count} have been made!",
            )
        ]

        return Command(
            update={
                "messages": new_messages,
                "revision_count_writing": state["revision_count_writing"] + 1,
                "draft_review": review_draft_full,
                "to_review_now": "draft review",
                "content_discussion": section_discussions,
            },
            goto=REVIEW_CONTENT,
        )

    # Editing the paper ======================================================================================
    def edit_content(
        state: LiRAState,
    ) -> Command:

        # Get the revision count
        revision_count = state["revision_count_editing"] + 1

        # If this is the first iteration, create the initial outline
        print("Currently editing. This may take a while...\n")
        if revision_count <= 1:
            # Edit the current draft
            edited_review = editor_agent.edit_paper(
                state["draft_review"], revision_count
            )

        else:
            time.sleep(2)  # To slow down requests
            edited_review = editor_agent.revise_edit(
                state["draft_review"], state["editing_discussion"], revision_count
            )

        # Extract the content from the response
        final_paper = edited_review[-1].content

        # Update the state
        new_editing_discuss = state["editing_discussion"] + edited_review
        new_messages = state["messages"] + [
            AIMessage(
                content=f"UPDATE: Edit pass {revision_count} done!",
                additional_kwargs={"agent": editor_agent.name},
            )
        ]

        return Command(
            update={
                "messages": new_messages,
                "final_review": final_paper,
                "revision_count_editing": state["revision_count_editing"] + 1,
                "editing_discussion": new_editing_discuss,
                "to_review_now": "edited review",
            },
            # goto=REVIEW_CONTENT,
            goto=END,  # Temporary
        )

    # Review the parts =======================================================================================
    def review_content(
        state: LiRAState,
    ) -> Command:

        # Fetch the correct revision counter
        if state["to_review_now"] == "outline":
            counter_key = "revision_count_outline"
            discussion = "outline_discussion"
            mode = 0

        elif state["to_review_now"] == "draft review":
            counter_key = "revision_count_writing"
            discussion = "content_discussion"
            mode = 1

        elif state["to_review_now"] == "edited review":
            counter_key = "revision_count_editing"
            discussion = "editing_discussion"
            mode = 2

        # Conduct the review if revision is allowed
        messages, to_review_now, revision_count = (
            state[discussion],
            state["to_review_now"],
            state[counter_key],
        )

        # Only used for the content evaluation
        discussion_dict = messages

        # Check if revision is still allowed
        if revision_count <= max_revisions:
            # For the Outline (only one review total)
            if mode != 1:
                review_message, verdict = reviewer_agent.review_component(
                    state,
                    messages[-1].content,
                    review_number=revision_count,
                    component=state["to_review_now"],
                )
                review_messages = messages + review_message

            else:
                review_message, verdict = reviewer_agent.review_component(
                    state,
                    state["draft_review"],
                    review_number=revision_count,
                    component=state["to_review_now"],
                )

                # Here `messages` is a Dict
                for part in messages.keys():

                    discussion_dict[part] = messages[part]
                    discussion_dict[part].extend(review_message)

        else:
            # Assume it's done
            # Update the messages for later processing
            stop_message = [AIMessage(content="REVIEWING STOPPED!")]
            if mode != 1:
                messages += stop_message
                review_messages = messages

            else:
                for part in messages.keys():

                    discussion_dict[part].extend(stop_message)

            verdict = True

        # Determine which node to go to next
        destinations = {
            "outline": [DRAFT_OUTLINE, WRITE_CONTENT],
            "draft review": (
                [WRITE_CONTENT, EDIT_CONTENT]
                if editor_model.lower() != "none"
                else [WRITE_CONTENT, END]
            ),
            "edited review": [EDIT_CONTENT, END],
        }

        target_index = 1 if verdict else 0
        goto = destinations[state["to_review_now"]][target_index]
        new_revision_count = state[counter_key]

        # Update the state
        new_messages = state["messages"] + [
            AIMessage(
                content=f"UPDATE: Evaluation of {to_review_now} pass {revision_count} done!",
            )
        ]

        update = {
            "messages": new_messages,
            discussion: review_messages if mode != 1 else discussion_dict,
            "to_review_now": to_review_now,
            counter_key: new_revision_count,
        }

        return Command(
            update=update,
            goto=goto,
        )

    # Build the graph ========================================================================================
    workflow = StateGraph(LiRAState)
    workflow.add_node(RESEARCHER_GROUP, researcher_group)
    workflow.add_node(DRAFT_OUTLINE, draft_outline)
    workflow.add_node(REVIEW_CONTENT, review_content)
    workflow.add_node(WRITE_CONTENT, write_content)

    if editor_model.lower() != "none":
        workflow.add_node(EDIT_CONTENT, edit_content)

    workflow.add_edge(RESEARCHER_GROUP, DRAFT_OUTLINE)
    workflow.set_entry_point(RESEARCHER_GROUP)

    # Compile the workflow
    return workflow.compile(checkpointer=memory)
