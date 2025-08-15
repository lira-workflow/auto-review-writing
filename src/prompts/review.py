# The prompt to use for response improvement
# Some prompts are adapted from https://langchain-ai.github.io/langgraph/tutorials/reflexion/reflexion/#initial-responder.

REVIEW_PROMPT = """
You are an expert literature review paper reviewer tasked with evaluating the components of the final literature review.
Be fair and strict to maximize improvement.
""".strip()

# This is to tell the system that only abstracts are provided
REVIEW_ABSTRACT = """
Keep in mind that ONLY the reference titles and abstracts are available, meaning it is more difficult to go into depth in 
certain aspects and provide specific examples.
""".strip()

# =============================================================================================================================

REVIEW_OUTLINE = """
You have been tasked with evaluating an outline with the topic `{REVIEW_TOPIC}` as shown below:

---
{CONTENT}
---

Review this outline and decide if it is sufficiently good for a literature review. 
Use the following points to guide your review:

1. The organization of the outline in relation to the topic.
2. Some indication of a clear and valuable contribution to the topic.
3. The inclusion of a section for questions or directions for further research.

Answer if these are fulfilled using the `per criteria` section below. Use this to determine the final sufficiency.
Focus mainly on the headers and check if the descriptions are sufficient.
Note that the methodology cannot be reported as it is outside the scope of the review. Also, do NOT expect examples from the outline.
Be objective, thorough, and as concise as possible in your analysis, ONLY including points that you are absolutely certain of.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for evaluation based on the above criteria. Think step-by-step>

## PER CRITERIA
1. <yes/no>
2. <yes/no>
3. <yes/no>

## SUFFICIENT
<yes/no>

## FEEDBACK (INCLUDE THIS ONLY IF YOUR ANSWER IS NOT YES)
<a numbered list of review points. Provide a very brief helpful example to help the writer if possible.>
</FORMAT>

Do NOT include anything else in your response.
""".strip()

# =============================================================================================================================

REVIEW_DRAFT = """
You have been tasked with evaluating a literature review paper draft with the topic `{REVIEW_TOPIC}` as shown below:

---
{CONTENT}
---

Use the following questions to guide your review:

1. The organization of the content in relation to the topic.
2. If the article synthesize the findings into a clear and valuable contribution to the topic.
3. Elaboration on questions or directions for further research.
4. Proper use of citations (in the format of "<sentence> [citation_title]").

Note that the methodology cannot be reported as it is outside the scope of the review.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for evaluation based on the above criteria. Think step-by-step>

## PER CRITERIA
1. <yes/no>
2. <yes/no>
3. <yes/no>
4. <yes/no>

## SUFFICIENT
<yes/no>

## FEEDBACK (include this ONLY IF YOUR ANSWER IS NO)
<a numbered list of review points PER SECTION, so ensure ALL major sections receive feedback if this part is included>
</FORMAT>

Do NOT include anything else in your response.
""".strip()

# =============================================================================================================================

REVIEW_EDIT = """
You have been tasked with evaluating an edited literature review with the topic `{REVIEW_TOPIC}`.
Review ONLY the most recent edited paper and decide if it is sufficiently good for a literature review. 
Use the following questions to guide your review:

1. Do the sections coherently connect with each other?
2. Is the result of the review reported in an appropriate and clear way?
3. Does the article synthesize the findings of the literature review into a clear and valuable contribution to the topic?
4. Does the edited article keep the original content of the paper intact, ONLY adjusting the writing style?
5. Is the style of writing scientific?
6. Does the article vary it's content and vocabulary enough?
 
As reference for checking, here is the original paper:

---
{CONTENT}
---

Structure your response in the following way (exclude the <FORMAT> parts and try to use less sub-subsections if possible):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for evaluation based on the above criteria. Think step-by-step>


## PER CRITERIA
1. <yes/no>
2. <yes/no>
3. <yes/no>
4. <yes/no>
5. <yes/no>
6. <yes/no>

## SUFFICIENT
<yes/no>

## FEEDBACK (INCLUDE THIS ONLY IF YOUR ANSWER IS NOT YES)
<a numbered list of review points. Be specific about where improvements can be made>
</FORMAT>

Do NOT include anything else in your response.
""".strip()
