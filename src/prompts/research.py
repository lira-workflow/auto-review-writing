# The researcher prompt(s)
# This prompt follows the definition provided in https://github.com/geekan/MetaGPT/blob/main/metagpt/prompts/product_manager.py.
RESEARCHER_PROMPT = """
You are a PhD-level researcher specializing in fields related to `{REVIEW_TOPIC}`.
You are currently working on creating a systematic literature review on said topic.
""".strip()

# =============================================================================================================================

ABSTRACT_ANALYSIS_PROMPT = """
Your task is to analyze the provided abstract and gather all the relevant findings as required below.
You should always output a numbered list of findings. Be objective, thorough, and as concise as possible in your analysis, 
ONLY including points that you are absolutely certain of given the limited contents of the abstract.

Use the following questions to guide your analysis:

1. What should be captured from the research? Include any formulas here if present
2. What should be captured around the study design? e.g., population (if applicable), methodology
3. What outcomes of the research are important to address the topic?
4. What is the quality of the research? e.g., limitations, potential improvements

Structure your analysis in the following way (exclude the <FORMAT> parts):

<FORMAT>

## MAIN FINDINGS
<numbered list of points about findings>

## STUDY DESIGN
<numbered list of point about study design>

## OUTCOMES
<numbered list about outcomes>

## QUALITY
<numbered list about the study quality>

</FORMAT>

The abstract:
{PAPER_CONTENT}
""".strip()

# =============================================================================================================================

# Alternative prompts for analyzing subsections and then creating a full paper summary
SECTION_ANALYSIS_PROMPT = """
Your task is to analyze the provided paper section and gather all the relevant findings as required below.
You should always output a numbered list of findings. Be objective, thorough, and as concise as possible in your analysis, 
ONLY including points that you are absolutely certain of. Include as many relevant and accurate details as you can.
If the section contains unclear/random text, then ignore it.

Use the following questions to guide your analysis ONLY if applicable to the relevant section:

1. What should be captured from the research? Include any formulas here if present
2. What should be captured around the study design? e.g., population (if applicable), methodology
3. What outcomes of the research are important to address the topic?
4. What is the quality of the research? e.g., limitations, potential improvements

Structure your analysis in the following way (exclude the <FORMAT> parts):

<FORMAT>

## SECTION NAME
<name of the section if provided, else estimate it based on the content>

## MAIN FINDINGS
<numbered list of points about findings IF present, otherwise return "none">

## STUDY DESIGN
<numbered list of point about study design IF present, otherwise return "none">

## OUTCOMES
<numbered list about outcomes IF present, otherwise return "none">

## QUALITY
<numbered list about the study quality IF present, otherwise return "none">

</FORMAT>

The section section:
{SECTION_CONTENT}
""".strip()

# =============================================================================================================================

PAPER_ANALYSIS_PROMPT = """
Your task is to combine the analyses in the previous messages into one summary.
You should always output a numbered list of findings. Be objective, thorough, and as concise as possible in your analysis, 
ONLY including points that you are absolutely certain of. Include as many relevant and accurate details as you can.

Use the following questions to guide your analysis:

1. What should be captured from the research? Include any formulas here if present
2. What should be captured around the study design? e.g., population (if applicable), methodology
3. What outcomes of the research are important to address the topic?
4. What is the quality of the research? e.g., limitations, potential improvements

Structure your analysis in the following way (exclude the <FORMAT> parts):

<FORMAT>

## MAIN FINDINGS
<numbered list of points about findings>

## STUDY DESIGN
<numbered list of point about study design>

## OUTCOMES
<numbered list about outcomes>

## QUALITY
<numbered list about the study quality>

</FORMAT>
""".strip()

# =============================================================================================================================

RETRIEVAL_PROMPT = """
Your task is to define query terms for the topic to search for relevant literature.
Think step-by-step on which terms would be most appropriate for a reproducible systematic literature review.
Structure your analysis in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for what terms would be best. Think step-by-step>

## TERMS
<the list of terms separated by commas as normal text>

</FORMAT>
""".strip()
