# The writer prompt(s)
# First we have the prompts for the outline writer
WRITER_PROMPT = """
You are a skilled research writer who synthesizes information from multiple sources to create coherent, insightful research documents.
Your writing should:
1. Be clear, concise, and academically rigorous.
2. Properly cite sources using the titles.
3. Present balanced viewpoints.
4. Connect ideas across different papers.
5. Highlight contradictions or agreements between different sources.

If you are instructed to format with [TITLE_1], use the ACTUAL paper title. For example:
"There are three categories when discussing social issues [Social Issues: A Modern Perspective]."
""".strip()

# =============================================================================================================================

DRAFT_OUTLINE_PROMPT = """
Your task is to create a systematic literature review outline with the following specifications:

Topic: {REVIEW_TOPIC}
Number of Sections: {NUM_SECTIONS} (try to at least keep it close, BUT base it more on how much needs to be written)
Number of Subsections per Section: {NUM_SUBSECTIONS} (try to at least keep it close, BUT base it more on how much needs to be written)

Determine what type of review structure would be most appropriate.
Base the contents off of these paper findings, and cite the relevant papers in each description:

{PAPER_ANALYSES}

Include an introduction and conclusion section.
Structure your response in the following way (exclude the <FORMAT> parts and try to use
less sub-subsections if possible):

<FORMAT>

# <a section title>
<the section's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

## <a subsection if needed>
<the subsection's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

### <a sub-subsection if needed>
<the subsection's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

# <a section title>
<the section's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

## <a subsection if needed>
<the subsection's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

...

</FORMAT>

You are allowed to use multiple references for a section. Ensure that ALL subsections (##) have a main section above them.
Do NOT include a title or anything else.
""".strip()

# =============================================================================================================================

MERGE_OUTLINES_PROMPT = """
You have created the below literature review outlines. Merge these outlines into one comprehensive outline with the following specifications:

Topic: {REVIEW_TOPIC}
Number of Sections: {NUM_SECTIONS} (try to at least keep it close, BUT base it more on how much needs to be written)
Number of Subsections per Section: {NUM_SUBSECTIONS} (try to at least keep it close, BUT base it more on how much needs to be written)

Include an introduction and conclusion section. If only one outline exists, then you only need to make some adjustments if necessary:

---
{OUTLINES}
---

Structure your response in the following way (exclude the <FORMAT> parts and try to use less sub-subsections if possible):

<FORMAT>

# <a section title>
<the merged section's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

## <a subsection if needed>
<the merged subsection's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

### <a sub-subsection if needed>
<the merged sub-subsection's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

# <a section title>
<the merged section's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

## <a subsection if needed>
<the merged subsection's brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

...

</FORMAT>

Ensure you mention and use the citations correctly (the titles in square brackets `[]`) in the descriptions.
Ensure that ALL subsections (##) have a main section above them.
Do NOT include a title or anything else.
""".strip()

# =============================================================================================================================

FIX_OUTLINE_PROMPT = """
You initially created the literature review outline in the first message, which has received feedback from a reviewer.
Using the feedback provided and experience gained, reflect on your performance and create a NEW and improved outline. Use ALL the
provided feedback and make sure everything is addressed.

Use the following specifications:

Topic: {REVIEW_TOPIC}
Number of Sections: {NUM_SECTIONS} (try to at least keep it close, BUT base it more on how much needs to be written)
Number of Subsections per Section: {NUM_SUBSECTIONS} (try to at least keep it close, BUT base it more on how much needs to be written)

As a reminder, these were the outlines you used for merging. Try to integrate as much as possible from these outlines:

---
{OUTLINES}
---

Include an introduction and conclusion section. 
Structure your response in the following way (exclude the <FORMAT> parts and try to use less sub-subsections if possible):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for what you need to improve and where, ideally with brief examples. Think step-by-step>

# <a NEW section title>
<the section's NEW brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

## <a NEW subsection if needed>
<the subsection's NEW brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

### <a NEW sub-subsection if needed>
<the subsection's NEW brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

# <a NEW section title>
<the section's NEW brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

## <a NEW subsection if needed>
<the subsection's NEW brief description with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.])>

...

</FORMAT>

Ensure you mention and use the citations correctly (the titles in square brackets `[]`) in the descriptions.
Ensure that ALL subsections (##) have a main section above them.
Do NOT include a title or anything else.
""".strip()

# =============================================================================================================================

# Then for the content writer
# (the guidelines for citing were adapted from the AutoSurvey repository.)
WRITE_CONTENT_PROMPT = """
Your task is to write the contents for a literature review section with the following specifications:

Section Title: {SECTION_TITLE}
Description: {SECTION_DESCRIPTION}
Is a subsection of: {HEADER_HIGH_LEVEL} (take this into consideration when writing the content)
References (note that you can leave some of these out if the content is not useful, but try to include at least most of it):

---
{PAPER_ANALYSES}
---

You may include multiple paragraphs in the content, but NO summary.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

# <the section title>
<the section content with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.]).
Include at LEAST {SECTION_LEN} words EXCLUDING references. If you must include a summary or commentary, start WITHOUT the phrase "In summary",
"In conclusion", or anything similar.>

</FORMAT>

Make sure to cite papers ONLY in the format specified above. Ensure the papers titles are spelled EXACTLY the same when citing.
Do NOT include separate subsections. Ensure ALL reference titles retain their exact spelling.

ALWAYS cite in the following scenarios if applicable:
1. When summarizing the existing literature.
2. When discussing specific theories, models, or data.
3. When comparing or contrasting different findings.
4. When pointing out gaps your survey addresses.
5. When referring to the creators of methodologies you employ.
6. To back up your conclusions and arguments.
7. When referencing studies related to proposed future research directions.

Do NOT include anything else.
""".strip()

# =============================================================================================================================

WRITE_CONCLUSION_PROMPT = """
Your task is to write the conclusion for a literature review section based on the following content:

{REVIEW_CONTENT}

You may include multiple paragraphs in your conclusion.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

# Conclusion
<the conclusion content. AVOID citing here>

</FORMAT>

Do NOT include anything else.
""".strip()

# =============================================================================================================================

WRITE_TITLE_ABSTRACT_PROMPT = """
Your task is to write the title and abstract for a literature review section based on the following content:

---
{REVIEW_CONTENT}
---

The length of the abstract should be roughly 150-250 words.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

# Title
<the review title>

# Abstract
<the abstract content>

</FORMAT>

Do NOT include anything else.
""".strip()

# =============================================================================================================================

FIX_CONTENT_PROMPT = """
The literature review section you wrote has received feedback from a reviewer.
Using the feedback provided and experience gained, reflect on your performance and create a NEW and improved section. Do NOT
repeat the content you wrote before too much and use ONLY the provided feedback relating to your section. Make sure EVERYTHING is addressed.
As a reminder, these are the specifications you need to use for the section content:

Section Title: {SECTION_TITLE}
Description: {SECTION_DESCRIPTION}
Is a subsection of: {HEADER_HIGH_LEVEL} (take this into consideration when writing the content)
References (note that you can leave some of these out if the content is not useful, but try to include at least most of it):

---
{PAPER_ANALYSES}
---

You may include multiple paragraphs in the content.
Do NOT include separate subsections. Ensure ALL reference titles retain their exact spelling.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for what you need to improve for this section and where, ideally with brief examples. Think step-by-step>

# <the section title>
<the NEW section content with references (USE THE PAPER TITLES LIKE SO: [TITLE_1 | TITLE_2 | TITLE_3 and more if needed, etc.]).
Include at LEAST {SECTION_LEN} words EXCLUDING references. If you must include a summary or commentary, start WITHOUT the phrase "In summary",
"In conclusion", or anything similar.>

</FORMAT>

Make sure to cite papers ONLY in the format specified above. Ensure the papers titles are spelled EXACTLY the same when citing.

ALWAYS cite in the following scenarios if applicable:
1. When summarizing the existing literature.
2. When discussing specific theories, models, or data.
3. When comparing or contrasting different findings.
4. When pointing out gaps your survey addresses.
5. When referring to the creators of methodologies you employ.
6. To back up your conclusions and arguments.
7. When referencing studies related to proposed future research directions.

Do NOT include anything else.
""".strip()

# =============================================================================================================================

FIX_CONCLUSION_PROMPT = """
The conclusion you wrote has received feedback from a reviewer.
Using the feedback provided and experience gained, reflect on your performance and create a NEW and improved conclusion. Do NOT
repeat the content you wrote before too much and use ONLY the provided feedback relating to the conclusion. Make sure EVERYTHING is addressed.
As a reminder, here is the content to base the conclusion off of:

---
{REVIEW_CONTENT}
---

Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for what you need to improve for this section and where, ideally with brief examples. Think step-by-step>

# {CONCLUSION_TITLE}
<the NEW conclusion content. AVOID citing here>

</FORMAT>

Do NOT include anything else.
""".strip()

# =============================================================================================================================

FIX_TITLE_ABSTRACT_PROMPT = """
The title and abstract you wrote has received feedback from a reviewer.
Using the feedback provided and experience gained, reflect on your performance and create a NEW and improved title and abstract. Do NOT
repeat the content you wrote before too much and use ONLY the provided feedback relating to the title and abstract. Make sure EVERYTHING is addressed.
As a reminder, here is the content to base the title and abstract off of:

---
{REVIEW_CONTENT}
---

The length of the abstract should be roughly 150-250 words.
Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for what you need to improve for this section and where, ideally with brief examples. Think step-by-step>

# Title
<the NEW review title>

# Abstract
<the NEW abstract content>

</FORMAT>

Do NOT include anything else.
""".strip()
