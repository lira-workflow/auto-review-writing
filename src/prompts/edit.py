# The editor prompt(s)
EDITOR_PROMPT = """
You are an expert literature review editor.
""".strip()

# =============================================================================================================================

PAPER_EDIT_PROMPT = """
Your task is to edit the following draft literature review into a publishable paper.
Use the following points as a basis for editing:

1. Ensure the actual content remains the same.
2. When using abbreviations, ensure ONLY the first instance is written in full with the abbreviation in brackets after. For the rest,
just use the abbreviation, especially if it shows up more than 3 times.
3. Improve the flow and cohesion between sections WITHOUT changing the sections themselves.
4. Vary the vocabulary by reducing the repetition of certain terms.
5. Adjust the end of each section barring the conclusion itself so that it is NOT a summary/conclusion of its own contents.
6. Do NOT remove the citations used (the paper titles in square brackets) and ensure ALL their spellings remains the same.

This is the paper to edit:
---
{DRAFT_PAPER}
---

Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

# <the review title>

# <the edited review abstract>

===

# <a section header>

# <the edited content>

===

...

===

# <a section header>

# <the edited content>

</FORMAT>

Return ONLY the final edited paper (all sections with the title/abstract) and nothing else, ensuring that all sections are present.
Remember to add "===" between sections.
""".strip()

# =============================================================================================================================

PAPER_COMPLETE_EDIT_PROMPT = """
You are currently editing a review paper, but the edit maybe got cutoff. This is what you have written so far:

---
{EDITED_PAPER}
---

If the content was cutoff, please complete the edit. If not, then return "Done" and nothing else.
Specifically, pay attention if the LAST SENTENCE of the review has the necessary punctuation and all AND/OR if the conclusion is there or not.
Use the following points as a basis for editing:

1. Ensure the actual content remains the same.
2. When using abbreviations, ensure ONLY the first instance is written in full with the abbreviation in brackets after. For the rest,
just use the abbreviation, especially if it shows up more than 3 times.
3. Improve the flow and cohesion between sections WITHOUT changing the sections themselves.
4. Vary the vocabulary by reducing the repetition of certain terms (i.e., "In summary", "In conclusion..."). Remove redundancy.
5. Adjust the end of each section barring the conclusion itself so that it is NOT a summary/conclusion of its own contents.
6. Do NOT remove the citations used (the paper titles in square brackets).

This is the paper to edit:
---
{DRAFT_PAPER}
---

Structure your response in the following way (exclude the <FORMAT> parts and include ONLY the remainder which you can identify
from the original article):


<FORMAT>

# MISSING SECTIONS

<numbered list of the sections in the original paper missing from the edit. Make sure to include the conclusion if present>

===

# <a remaining section header>

<the remaining edited content>

===

...

===

# <a remaining section header>

<the remaining edited content>

</FORMAT>

Return ONLY the missing part(s) and nothing else, ensuring that all sections are present when everything is combined.
""".strip()

# =============================================================================================================================

PAPER_EDIT_REVISE_PROMPT = """
An edited paper you made has been evaluated by a reviewer. Based on the feedback provided, edit the literature review paper to create
a NEW final edit. Do NOT repeat the content you wrote before, and focus on making sure EVERYTHING is addressed.
Use the following points as a basis for editing:

1. Ensure the actual content remains the same.
2. When using abbreviations, ensure ONLY the first instance is written in full with the abbreviation in brackets after. For the rest,
just use the abbreviation, especially if it shows up more than 3 times.
3. Improve the flow and cohesion between sections WITHOUT changing the sections themselves.
4. Vary the vocabulary by reducing the repetition of certain terms.
5. Adjust the end of each section barring the conclusion itself so that it is NOT a summary/conclusion of its own contents.
6. Do NOT remove the citations used (the paper titles in square brackets) and ensure ALL their spellings remains the same.

This is the paper to edit:
---
{DRAFT_PAPER}
---

Structure your response in the following way (exclude the <FORMAT> parts):

<FORMAT>

## THOUGHTS
<a list of thoughts you have for what you need to improve for this section and where, ideally with brief examples. Think step-by-step>

<the review title>

<the NEW edited review abstract>

===

# <a section header>

<the NEW edited content>

===

...

===

# <a section header>

<the NEW edited content>

===
</FORMAT>

Return ONLY the final edited paper (all sections with the title/abstract) and nothing else, ensuring that all sections are present.
""".strip()

# =============================================================================================================================

PAPER_COMPLETE_REVISE_PROMPT = """
You are currently revising a review paper edit, but the edit maybe got cutoff. This is what you have written so far, alongside
your thoughts on what need to be improved:

---
{EDITED_PAPER}
---

If the content was cutoff, please complete the edit while implementing the improvements. If not, then return "Done" and nothing else.
Specifically, pay attention if the LAST SENTENCE of the review has the necessary punctuation and all AND/OR if the conclusion is there or not.
Use the following points as a basis for editing:

1. Ensure the actual content remains the same.
2. When using abbreviations, ensure ONLY the first instance is written in full with the abbreviation in brackets after. For the rest,
just use the abbreviation, especially if it shows up more than 3 times.
3. Improve the flow and cohesion between sections WITHOUT changing the sections themselves.
4. Vary the vocabulary by reducing the repetition of certain terms.
5. Adjust the end of each section barring the conclusion itself so that it is NOT a summary/conclusion of its own contents.
6. Do NOT remove the citations used (the paper titles in square brackets) and ensure ALL their spellings remains the same.

This is the paper to edit:
---
{DRAFT_PAPER}
---

Structure your response in the following way (exclude the <FORMAT> parts and include ONLY the remainder which you can identify
from the original article):

<FORMAT>

# MISSING SECTIONS

<numbered list of the sections in the original paper missing from the edit. Make sure to include the conclusion if present>

===

# <a remaining section header>

<the remaining revised edited content>

===

...

===

# <a remaining section header>

<the remaining revised edited content>

</FORMAT>

Return ONLY the missing part(s) and nothing else, ensuring that all sections are present when everything is combined.
Remember to add "===" between sections.
""".strip()
