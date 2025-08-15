# All prompts for the baseline and from the original implementation
BASELINE_PROMPT = """
The following are some references and their abstracts. Please analyze these references and
write a systematic literature review based on it:

{content}

Please include the title, abstract, section headers, and conclusion. Mark the sections
so that they can be parsed. The output should be at least 3000 characters long excluding
the title, abstract, and conclusion. Make sure all parts are written. Do not include #.
Please cite the references used by including the number at the end of the sentence. For example: "sentence" [1].
Please follow this specific structure:

Title: [title_here]
Abstract: [abstract_here]

Section [section_number_here]: [section_header]

Conclusion: [conclusion_here]
"""

REFERENCE_ANALYSIS_PROMPT = """
The following are some references and their abstracts. Please analyze these references, find their logical relationships in a review, and classify them. You are not allowed to omit any references; you must classify all of them. Output structured section titles with the related references listed under each section title. Here is an example:

**Analysis of References and Logical Relationships**

**Section 1: Navigation and Route Planning**
- [1] M. Duckham, L. Kulik, "Simplest" Paths: Automated Route Selection for Navigation. This paper introduces the concept of "simplest" paths in contrast to shortest paths, focusing on reducing instruction complexity for human navigators. It proposes an algorithm with quadratic computation time and demonstrates its potential benefits for navigation systems, suggesting future cognitive studies to validate the computational results.

**Section 2: Exoplanet Research and Atmospheric Characterization**
- The reference provided seems to be a misnumbered continuation from another part of the text and does not have a clear title or number associated with the abstract provided. However, it discusses the measurement of exoplanet eclipse depths and the implications for the planet's atmosphere, indicating research into astrobiology or extrasolar planet characterization.

**Section 3: Music Analysis and Generation**
- [306] A. Anglade, R. Ramirez, S. Dixon, et al., Genre Classification Using Harmony Rules Induced from Automatic Chord Transcriptions. This study explores techniques for genre classification in music through harmony rules derived from automatic chord transcriptions.
- [123] T. E. Ahonen, K. Lemström, S. Linkola, Compression-based Similarity Measures in Symbolic, Polyphonic Music. This work focuses on developing compression-based similarity measures for symbolic, polyphonic music, contributing to the field of music information retrieval.

These classifications group the references into thematic sections based on their primary subjects, such as navigation technology, AI safety, exoplanetary science, music analysis and generation, linguistic and music theoretical frameworks, and interactive content creation.

Please start the analysis(Most Important!!!!Most Important!!!!Most Important!!!!: Do not group any multiple references into a single entry. Each reference must be listed as a separate entry, even if they cover related topics. Ensure that every reference is individually listed and described, avoiding any form of consolidation or summary of multiple references into one. Without hashtag#.
):

{content}
"""

TITLE_ABSTRACT_PROMPT = """
Based on the following section titles, please generate an appropriate title and a brief abstract for a review paper.

Section Titles: {content}

Title: [title_here]
Abstract: [abstract_here]

Please generate the paper title and abstract.
"""

# Prompt was modified for parsing purposes and in the following way:
# "... specified requirements. -> specified requirements. \nPlease cite ..." (until "[1].")
CHAPTER_CONTENT_PROMPT = """
The following are section titles and related references. Please write the text content for each section.

{content}

Please write the section content. Only the body text and the headers to be outputted, omitting all other parts. 
Do not include #, and do not reply in Markdown format. Respond as per the specified requirements. 
Please cite the references used by including the number at the end of the sentence. For example: "sentence" [1].
Ensure that each section's content is sufficiently detailed and provides comprehensive analysis. 
The output should be at least 3000 characters long. Without hashtag#. Here is an example:

Section 1: [section_title_here]

[chapter_content_here]
"""

CONCLUSION_PROMPT = """
Based on the following section contents, please write the Conclusion section for the review paper.

Section Contents:

{content}

Do not include #, Please write the Conclusion:
"""
