import gc, os, re
import pandas as pd
from typing import Dict, List, Tuple


# RegEx Pattern(s)
header_pattern = r"# (.+)\n"


def adjust_review(review: str, references: List) -> Tuple[str, Dict]:
    """
    Adjusts the references in a review and removes any residual formatting.

    Parameters:
    review - the literature review to adjust.
    references - the list of references used for the paper.

    Returns:
    a Tuple containing the review (text) with references converted to numbers and the references
    used as a dictionary.
    """

    # First remove stray abstracts
    review = re.sub(r"# Abstract\s+", "", review)

    # ...and the title in case it's formatted wrong
    review = re.sub(r"# Title\n", "# ", review)

    # Then convert the references into a dictionary
    ref_dict = {}
    for ref in references:

        ref_dict[ref["num"]] = ref["title"]

    # Then replace all citations with numbers
    for idx, title in ref_dict.items():

        title_esc = re.escape(title)  # For special characters
        review = re.sub(rf"\[{title_esc}\]", f"[{idx}]", review, flags=re.IGNORECASE)
        review = re.sub(
            rf"\[{title_esc}\s*\|", f"[{idx} |", review, flags=re.IGNORECASE
        )
        review = re.sub(
            rf"\|\s*{title_esc}\s*\|", f"| {idx} |", review, flags=re.IGNORECASE
        )
        review = re.sub(
            rf"\|\s*{title_esc}\]", f"| {idx}]", review, flags=re.IGNORECASE
        )

    # Now replace all "|" with semicolons
    output = re.sub(r" \| ", "; ", review)

    # Also fix the numbers which are in quotations (these are citations)
    output = re.sub(r'"(\d+)([^a-zA-Z\d\s:]?)"', r"[\1]\2", output)

    # ...and remove any leftover formatting
    output = re.sub(r"<\/?FORMAT>\n{0,2}", "", output)
    return output, ref_dict


def parse_review(review: str) -> Tuple[str, str, List[str], List[str]]:

    # Split the review by sections
    sections = re.split(r"===", review.strip())

    # In case of any weird formatting, remove empty strings
    sections = [section.strip() for section in sections if section != ""]

    # Check if the review has extra content leftover (from Chain-of-Thought)
    cot_match = re.search(r"\d+\.\s.+\n+", sections[0])
    if cot_match:
        sections[0] = re.sub(r"\d+\.\s.+\n+", "", sections[0])

    # Then get the sections
    title_abstract = sections[0].split("\n\n")
    title, abstract = re.sub(header_pattern, "", title_abstract[0]), title_abstract[1]

    # Clean the title and abstract
    title = re.sub("# ", "", title)
    abstract = re.sub(r"^#(?: Abstract)?\s?", "", abstract)

    section_list, text_list = [], []
    for section in sections[1:]:

        # In case the generation is incorrect (a section is too short)
        if len(section) <= 10:
            continue

        # Remove all hallucinated references
        citations = re.findall(r"\[(.*?)\]", section)
        for citation in citations:

            if len(citation) > 4:

                citation_esc = re.escape(citation)

                # Handle ONLY cases where it's in a bracket
                section = re.sub(
                    rf"\[(.*?){citation_esc}(.*?)\]", r"\[\1REDACTED\2\]", section
                )
                section = re.sub("\\\\", "", section)

        section_header = re.search(header_pattern, section)
        section_text = re.sub(header_pattern, "", section)

        # Handle if the header is missing
        header = "MISSING_HEADER" if section_header is None else section_header.group(1)
        section_list.append(header)
        text_list.append(section_text.strip())

    return title, abstract, section_list, text_list


def convert_save_to_table(entries: Dict, save_path: str):
    """
    Converts the reviews into a table and saves them (alongside the references).

    Parameters:
    entries - the literature review entries.
    save_path - the file location to save the results in.
    """

    # save_dir = os.path.join(args.)
    to_dict = {"paper_id": [], "title": [], "abstract": [], "section": [], "text": []}
    for idx, entry in entries.items():

        parsed_review = parse_review(entry[0])
        to_dict["paper_id"].append(idx)
        to_dict["title"].append(parsed_review[0])
        to_dict["abstract"].append(parsed_review[1])
        to_dict["section"].append(parsed_review[2])
        to_dict["text"].append(parsed_review[3])

    # Convert to a dataframe
    df = pd.DataFrame.from_dict(to_dict)

    # Save the output
    df.to_csv(save_path, index=False)

    # Memory leak prevention
    del df
    gc.collect()
