import gc, os, re
import pandas as pd
from tqdm import tqdm
from argparse import Namespace
from typing import Optional, Tuple
from joblib import Parallel, delayed
from utils.pbar_utils import tqdm_joblib
from utils.text_utils import format_text_from_table
from baseline_and_mass.utils.prompting import prompt_llm
from baseline_and_mass.utils.file_utils import pattern_selector
from baseline_and_mass.prompts import (
    TITLE_ABSTRACT_PROMPT,
    REFERENCE_ANALYSIS_PROMPT,
    CHAPTER_CONTENT_PROMPT,
    CONCLUSION_PROMPT,
)
from baseline_and_mass.utils.file_utils import (
    read_file,
    save_to_file,
    pattern_selector,
    check_paper_in_subset,
)


# Set this to remove the text cutoff from pandas
pd.set_option("display.max_colwidth", None)


class DocumentProcessor:
    """
    Document processor class, which serves as the agent base class for the system.
    """

    def __init__(self, args: Namespace, prompt: str, description: str):
        self.args = args
        self.n_jobs = args.n_jobs
        self.dataset = args.dataset
        self.prompt_template = prompt
        self.description = description

    def process_single_document(
        self,
        input_file: str,
        output_dir: str,
        input_suffix: str,
        output_suffix: str,
        additional_processing: Optional[callable] = None,
    ) -> Tuple[str, str]:
        """
        Generic function to process a single document with an LLM.

        Parameters:
        input_file - the input file to read from (contains the information the LLM needs).
        output_dir - the directory to save the generated files to.
        input_suffix - the suffix to use for determining which file type to read.
        output_suffix - the suffix to use for saving the file as.
        prompt_template - the prompt template to use for the specific task (formattable string).
        additional_processing - any additional processes to perform on the LLM output.

        Returns:
        a Tuple of (input_file, generated_file_path) or (input_file, None) if failed.
        """

        try:
            # Read the input content
            content = read_file(input_file)

            # Format the prompt
            prompt_content = self.prompt_template.format(content=content)

            # Get the LLM response
            response = prompt_llm(
                prompt_content, input_file, self.prompt_template, self.args
            )

            # Apply any additional processing
            if additional_processing:
                response = additional_processing(response)

            # Extract filename portions using regex
            pattern = pattern_selector(self.dataset, input_suffix)
            match = re.search(pattern, input_file)

            if not match:
                raise ValueError(
                    f"Filename `{input_file}` does not match the expected pattern"
                )

            # Reformat Output Name Reformat
            new_filename = re.sub(input_suffix, output_suffix, match.group(0))
            new_filename = os.path.basename(new_filename)
            new_filename = os.path.join(output_dir, new_filename)

            # Save and return the output file
            output_file = save_to_file(response, new_filename)
            return (input_file, output_file)

        except Exception as e:
            print(f"An error occurred while processing `{input_file}`: {e}")
            return (input_file, None)

    def process_document_batch(
        self,
        input_dir: str,
        output_dir: str,
        input_suffix: str,
        output_suffix: str,
        use_subset: bool = True,
        additional_processing: Optional[callable] = None,
    ):
        """
        Process all matching documents in a directory in parallel.

        Parameters:
        input_dir - the input directory to read from (contains the files for the specific section).
        output_dir - the directory to save the resulting files to.
        input_suffix - the suffix to use for determining which file type to read.
        output_suffix - the suffix to use for saving the file as.
        prompt_template - the prompt template to use for the specific task.
        description - the task description to display to the user.
        use_subset - whether to use a subset of the data (should always be `True`).
        additional_processing - any additional processes to perform on the LLM output.
        """

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Get file list (only if from SciReviewGen)
        file_list = os.listdir(input_dir)
        if use_subset and self.dataset == "srg":
            file_list = check_paper_in_subset(
                file_list,
                self.args.n_samples,
                self.args.seed,
            )

        # Filter files and prepare the processing list
        processing_list = []
        skip_count = 0
        for filename in file_list:

            pattern = pattern_selector(self.dataset, input_suffix)
            match = re.search(pattern, filename)

            if match:
                new_filename = re.sub(input_suffix, output_suffix, match.group(0))
                output_file = os.path.join(output_dir, new_filename)

                # Skip if the output exists
                if os.path.exists(output_file):
                    skip_count += 1
                    continue

                # Add to the processing list
                input_path = os.path.join(input_dir, filename)
                processing_list.append(input_path)

        # Process files in parallel with a progress bar
        if processing_list:

            with tqdm_joblib(
                tqdm(
                    desc=self.description,
                    unit="file(s)",
                    total=len(processing_list),
                )
            ):

                _ = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(self.process_single_document)(
                        input_file=input_file,
                        output_dir=output_dir,
                        input_suffix=input_suffix,
                        output_suffix=output_suffix,
                        additional_processing=additional_processing,
                    )
                    for input_file in processing_list
                )

        print(f"Skipped {skip_count} existing files for {self.description.lower()}.\n")

        # Memory Leak prevention by deleting variables
        del file_list, processing_list
        gc.collect()


def clean_title_abstract(response: str) -> str:
    """
    Clean title and abstract response by removing text formatting.

    Parameters:
    response - the LLM response.

    Returns:
    the response without bold formatting and adjusted quotation marks.
    """

    response = re.sub(r"\*+", "", response)
    return re.sub('"', "", response)


# Main Processing Functions
def analyze_all_references(args: Namespace):
    processor = DocumentProcessor(
        args,
        prompt=REFERENCE_ANALYSIS_PROMPT,
        description="Generating Structured References",
    )
    processor.process_document_batch(
        input_dir=args.processed_dir,
        output_dir=args.structured_ref_dir,
        input_suffix="_references",
        output_suffix="_structured_references",
        use_subset=True,
    )


def generate_all_titles_and_abstracts(args: Namespace):
    processor = DocumentProcessor(
        args,
        prompt=TITLE_ABSTRACT_PROMPT,
        description="Generating Titles and Abstracts",
    )
    processor.process_document_batch(
        input_dir=args.structured_ref_dir,
        output_dir=args.title_abstract_dir,
        input_suffix="_structured_references",
        output_suffix="_title_and_abstract",
        use_subset=True,
        additional_processing=clean_title_abstract,
    )


def generate_all_chapter_contents(args: Namespace):
    processor = DocumentProcessor(
        args, prompt=CHAPTER_CONTENT_PROMPT, description="Generating Chapter Contents"
    )
    processor.process_document_batch(
        input_dir=args.structured_ref_dir,
        output_dir=args.chapter_contents_dir,
        input_suffix="_structured_references",
        output_suffix="_chapter_contents",
    )


def generate_all_conclusions(args: Namespace):
    processor = DocumentProcessor(
        args, prompt=CONCLUSION_PROMPT, description="Generating Conclusions"
    )
    processor.process_document_batch(
        input_dir=args.chapter_contents_dir,
        output_dir=args.conclusion_dir,
        input_suffix="_chapter_contents",
        output_suffix="_conclusion",
    )


def generate_csv(args: Namespace):
    """
    Compiles all files for each paper across generated folders.
    The output is then saved to a CSV file.

    Parameters:
    args - a Namespace containing the following variables which get read:

    base_dir - the folder containing folders of each document.
    title_abstract_dir - the directory of the title and abstracts.
    chapter_contents_dir - the directory of the chapter contents.
    conclusion_dir - the directory of the conclusions.
    processed_dir - the directory of the pre-processed (not structured) references.
    """

    # For progress visualization
    file_list = os.listdir(args.title_abstract_dir)
    if args.dataset == "srg":
        file_list = check_paper_in_subset(file_list, args.n_samples, args.seed)

    # Initialize an empty list to store the rows for the DataFrame
    rows = []

    # Process each file
    for filename in tqdm(
        file_list, desc="Integrating Files", unit="file(s)", total=len(file_list)
    ):

        pattern = pattern_selector(
            args.dataset, input_suffix="_title_and_abstract", use_group=True
        )
        match = re.search(pattern, filename)
        if not match:
            raise ValueError(f"Error: Cannot extract paper_id from filename {filename}")

        # Get the base paper_id and name
        paper_id = match.group(1)
        base_name = match.group(0)

        if args.dataset == "srg":
            match = re.search(r"^\d+\_subset\_\d+\_\d+", base_name)
            base_name = match.group(0)

        else:
            raise ValueError(f"Dataset `{args.dataset}` not supported!")

        title_and_abstract_path = os.path.join(args.title_abstract_dir, filename)
        title_and_abstract = read_file(title_and_abstract_path)
        title_and_abstract = title_and_abstract.strip()

        # Parse the title and abstract
        title_match = re.search(r"Title:\s*(.*)", title_and_abstract)
        abstract_match = re.search(
            r"Abstract:\s*(.*?)(?=\n\s*\n|$)", title_and_abstract
        )

        if title_match and abstract_match:
            title = title_match.group(1).strip()
            abstract = abstract_match.group(1).strip()

        else:
            raise ValueError(
                "Error: Title or Abstract not found in the expected format!"
            )

        # Read the files in the chapter_contents folder
        chapter_file = os.path.join(
            args.chapter_contents_dir, f"{base_name}_chapter_contents.txt"
        )

        chapter_contents = read_file(chapter_file)
        chapter_contents = chapter_contents.strip()

        # Remove all the * characters and extract chapter titles and contents
        chapter_contents = re.sub("\*+", "", chapter_contents)

        # Use regex to match chapter titles and content
        sections = re.findall(r"Section \d+: ([^\n]+)", chapter_contents)

        # Split the text into sections based on headers
        texts = re.split(r"Section \d+: [^\n]+", chapter_contents)[1:]

        # Clean the content by stripping leading/trailing whitespaces
        texts = [text.strip() for text in texts]

        # Read the files in the conclusion folder
        conclusion_file_path = os.path.join(
            args.conclusion_dir, f"{base_name}_conclusion.txt"
        )
        conclusion = read_file(conclusion_file_path)
        conclusion = conclusion.strip()

        # Add the conclusion section
        sections.append("Conclusion")
        texts.append(conclusion)

        # Append the row to the DataFrame
        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "section": str(sections),
                "text": str(texts),
            }
        )

    # Create a DataFrame from the collected rows
    df = pd.DataFrame(rows)

    # Write the DataFrame to a CSV file
    df.to_csv(args.final_papers_file, index=False)

    print(f"CSV file saved to '{args.final_papers_file}'! \n")

    # Memory Leak prevention by deleting variables
    del df, rows, texts, file_list, sections, chapter_contents
    gc.collect()


def generate_txt(args: Namespace):
    """
    Takes the generated articles (from `generate_csv`) and formats them into
    `.txt` files.

    Parameters:
    args - a Namespace containing the following variables which get read:

    n_samples - the number of samples used for selection.
    seed - the seed used for sample selection.
    dataset - the dataset used for testing (used for file saving and some processing).
    base_dir - the folder containing folders of each document.
    title_abstract_dir - the directory of the title and abstracts.
    chapter_contents_dir - the directory of the chapter contents.
    conclusion_dir - the directory of the conclusions.
    processed_dir - the directory of the pre-processed (not structured) references.
    final_papers_dir - the target directory to save the resulting CSV file to.
    """

    # Ensure the target directory exists
    if args.final_papers_dir != "" and not os.path.isdir(args.final_papers_dir):
        os.makedirs(args.final_papers_dir, exist_ok=True)

    # Read the paper files
    df = pd.read_csv(args.final_papers_file, index_col=False)
    data = list(df.iterrows())
    for _, paper in tqdm(data, desc="Formatting to .txt"):

        text = format_text_from_table(paper)
        filename = (
            f"{paper['paper_id']}_subset_{args.n_samples}_{args.seed}.txt"
            if args.dataset == "srg"
            else f"{paper['paper_id']}.txt"
        )
        output_file = os.path.join(args.final_papers_dir, filename)
        save_to_file(text, output_file)

    print("Paper files generated!")

    # Memory Leak prevention by deleting variables
    del df, data
    gc.collect()
