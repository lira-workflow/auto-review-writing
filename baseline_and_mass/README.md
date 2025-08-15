# MASS-Survey

> [!NOTE]
> The contents of this repository and README have been adapted to better suit the current project's needs.

MASS-Survey (Multi-Agent Scientific Survey) is an automated literature survey generation tool that leverages Large Language Models (LLMs) and Multi-Agent Systems (MAS) to produce comprehensive scientific literature surveys.

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://link.springer.com/chapter/10.1007/978-981-97-9443-0_14)
[![Conference](https://img.shields.io/badge/Conference-NLPCC%202024-blue)](https://link.springer.com/book/10.1007/978-981-97-9443-0)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of our NLPCC 2024 paper: "Generation of Scientific Literature Surveys Based on Large Language Models (LLM) and Multi-Agent Systems (MAS)". The system utilizes a pipeline approach to automatically generate literature surveys through multiple processing stages.

## Pipeline Architecture

The system operates through six sequential stages:

1. **Extract Stage** ([`extract.py`](extract.py))
   - Processes JSON input files
   - Extracts and structures reference information
   - Generates structured bibliography data

2. **Structure Stage** ([`structured_ref.py`](structured_ref.py))
   - Transforms bibliography data into structured chapter content
   - Implements initial content organization
   - Prepares framework for survey generation

3. **Title and Abstract Stage** ([`title_abstract.py`](title_abstract.py))
   - Generates chapter titles and abstracts
   - Ensures logical document structure
   - Creates content overview

4. **Content Stage** ([`chapter_content.py`](chapter_content.py))
   - Produces detailed chapter content
   - Expands on structured framework
   - Implements in-depth analysis

5. **Conclusion Stage** ([`conclusion.py`](conclusion.py))
   - Synthesizes overall findings
   - Generates comprehensive conclusions
   - Summarizes key contributions

6. **Integration Stage** ([`integrator.py`](integrator.py))
   - Merges all components
   - Ensures document coherence
   - Produces final survey output

## Installation

```bash

git clone https://github.com/may3rr/MASS-Survey.git
cd MASS-Survey
pip install -r requirements.txt

```

## Usage

1. Please set the OpenAI environment variables in your system if you have not done so already. Instructions are present [here](../README.md).

2. Prepare your input JSON files (or SciReviewGen dataset download) in the designated directory.

3. Run the script:

```bash

# Run this in the main directory (not in a subfolder)
python run_mass.py

```

## Requirements

- Python 3.8+
- OpenAI API key
- Required Python packages:
  - openai


## Citation
If you use this code for your research, please cite our paper:

```bibtex

@InProceedings{10.1007/978-981-97-9443-0_14,
      author="Qi, Ruihua
      and Li, Weilong
      and Lyu, Haobo",
      editor="Wong, Derek F.
      and Wei, Zhongyu
      and Yang, Muyun",
      title="Generation of Scientific Literature Surveys Based on Large Language Models (LLM) and Multi-Agent Systems (MAS)",
      booktitle="Natural Language Processing and Chinese Computing",
      year="2025",
      publisher="Springer Nature Singapore",
      address="Singapore",
      pages="169--180",
      abstract="With the rapid increase in the number and speed of scientific publications, researchers face significant time pressure when conducting literature reviews. This paper presents an automatic literature review generation method leveraging large language models (LLMs) and multi-agent systems (MAS). By designing multiple agent roles, including reference parsing, analysis, generation, and integration agents-this method fully utilizes the natural language processing capabilities of LLMs and the collaborative strengths of MAS to produce high-quality literature reviews. In the NLPCC2024 evaluation task, our method excelled in multiple automatic evaluation metrics (such as SoftHeadingRecall and ROUGE) and manual evaluations, showcasing its great potential for practical applications.",
      isbn="978-981-97-9443-0"
}

```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact:
- Email: lvhaobo@foxmail.com
- GitHub: [@may3rr](https://github.com/may3rr)