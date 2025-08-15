# Base variables to use
import os

# Aliases
ALIASES = {"srg": "SciReviewGen"}

# For tqdm display
BAR_FORMAT = "{l_bar}{bar:12}{r_bar}{bar:-10b}"

# Default variables
LLM_SEED = 42
DATA_FOLDER = "data"
TEMP_DIR = "temp_analysis"
BEGIN_PROMPT = "START: Begin generation for article {REVIEW_ID}"
N_JOBS = -4
GROUP_SIZE = 25
MAX_ITEMS = 50
TEMPERATURE = 0.0
CONTEXT_SIZE = 128_000
NUM_SECTIONS = 8
NUM_SUBSECTIONS = 4
SECTION_LEN = 1_000
SETTING_NAME = "abs_same_edit"

# The (Azure)OpenAI variables
ENCODING_NAME = "gpt-4o-mini"
MODEL_VER = "gpt4o-mini-240718"
BASE_URL = os.getenv("AZURE_OPENAI_URL_BASE")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
API_KEY = os.getenv("OPENAI_ORGANIZATION_KEY")

# The node names
SETUP_STATE = "setup_state"
ANALYZE_ALL_PAPERS = "analyze_all_papers"
PREPARE_GROUPS = "prepare_groups"
RESEARCHER_GROUP = "researcher_group"
DRAFT_OUTLINE = "draft_outline"
REVIEW_CONTENT = "review_content"
WRITE_CONTENT = "write_content"
EDIT_CONTENT = "edit_content"
