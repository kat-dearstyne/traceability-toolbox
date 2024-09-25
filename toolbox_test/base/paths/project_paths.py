from os.path import join

from toolbox_test.base.paths.base_paths import toolbox_TEST_DATA_PATH

# Prompts
toolbox_TEST_PROMPT_JSONL_PATH = join(toolbox_TEST_DATA_PATH, "repo")
# Repo
toolbox_TEST_PROJECT_REPO_PATH = join(toolbox_TEST_DATA_PATH, "repo")
toolbox_TEST_PROJECT_REPO_ONE_PATH = join(toolbox_TEST_PROJECT_REPO_PATH, "one")
toolbox_TEST_PROJECT_REPO_TWO_PATH = join(toolbox_TEST_PROJECT_REPO_PATH, "two")
# Structure
toolbox_TEST_PROJECT_STRUCTURE_PATH = join(toolbox_TEST_DATA_PATH, "structure")
# CSV
toolbox_TEST_PROJECT_CSV_PATH = join(toolbox_TEST_DATA_PATH, "csv", "test_csv_data.csv")
# SAFA
toolbox_TEST_PROJECT_SAFA_PATH = join(toolbox_TEST_DATA_PATH, "safa")
# PreTrainTrace
toolbox_TEST_PROJECT_PRETRAIN_PATH = join(toolbox_TEST_DATA_PATH, "pre_train_trace", "handbook.txt")
# Data Frame
toolbox_TEST_PROJECT_DATAFRAME_PATH = join(toolbox_TEST_DATA_PATH, "dataframe")
# prompt
toolbox_TEST_PROJECT_PROMPT_LHP_PATH = join(toolbox_TEST_DATA_PATH, "prompt", "lhp.jsonl")
toolbox_TEST_PROJECT_PROMPT_SAFA_PATH = join(toolbox_TEST_DATA_PATH, "prompt", "safa_proj.jsonl")

# Chunks
toolbox_TEST_PROJECT_CHUNK_PATH = join(toolbox_TEST_DATA_PATH, "chunker")
toolbox_TEST_PROJECT_CHUNK_TESTJAVA_PATH = join(toolbox_TEST_PROJECT_CHUNK_PATH, "test_java.java")
toolbox_TEST_PROJECT_CHUNK_TESTPYTHON_PATH = join(toolbox_TEST_DATA_PATH, "chunker/test_python.py")
# PreTrain
toolbox_TEST_PROJECT_PRE_TRAIN_PATH = join(toolbox_TEST_DATA_PATH, "pre_train")
# state
toolbox_TEST_PROJECT_STATE_PATH = join(toolbox_TEST_DATA_PATH, "state")
# Result reader
toolbox_TEST_PROJECT_RESULT_READER_PATH = join(toolbox_TEST_DATA_PATH, "result_reader")
# Cleaning
toolbox_TEST_PROJECT_CLEANING_PATH = join(toolbox_TEST_DATA_PATH, "cleaning")
toolbox_TEST_PROJECT_CLEANING_CPP = join(toolbox_TEST_PROJECT_CLEANING_PATH, "test.hpp")
toolbox_TEST_PROJECT_CLEANING_JAVA = join(toolbox_TEST_PROJECT_CLEANING_PATH, "test.java")
