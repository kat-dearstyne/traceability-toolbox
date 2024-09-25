from os.path import abspath, dirname, join

toolbox_TEST_DIR_PATH = dirname(dirname(dirname(abspath(__file__))))
toolbox_TEST_DATA_PATH = join(toolbox_TEST_DIR_PATH, "test_data")

toolbox_TEST_OUTPUT_PATH = join(toolbox_TEST_DIR_PATH, "output")
# Files
toolbox_TEST_TESTPYTHON_PATH = join(toolbox_TEST_DATA_PATH, "test_python.py")
toolbox_TEST_VOCAB_PATH = join(toolbox_TEST_DATA_PATH, "test_vocab.txt")
