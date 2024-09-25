from toolbox.data.tdatasets.pre_train_dataset import PreTrainDataset
from toolbox_test.base.paths.base_paths import toolbox_TEST_VOCAB_PATH
from toolbox_test.base.tests.base_test import BaseTest


class TestPreTrainDataset(BaseTest):

    def get_pre_train_dataset(self):
        return PreTrainDataset(toolbox_TEST_VOCAB_PATH, block_size=128)
