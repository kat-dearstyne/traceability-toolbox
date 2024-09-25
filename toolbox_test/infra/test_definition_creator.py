import os
from copy import deepcopy

from toolbox.data.managers.trainer_dataset_manager import TrainerDatasetManager
from toolbox.data.tdatasets.dataset_role import DatasetRole
from toolbox.infra.experiment.object_creator import ObjectCreator
from toolbox.infra.experiment.variables.typed_definition_variable import TypedDefinitionVariable
from toolbox_test.base.paths.base_paths import toolbox_TEST_DATA_PATH, toolbox_TEST_OUTPUT_PATH
from toolbox_test.base.tests.base_test import BaseTest


class TestDefinitionCreator(BaseTest):
    JOB_ARGS_DEFINITION = {
        "output_dir": toolbox_TEST_OUTPUT_PATH
    }
    MODEL_MANAGER_DEFINITION = {
        "model_path": "roberta-base"
    }
    DATASET_CREATOR_DEFINITION = {
        TypedDefinitionVariable.OBJECT_TYPE_KEY: "TRACE",
        "project_reader": {
            TypedDefinitionVariable.OBJECT_TYPE_KEY: "STRUCTURE",
            "project_path": os.path.join(toolbox_TEST_DATA_PATH, "structure")
        }
    }
    DATASET_MANAGER_DEFINITION = {
        "eval_dataset_creator": {
            TypedDefinitionVariable.OBJECT_TYPE_KEY: "TRACE",
            **DATASET_CREATOR_DEFINITION
        }
    }
    TRAINER_ARGS_DEFINITION = {"output_dir": toolbox_TEST_OUTPUT_PATH}
    DEFINITION = {
        "task": "PREDICT",
        "job_args": JOB_ARGS_DEFINITION,
        "model_manager": MODEL_MANAGER_DEFINITION,
        "trainer_dataset_manager": DATASET_MANAGER_DEFINITION,
        "trainer_args": TRAINER_ARGS_DEFINITION
    }

    def test_trainer_creation(self):
        definition = {
            "train_dataset_creator": deepcopy(self.DATASET_CREATOR_DEFINITION)
        }
        trainer_dataset_manager = ObjectCreator.create(TrainerDatasetManager, override=True, **definition)
        self.verify_trainer_dataset_manager(trainer_dataset_manager)

    def verify_trainer_dataset_manager(self, trainer_dataset_manager: TrainerDatasetManager,
                                       target_role: DatasetRole = DatasetRole.TRAIN):
        roles = [e for e in DatasetRole]
        roles.remove(target_role)
        for dataset_role in roles:
            self.assertIsNone(trainer_dataset_manager[dataset_role])
        dataset = trainer_dataset_manager[target_role]

        self.assertEqual(len(dataset._pos_link_ids), 4)
        self.assertEqual(len(dataset._neg_link_ids), 4)
        self.assertEqual(len(dataset.trace_df), 8)
