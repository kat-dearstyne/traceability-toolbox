from copy import deepcopy
from typing import Dict, Type, TypeVar

from toolbox.constants.dataset_constants import VALIDATION_PERCENTAGE_DEFAULT
from toolbox.data.creators.trace_dataset_creator import TraceDatasetCreator
from toolbox.data.managers.trainer_dataset_manager import TrainerDatasetManager
from toolbox.data.processing.augmentation.data_augmenter import DataAugmenter
from toolbox.data.readers.api_project_reader import ApiProjectReader
from toolbox.infra.experiment.definition_creator import DefinitionCreator
from toolbox.infra.experiment.variables.typed_definition_variable import TypedDefinitionVariable
from toolbox.jobs.job_args import JobArgs
from toolbox.llm.args.hugging_face_args import HuggingFaceArgs
from toolbox_test.base.paths.base_paths import toolbox_TEST_OUTPUT_PATH
from toolbox_test.test_data.test_data_manager import TestDataManager

ObjectType = TypeVar("ObjectType")


class ObjectCreator:
    DATASET_ARGS_PARAMS = {
        "validation_percentage": VALIDATION_PERCENTAGE_DEFAULT
    }

    augmenter_definition = {"steps":
                                {"*": [[], [{"object_type": "RESAMPLE"}]]}
                            }

    trainer_args_definition = {
        "output_dir": toolbox_TEST_OUTPUT_PATH,
        "num_train_epochs": 1,
        "metrics": ["classification", "map"]
    }
    job_args_definition = {"output_dir": toolbox_TEST_OUTPUT_PATH}
    api_project_reader = {
        "api_definition": {
            "artifacts": TestDataManager.get_artifacts(),
            "layers": TestDataManager.get_path([TestDataManager.Keys.LAYERS]),
            "links": TestDataManager.get_path(TestDataManager.Keys.TRACES)
        }
    }
    dataset_creator_definition = {
        "project_reader": {
            TypedDefinitionVariable.OBJECT_TYPE_KEY: "API",
            **api_project_reader,
            "overrides": {
                "ALLOWED_ORPHANS": 2
            }
        }
    }

    trainer_dataset_manager_definition = {
        "train_dataset_creator": {
            TypedDefinitionVariable.OBJECT_TYPE_KEY: "TRACE",
            **dataset_creator_definition
        }
    }

    SUPPORTED_OBJECTS = {
        HuggingFaceArgs: trainer_args_definition,
        JobArgs: job_args_definition,
        TraceDatasetCreator: dataset_creator_definition,
        TrainerDatasetManager: trainer_dataset_manager_definition,
        DataAugmenter: augmenter_definition,
        ApiProjectReader: api_project_reader
    }

    @staticmethod
    def create(class_type: Type[ObjectType], override=False, **kwargs) -> ObjectType:
        """
        Creates an object of the given type using any additional arguments provided
        :param class_type: The type of object to create
        :param override: Will override default args if True
        :param kwargs: Additional arguments to use for intialization
        :return: The object
        """
        kwargs = deepcopy(kwargs)
        if override:
            args = kwargs
        else:
            args = ObjectCreator.get_definition(class_type)
            args = deepcopy(args)
            args.update(kwargs)
        return DefinitionCreator.create(class_type, args)

    @staticmethod
    def get_definition(class_type: Type[ObjectType]) -> Dict:
        """
        Gets the definition for instantiating an object
        :param class_type: The type of object to get a definition for
        :return: The definition
        """
        if class_type in ObjectCreator.SUPPORTED_OBJECTS:
            return deepcopy(ObjectCreator.SUPPORTED_OBJECTS[class_type])

        raise ValueError("Unable to find definition for:" + class_type)
