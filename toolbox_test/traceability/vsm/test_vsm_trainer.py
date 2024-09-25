import uuid
from typing import Dict, List

from toolbox.constants.hugging_face_constants import Metrics
from toolbox.data.managers.trainer_dataset_manager import TrainerDatasetManager
from toolbox.data.tdatasets.dataset_role import DatasetRole
from toolbox.infra.experiment.object_creator import ObjectCreator
from toolbox.infra.experiment.variables.typed_definition_variable import TypedDefinitionVariable
from toolbox.jobs.job_result import JobResult
from toolbox.traceability.vsm.vsm_trainer import VSMTrainer
from toolbox_test.base.assertions import TestAssertions
from toolbox_test.base.tests.base_trace_test import BaseTraceTest
from toolbox_test.test_data.test_data_manager import TestDataManager


class TestVSMTrainer(BaseTraceTest):
    EXPECTED_PREDICTION_SIZE = TestDataManager.get_n_candidates()
    TEST_METRIC_DEFINITION = [["map", ["map"]],
                              ["classification", ["precision", "recall", "f1", "f2"]]]
    TEST_METRICS_NAMES = [m for m, aliases in TEST_METRIC_DEFINITION]

    def test_perform_prediction(self):
        test_trace_trainer = self.get_custom_trace_trainer(dataset_container_args={"val_dataset_creator": None})
        test_trace_trainer.perform_training()
        trace_prediction_output = test_trace_trainer.perform_prediction()
        trace_prediction_job_result = JobResult(job_id=uuid.uuid4(), body=trace_prediction_output)
        eval_dataset = test_trace_trainer.trainer_dataset_manager[DatasetRole.EVAL]
        TestAssertions.verify_prediction_output(self, trace_prediction_job_result, eval_dataset, base_score=0.0)
        self.assert_metrics(trace_prediction_output.metrics)

    def get_custom_trace_trainer(self, dataset_container_args: Dict = None):
        trainer_dataset_manager = create_trainer_dataset_manager([DatasetRole.EVAL], **dataset_container_args)
        return VSMTrainer(trainer_dataset_manager=trainer_dataset_manager, metrics=self.TEST_METRICS_NAMES, select_predictions=False)

    def assert_metrics(self, metrics: Metrics):
        """
        Verifies that metrics contains all the desired metrics.
        :param metrics: The metrics object being checked.
        """
        for metric, aliases in self.TEST_METRIC_DEFINITION:
            for alias in aliases:
                self.assertIn(alias, metrics)


def create_trainer_dataset_manager(additional_roles: List[DatasetRole] = None, val_percentage: float = 0.3,
                                   **kwargs) -> TrainerDatasetManager:
    """
    Creates dataset manager containing datasets in roles.
    :param additional_roles: Additional dataset roles to include.
    :param val_percentage: Percentage of data to use for validation.
    :param kwargs: Dictionary of properties to overwrite in dataset manager definition.
    :return: Dataset manager created.
    """
    if additional_roles is None:
        additional_roles = [DatasetRole.VAL, DatasetRole.EVAL]
    if kwargs is None:
        kwargs = {}
    trainer_dataset_manager_definition = {**kwargs}
    if DatasetRole.EVAL in additional_roles and _to_creator_param(DatasetRole.EVAL) not in trainer_dataset_manager_definition:
        trainer_dataset_manager_definition[_to_creator_param(DatasetRole.EVAL)] = {
            TypedDefinitionVariable.OBJECT_TYPE_KEY: "TRACE",
            **ObjectCreator.dataset_creator_definition
        }
    if DatasetRole.VAL in additional_roles and _to_creator_param(DatasetRole.VAL) not in trainer_dataset_manager_definition:
        trainer_dataset_manager_definition[_to_creator_param(DatasetRole.VAL)] = {
            TypedDefinitionVariable.OBJECT_TYPE_KEY: "SPLIT",
            "val_percentage": val_percentage
        }
    return ObjectCreator.create(TrainerDatasetManager, **trainer_dataset_manager_definition)


def _to_creator_param(role: DatasetRole):
    return f"{role.name.lower()}_dataset_creator"
