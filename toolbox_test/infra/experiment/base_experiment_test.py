import json
import os
import random

from toolbox.constants.experiment_constants import OUTPUT_FILENAME
from toolbox.infra.experiment.variables.definition_variable import DefinitionVariable
from toolbox.infra.experiment.variables.experimental_variable import ExperimentalVariable
from toolbox.infra.experiment.variables.multi_variable import MultiVariable
from toolbox.infra.experiment.variables.typed_definition_variable import TypedDefinitionVariable
from toolbox.infra.experiment.variables.undetermined_variable import UndeterminedVariable
from toolbox.infra.experiment.variables.variable import Variable
from toolbox.traceability.output.trace_prediction_output import TracePredictionOutput
from toolbox_test.base.paths.base_paths import toolbox_TEST_OUTPUT_PATH
from toolbox_test.base.tests.base_test import BaseTest


class BaseExperimentTest(BaseTest):
    # TODO: REplace huggingface jobs with other jobs.
    accuracies = []
    EXPERIMENT_DEFINITION = DefinitionVariable({
        "steps": MultiVariable([
            DefinitionVariable({
                "jobs": MultiVariable([TypedDefinitionVariable({
                    TypedDefinitionVariable.OBJECT_TYPE_KEY: "HuggingFaceJob",
                    "task": "TRAIN",
                    "job_args": DefinitionVariable({
                    }),
                    "model_manager": DefinitionVariable({
                        "model_path": ExperimentalVariable([Variable("roberta-base"), Variable("bert-base-uncased")]),
                    }),
                    "trainer_dataset_manager": DefinitionVariable({
                        "train_dataset_creator": TypedDefinitionVariable({
                            TypedDefinitionVariable.OBJECT_TYPE_KEY: "TRACE",
                            "project_reader": TypedDefinitionVariable({
                                TypedDefinitionVariable.OBJECT_TYPE_KEY: "STRUCTURE",
                                "project_path": ExperimentalVariable([Variable("safa1"), Variable("safa2")])
                            })
                        })
                    }),
                    "trainer_args": DefinitionVariable({
                        "output_dir": Variable(toolbox_TEST_OUTPUT_PATH),
                        "num_train_epochs": ExperimentalVariable([Variable(100), Variable(200)])
                    })
                })]),
                "comparison_criterion": DefinitionVariable({
                    "metrics": Variable(["accuracy"])
                })
            }),
            DefinitionVariable({"jobs": MultiVariable([TypedDefinitionVariable({
                TypedDefinitionVariable.OBJECT_TYPE_KEY: "HuggingFaceJob",
                "task": "PredictTask",
                "job_args": DefinitionVariable({
                }),
                "model_manager": DefinitionVariable({
                    "model_path": UndeterminedVariable(),
                }),
                "trainer_dataset_manager": DefinitionVariable({
                    "eval_dataset_creator":
                        TypedDefinitionVariable({
                            TypedDefinitionVariable.OBJECT_TYPE_KEY: "TRACE",
                            "project_reader": TypedDefinitionVariable({
                                TypedDefinitionVariable.OBJECT_TYPE_KEY: "STRUCTURE",
                                "project_path": "SAFA"
                            })
                        }),
                }),
                "trainer_args": DefinitionVariable({
                    "output_dir": Variable(toolbox_TEST_OUTPUT_PATH),
                })
            })])
            })
        ]),
        "output_dir": toolbox_TEST_OUTPUT_PATH,
    })

    def setUp(self):
        super(BaseExperimentTest, self).setUp()
        self.accuracies = []

    @staticmethod
    def _load_step_output(output_file_path=None):
        if not output_file_path:
            output_file_path = os.path.join(toolbox_TEST_OUTPUT_PATH, OUTPUT_FILENAME)
        with open(output_file_path) as out_file:
            output = json.load(out_file)
        return output

    def job_fake_run(self):
        accuracy = random.randint(1, 100) / 100
        self.accuracies.append(accuracy)
        return TracePredictionOutput(metrics={"accuracy": accuracy})
