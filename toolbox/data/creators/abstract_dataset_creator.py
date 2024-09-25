from abc import ABC, abstractmethod
from typing import Generic, Optional, Type, TypeVar

from toolbox.data.processing.cleaning.data_cleaner import DataCleaner
from toolbox.data.tdatasets.idataset import iDataset
from toolbox.infra.base_object import BaseObject
from toolbox.util.override import overrides

DatasetType = TypeVar("DatasetType", bound=iDataset)


class AbstractDatasetCreator(BaseObject, ABC, Generic[DatasetType]):

    def __init__(self, data_cleaner: Optional[DataCleaner] = None):
        """
        Responsible for creating data in format for defined models.
        :param data_cleaner: the data cleaner to use on the data
        """
        self.data_cleaner = DataCleaner([]) if data_cleaner is None else data_cleaner

    @classmethod
    @overrides(BaseObject)
    def _get_enum_class(cls, child_class_name: str) -> Type:
        """
        Returns the correct enum class mapping name to class given the abstract parent class type and name of child class
        :param child_class_name: the name of the child class
        :return: the enum class mapping name to class
        """
        from toolbox.data.creators.supported_dataset_creator import SupportedDatasetCreator
        return SupportedDatasetCreator

    @abstractmethod
    def create(self) -> DatasetType:
        """
        Creates the data
        :return: the data
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        :return: Returns the name of the dataset.
        """

    def from_yaml(self) -> DatasetType:
        """
        Creates a dataset from the yaml representation (dataset creator)
        :param self: The yaml representation (dataset creator)
        :return: The dataset created
        """
        return self.create()
