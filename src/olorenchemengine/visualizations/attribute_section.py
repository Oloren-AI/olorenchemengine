""" Module containing classes describing available attributes for visualizations.

These attributes are used to standardize the parameters of visualizations to make
them more accessible (to the extent possible)."""

import os
from abc import ABC, abstractmethod

from olorenchemengine.base_class import *
from olorenchemengine.representations import *
from olorenchemengine.dataset import *
from olorenchemengine.uncertainty import *
from olorenchemengine.interpret import *
from olorenchemengine.internal import *

class BaseAttribute(ABC):
    """Base class for all attributes for a visualization

    Parameters:
        name (str): The name of the attribute
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def to_json(self) -> list:
        """
        Convert the attribute to a json object

        Parameters:
            name (str): The name of the attribute

        Returns:
            The json representation of the object
        """
        pass


class SimpleAttribute(BaseAttribute):
    """A simple attribute for a visualization is
    and attribute whose only parameter is a name.

    This describes all attributes except for the InputThreshold class,
    which requires a minimum and maximum value alongside a name."""

    def to_json(self) -> list:
        return [self.name, self.attribute_name]

class ModelSelector(SimpleAttribute):
    """Class to create a model selector attribute

    Parameters:
        name (str): The name of the attribute
    """
    attribute_name = "modelSelector"


class DatasetSelector(SimpleAttribute):
    """Class to create a dataset selector attribute

    Parameters:
        name (str): The name of the attribute
    """
    attribute_name = "datasetSelector"

class InputNumber(SimpleAttribute):
    """Class to create an input number attribute

    Parameters:
        name (str): The name of the attribute
    """
    attribute_name = "inputNumber"

class InputString(SimpleAttribute):
    """Class to create an input string attribute

    Parameters:
        name (str): The name of the attribute
    """
    attribute_name = "inputString"

class ColorPicker(SimpleAttribute):
    """Class to create a color picker attribute

    Parameters:
        name (str): The name of the attribute
    """
    attribute_name = "colorPicker"

class InputThreshold(BaseAttribute):
    """Class to create an input threshold attribute

    Parameters:
        name (str): The name of the attribute
        min (int): The minimum value of the threshold
        max (int): The maximum value of the threshold
    """
    @log_arguments
    def __init__(self, name:  str, min: int = 0, max: int = 100,  ):
        super().__init__(name)
        self.min = min
        self.max = max


    def to_json(self):
        inputThreshold = [self.name, "threshold", "min: " + str(self.min), "max: " + str(self.max)]
        return inputThreshold


class AttributeSection(BaseAttribute):
    """Class to create an attribute section

    Parameters:
        attribute_name (str): The name of the attribute section
        attributes (list): The list of attributes of type BaseAttribute
    """
    @log_arguments
    def __init__(self, attribute_name:  str, attributes: List[BaseAttribute],  ):
        super().__init__(attribute_name)
        self.attributes = attributes


    def to_json(self):
        attribute_section = {self.name: [attribute.to_json() for attribute in self.attributes]}

        return attribute_section










