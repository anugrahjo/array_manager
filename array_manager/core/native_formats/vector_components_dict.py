"""Define the VectorComponentsDict class"""
import numpy as np
from typing import Dict


class VectorComponentsDict(dict):
    """
    Dictionary of dictionaries representing a vector composed of multiple subvectors. Each dictionary within this dictionary represents the data corresponding to a subvector.
    For example, this can be a dictionary of variables in which each variable is a design variable in an optimization problem.
    Each dictionary in this dictionary (which stores the data pertaining to a subvector) contains values corresponding to six keys, namely,
    - shape : shape of the subvector
    - start_index : starting index of the subvector in the concatenated contiguous vector containing all subvectors from the same class
    - end_index : ending index of the subvector in the concatenated contiguous vector containing all subvectors from the same class

    Attributes
    ----------
    vector_size : int
        Size of the vector that contains all the subvectors
    """

    def __init__(self):
        """
        Initialize a dictionary object with a default value for the vector_size attribute. 
        """
        self.vector_size = 0
        super().__init__()

    def __setitem__(self, key, component_dict: Dict):
        """
        Add/replace a dictionary corresponding to a subvector in the current dictionary of subvector dictionaries.

        Parameters
        ----------
        component_dict : dict
            Subvector dictionary to be added/replaced in self.
        """
        if key in self:
            raise KeyError('A subvector called {} has already been added'.format(key))

        size = np.prod(component_dict['shape'])

        component_dict['size'] = size
        component_dict['start_index'] = self.vector_size
        self.vector_size += size
        component_dict['end_index'] = self.vector_size

        super().__setitem__(key, component_dict)