"""Define the VectorComponentsDict class"""
import numpy as np
from typing import dict


class VectorComponentsDict(dict):
    """
    Dictionary of dictionaries representing a vector composed of multiple subvectors. Each dictionary within this dictionary represents the data corresponding to a subvector.
    For example, this can be a dictionary of variables in which each variable corresponds to a specific class of variables such as the design variables in an optimization problem.
    Each dictionary in this dictionary (which stores the data pertaining to a subvector) contains values corresponding to six keys, namely,
        1. shape : shape of the subvector
        2. lower : lower bounds on the subvector (optional)
        3. upper : upper bounds on the subvector (optional)
        4. equals : for constraints (optional)
        5. start_index : starting index of the subvector in the concatenated contiguous vector containing all subvectors from the same class
        6. end_index : ending index of the subvector in the concatenated contiguous vector containing all subvectors from the same class

    Attributes
    ----------
    vector_size : int
        Size of the vector that contains all the subvectors
    """
    # Do we need lower, upper, equals above?

    def __init__(self):
        """
        Initialize a dictionary object with a default value for the vector_size attribute. 
        """
        self.vector_size = 0
        super().__init__()

    def __setitem__(self, key, component_dict: dict):
        """
        Add/replace a dictionary corresponding to a subvector in the current list of subvector dictionaries.

        Parameters
        ----------
        component_dict : dict
            Subvector dictionary to be added/replaced in self.
        """
        if key in self:
            raise KeyError('A vector called {} has already been added'.format(key))

        size = np.prod(component_dict['shape'])

        # component_dict['size'] = component_dict['size']
        component_dict['start_index'] = self.vector_size
        self.vector_size += size
        component_dict['end_index'] = self.vector_size

        super()[key] = component_dict