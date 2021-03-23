"""Define the VariablesList class"""
import numpy as np
from typing import dict


class VectorComponentsDict(dict):
    """
    Dictionary of dictionaries in which each dictionary represents the data corresponding to a subvector.
    For example, this can be a dictionary of variables in which each variable corresponds to a specific class of variables such as design variables, state_variables, constraint_list, dual variables, etc.
    Each dictionary in this dictionary (which stores the data pertaining to a subvector) contains values corresponding to six keys, namely,
        1. shape : shape of the subvector
        2. lower : lower bounds on the subvector (optional)
        3. upper : upper bounds on the subvector (optional)
        # 4. equals : for constraints
        4. start_index : starting index of the subvector in the concatenated vector of subvectors from the same class
        5. end_index : ending index of the variable in the concatenated vector of subvectors from the same class

    """

    def __init__(self):
        self.vector_size = 0
        super().__init__()

    def __setitem__(self, key, component_dict: dict):
        """
        Append a new variable dictionary to the current list of variable dictionaries.

        Parameters
        ----------
        var_dict : dict
            Variable dictionary to be appended to self.
        """
        if key in self:
            raise KeyError('A vector called {} has already been added'.format(key))

        size = np.prod(component_dict['shape'])

        # component_dict['size'] = component_dict['size']
        component_dict['start_index'] = self.vector_size
        self.vector_size += size
        component_dict['end_index'] = self.vector_size

        super()[key] = component_dict