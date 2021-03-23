"""Define the Vector class"""
import numpy as np


class Vector(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a subvector represents the view of that subvector which is generated from the self.data attribute of the Vector object.

    Attributes
    ----------
    data : np.ndarray
        Concatenated vector from the dixtionary of subvectors 

    """

    def __init__(self, vector_components_dict):
        self.vector_components_dict = vector_components_dict

    def allocate(self, data=None, setup_views=False):
        if data:
            self.data = data
        else:
            self.data = data = np.zeros(self.vector_components_dict.vector_size)

        dict_ = {}
        for key, component_dict in self.vector_components_dict.items():
            shape = component_dict['shape']
            ind1 = component_dict['start_index']
            ind2 = component_dict['end_index']
            dict_[name] = data[ind1:ind2].reshape(shape)

        return dict_



    def __init__(self, vector_components_dict, setup_views=False):
        """
        Initialize the Vector object by allocating a zero vector of desired size.

        Parameters
        ----------
        variables_list : VariablesList
            List of variables that are concatenated
        """

        self.data = np.zeros(vector_components_dict.vector_size)

        if setup_views:
            self.dict_ = self.setup_views(vector_components_dict)

        else:
            self.dict_ = None

    def setup_views(self, vector_components_dict):
        """
        Setup views for variables that are concatenated into a single vector.

        Parameters
        ----------
        variables_set : VariablesSet
            List of variables that are concatenated
        """
        dict_ = {}
        for key, component_dict in vector_components_dict.items():
            shape = component_dict['shape']
            ind1 = component_dict['start_index']
            ind2 = component_dict['end_index']
            dict_[name] = self.data[ind1:ind2].reshape(shape)

        return dict_

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, value):
        self.dict_[key][:] = value

    def __len__(self):
        return len(self.data)

    def __iadd__(self, other):
        self.data += other.data

    def __isub__(self, other):
        self.data -= other.data

        # vec == other
        # vec.set(other)
        # vec *= alpha
        # vec += other
        # ...
        # vec.set_const(0.)
