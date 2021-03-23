"""Define the Vector class"""
import numpy as np


class SparseMatrix(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a variable represents the view of that variable which is generated from the self.data attribute of the Vector object.

    Attributes
    ----------
    data : np.ndarray
        Concatenated vector of a list of variables 

    """

    def __init__(self, native_matrix):
        """
        Initialize the Vector object by allocating a zero vector of desired size.

        Parameters
        ----------
        matrices : list or dict
            List of variables that are concatenated
        """
        self.native = native_matrix
        self.num_nonzeros = native_matrix.num_nonzeros
        self.data = np.zeros(self.num_nonzeros)
            
    def update_bottom_up(self):
        self.native.update_bottom_up()
        self.data = self.native.vals.data[self.combined_sorting_indices]

    def update_top_down(self):
        self.native.vals.data = self.data[self.inverse_combined_sorting_indices]
        self.native.update_bottom_up()

    
