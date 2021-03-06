"""Define the SparseMatrix class"""
import numpy as np


class SparseMatrix(object):
    """
    Container class for all classes that generate standard sparse matrix formats (coo, csc, and csr) from given matrix in native format.

    Attributes
    ----------
    native : Matrix or BlockMatrix
        Matrix in the native format that needs to be converted into any of the standard SparseMatrix formats
    data : np.ndarray
        Vector containing nonzeros of the sparse matrix
    num_nonzeros : int
        Number of nonzeros in the sparse matrix
    """

    def __init__(self, native_matrix, duplicate_indices=False):
        """
        Initialize the SparseMatrix object by allocating a zero vector of desired size (number of nonzeros in the sparse matrix).

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in the native format which needs to converted to any of the standard SparseMatrix formats
        """
        self.duplicate_indices = duplicate_indices
        self.native = native_matrix
        self.dense_shape = native_matrix.dense_shape
        # Need this (num_nonzeros)?
        self.num_nonzeros = native_matrix.num_nonzeros
        self.data = np.zeros(self.num_nonzeros)
            
    def update_bottom_up(self):
        """
        Request the native to update its data and then update self.data.
        """
        self.native.update_bottom_up()
        if self.duplicate_indices:
            self.data = np.bincount(self.inverse_duplicate_indices, weights=self.native.vals.data)

        else:
            self.data = self.native.vals.data[self.bottom_up_sorting_indices]

    def update_top_down(self):
        """
        Update the data in the native from self.data and request native to update its submatrices/children if there are any.
        """
        if not(self.duplicate_indices):
            self.native.vals.data = self.data[self.top_down_sorting_indices]
            self.native.update_top_down()
        else:
            raise Exception('Arrays with duplicate indices cannot be updated from top to bottom')

    
