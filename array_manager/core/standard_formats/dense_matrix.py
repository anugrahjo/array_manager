"""Define the DenseMatrix class"""
import numpy as np


class DenseMatrix(object):
    """
    Class that generates the standard dense matrix from given matrix in native format.

    Attributes
    ----------
    native : Matrix or BlockMatrix
        Matrix in native format that generates the standard DenseMatrix format
    data : np.ndarray
        Concatenated vector of a list of variables 

    """

    def __init__(self, native_matrix):
        """
        Initialize the DenseMatrix object by allocating a zero matrix of desired size.

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in native format which needs to converted to standard DenseMatrix format
        """
        self.native = native_matrix
        self.data = np.zeros(native_matrix.dense_shape)

        # (Can also support col major or row major if optimizer requests so. This implementation uses default python ordering which is row major. Col major would be faster with Fortran-based optimizers)
        self.flattened_indices_of_non_zeros = np.ravel_multi_index((native_matrix.rows, native_matrix.cols), native_matrix.dense_shape)

    def update_bottom_up(self):
        """
        Request the native to update its data and then update self.data.
        """
        self.native.update_bottom_up()
        # Replaces specified elements of an array with given values. The indexing works on the flattened target array.
        np.put(self.data, self.flattened_indices_of_non_zeros, self.native.vals.data)

    def update_top_down(self):
        """
        Update the data in the native from self.data and request native to update its submatrices/children if there are any.
        """
        self.native.vals.data = self.data[self.native.rows, self.native.cols]
        self.native.update_top_down()


