"""Define the Vector class"""
import numpy as np


class DenseMatrix(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a variable represents the view of that variable which is generated from the self.data attribute of the Vector object.

    Attributes
    ----------
    native : Matrix or BlockMatrix
        Native format that generates the standard DenseMatrix format
    data : np.ndarray
        Concatenated vector of a list of variables 

    """

    def __init__(self, native_matrix):
        """
        Initialize the DenseMAtrix object by allocating a zero vector of desired size.

        Parameters
        ----------
        native_matrix : list or dict
            List of variables that are concatenated
        """
        self.native = native_matrix
        self.data = np.zeros(native_matrix.dense_shape)

        # (Can also support col major or row major if optimizer requests so. This implementation uses default python ordering which is row major. Col major would be faster with Fortran-based optimizers)
        self.non_zero_flattened_indices = np.ravel_multi_index((native_matrix.rows, native_matrix.cols), native_matrix.dense_shape)

    def update_bottom_up(self):
        self.native.update()
        # Replaces specified elements of an array with given values. The indexing works on the flattened target array.
        np.put(self.data, self.non_zero_flattened_indices, self.native.vals.data)

    def update_top_down(self):
        self.native.vals.data = self.data[self.native.rows, self.native.cols]
        self.native.update_bottom_up()


