"""Define the Vector class"""
import numpy as np
from sparse_matrix import SparseMatrix

class CSRMatrix(SparseMatrix):
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
        super.__init__(native_matrix)

        # if sparse_format = 'csr':
        sorting_indices_cols = np.argsort(native_matrix.cols)
        new_row_indices = native_matrix.rows[sorting_indices_cols]
        sorting_indices_new_rows = np.argsort(new_row_indices)

        # precomputed fwd permutation matrix
        self.combined_sorting_indices = sorting_indices_cols[sorting_indices_new_rows]
        
        #optimizer requested format
        self.cols = native_matrix.cols[self.combined_sorting_indices]
        final_rows = native_matrix.rows[self.combined_sorting_indices]
        self.ind_ptr = np.insert(np.bincount(final_rows).cumsum(), 0, 0)

        # precomputed reverse permutation matrix
        self.inverse_combined_sorting_indices = np.argsort(combined_sorting_indices)
