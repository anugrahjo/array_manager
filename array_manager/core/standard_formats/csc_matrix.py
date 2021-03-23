"""Define the Vector class"""
import numpy as np
from sparse_matrix import SparseMatrix

class CSCMatrix(SparseMatrix):
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

        # if sparse_format = 'csc':
        sorting_indices_rows= np.argsort(native_matrix.rows)
        new_col_indices = native_matrix.cols[sorting_indices_rows]
        sorting_indices_new_cols = np.argsort(new_col_indices)

        # precomputed fwd permutation matrix
        self.combined_sorting_indices = sorting_indices_rows[sorting_indices_new_cols]
        
        #optimizer requested format
        self.rows = native_matrix.rows[self.combined_sorting_indices]
        final_cols = native_matrix.cols[self.combined_sorting_indices]
        self.ind_ptr = np.insert(np.bincount(final_cols).cumsum(), 0, 0)

        # precomputed reverse permutation matrix
        self.inverse_combined_sorting_indices = np.argsort(combined_sorting_indices)
