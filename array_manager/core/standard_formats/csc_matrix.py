"""Define the CSCMatrix class"""
import numpy as np
from array_manager.core.standard_formats.sparse_matrix import SparseMatrix


class CSCMatrix(SparseMatrix):
    """
    Class that generates the standard csc sparse matrix from given matrix in the native format.

    Attributes
    ----------
    native : Matrix or BlockMatrix
        Matrix in the native format that generates the standard CSCMatrix object
    data : np.ndarray
        Vector containing nonzeros of the sparse matrix sorted first by column index and then by row index.
    num_nonzeros : int
        Number of nonzeros in the sparse matrix
    rows : np.ndarray
        Vector containing sorted row indices (sorted in increasing order along each column) of nonzeros of the sparse matrix.
    ind_ptr : np.ndarray
        Vector whose first entry is zero and nth entry stores the number of nonzeros up to (n-1)th column starting from the first column.
    """

    def __init__(self, native_matrix):
        """
        Initialize the CSCMatrix object by initializing a SparseMatrix object and then compute the sorting indices (bottom-up and top-down) for conversions between self and its native.

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in the native format which needs to converted to the standard CSCMatrix format
        """
        super().__init__(native_matrix)

        # sparse_format = 'csc':
        sorting_indices_rows= np.argsort(native_matrix.rows)
        new_col_indices = native_matrix.cols[sorting_indices_rows]
        sorting_indices_new_cols = np.argsort(new_col_indices)

        # precomputed fwd permutation matrix
        self.bottom_up_sorting_indices = sorting_indices_rows[sorting_indices_new_cols]
        
        #optimizer requested format
        self.rows = native_matrix.rows[self.bottom_up_sorting_indices]
        final_cols = native_matrix.cols[self.bottom_up_sorting_indices]
        self.ind_ptr = np.insert(np.bincount(final_cols).cumsum(), 0, 0)

        # precomputed reverse permutation matrix
        self.top_down_sorting_indices = np.argsort(self.bottom_up_sorting_indices)

        # Initialize with the data given in the native_format
        self.data = self.native.vals.data[self.bottom_up_sorting_indices]