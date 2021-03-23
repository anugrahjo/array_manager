"""Define the CSRMatrix class"""
import numpy as np
from sparse_matrix import SparseMatrix

class CSRMatrix(SparseMatrix):
    """
    Class that generates the standard csr sparse matrix from given matrix in native format.

    Attributes
    ----------
    native : Matrix or BlockMatrix
        Matrix in native format that generates the standard CSRMatrix format
    data : np.ndarray
        Vector containing nonzeros of the sparse matrix sorted first by row index and then by column index.
    ind_ptr : np.ndarray
        Vector whose first entry is zero and the n-th entry stores the number of nonzeros up to (n-1)th row starting from the first row.
    cols : np.ndarray
        Vector containing col indices (sorted in increasing order along each row) of nonzeros of a sparse matrix.

    """

    def __init__(self, native_matrix):
        """
        Initialize the CSRMatrix object by initializing a SparseMatrix object and then compute the sorting indices (bottom-up and top-down) for conversions between self and its native.

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in native format which needs to converted to the standard CSRMatrix format
        """
        super.__init__(native_matrix)

        # sparse_format = 'csr'
        sorting_indices_cols = np.argsort(native_matrix.cols)
        new_row_indices = native_matrix.rows[sorting_indices_cols]
        sorting_indices_new_rows = np.argsort(new_row_indices)

        # precomputed fwd permutation matrix
        self.bottom_up_sorting_indices = sorting_indices_cols[sorting_indices_new_rows]
        
        #optimizer requested format
        self.cols = native_matrix.cols[self.bottom_up_sorting_indices]
        final_rows = native_matrix.rows[self.bottom_up_sorting_indices]
        self.ind_ptr = np.insert(np.bincount(final_rows).cumsum(), 0, 0)

        # precomputed reverse permutation matrix
        self.top_down_sorting_indices = np.argsort(self.bottom_up_sorting_indices)
