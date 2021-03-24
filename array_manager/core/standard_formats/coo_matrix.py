"""Define the COOMatrix class"""
import numpy as np
from sparse_matrix import SparseMatrix


class COOMatrix(SparseMatrix):
    # Do we need to add attributes of SparseMatrix in dicstring below?
    """
    Class that generates the standard coo sparse matrix from given matrix in the native format.

    Attributes
    ----------
    native : Matrix or BlockMatrix
        Matrix in the native format that generates the standard COOMatrix object
    data : np.ndarray
        Vector containing nonzeros of the sparse matrix sorted first by row index and then by column index.
    num_nonzeros : int
        Number of nonzeros in the sparse matrix
    rows : np.ndarray
        Vector containing sorted row indices (in increasing order) of nonzeros of a sparse matrix.
    cols : np.ndarray
        Vector containing col indices (sorted in increasing order along each row) of nonzeros of the sparse matrix.
    """

    def __init__(self, native_matrix):
        """
        Initialize the COOMatrix object by initializing a SparseMatrix object and then compute the sorting indices (bottom-up and top-down) for conversions between self and its native.

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in the native format which needs to converted to the standard COOMatrix format
        """
        super.__init__(native_matrix)

        # sparse_format == 'coo':
        sorting_indices_cols = np.argsort(native_matrix.cols)
        new_row_indices = native_matrix.rows[sorting_indices_cols]
        sorting_indices_new_rows = np.argsort(new_row_indices)

        # precomputed fwd permutation matrix
        self.bottom_up_sorting_indices = sorting_indices_cols[sorting_indices_new_rows]
        
        # requested format
        self.rows = native_matrix.rows[self.bottom_up_sorting_indices]
        self.cols = native_matrix.cols[self.bottom_up_sorting_indices]
        
        # precomputed reverse permutation matrix
        self.top_down_sorting_indices = np.argsort(self.bottom_up_sorting_indices)

