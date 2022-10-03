"""Define the COOMatrix class"""
import numpy as np
from array_manager.core.standard_formats.sparse_matrix import SparseMatrix
import scipy.sparse as sp


class COOMatrix(SparseMatrix):
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

    def __init__(self, native_matrix, duplicate_indices=False):
        """
        Initialize the COOMatrix object by initializing a SparseMatrix object and then compute the sorting indices (bottom-up and top-down) for conversions between self and its native.

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in the native format which needs to converted to the standard COOMatrix format
        """
        super().__init__(native_matrix, duplicate_indices=duplicate_indices)
        
        if self.duplicate_indices:
            rows_cols = np.append([native_matrix.rows], [native_matrix.cols], axis=0).T
            unique_sorted_rows_cols, indices, inverse_duplicate_indices = np.unique(rows_cols, return_index = True, return_inverse = True, axis = 0)

            # requested format
            self.cols = unique_sorted_rows_cols[:, 1]
            self.rows = unique_sorted_rows_cols[:, 0]

            self.inverse_duplicate_indices = inverse_duplicate_indices
            self.data = np.bincount(self.inverse_duplicate_indices, weights=self.native.vals.data)
        else:
            # precomputed fwd permutation matrix, sparse_format == 'coo':
            self.bottom_up_sorting_indices = np.lexsort((native_matrix.cols, native_matrix.rows))
            
            # requested format
            self.rows = native_matrix.rows[self.bottom_up_sorting_indices]
            self.cols = native_matrix.cols[self.bottom_up_sorting_indices]
            
            # precomputed reverse permutation matrix
            self.top_down_sorting_indices = np.argsort(self.bottom_up_sorting_indices)

            # Initialize with the data given in the native_format
            self.data = self.native.vals.data[self.bottom_up_sorting_indices]

    def get_std_array(self):
        return sp.coo_matrix((self.data, (self.rows, self.cols)), shape=self.dense_shape)
