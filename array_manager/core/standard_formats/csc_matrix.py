"""Define the CSCMatrix class"""
import numpy as np
from array_manager.core.standard_formats.sparse_matrix import SparseMatrix
import scipy.sparse as sp


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

    def __init__(self, native_matrix, duplicate_indices=False):
        """
        Initialize the CSCMatrix object by initializing a SparseMatrix object and then compute the sorting indices (bottom-up and top-down) for conversions between self and its native.

        Parameters
        ----------
        native_matrix : Matrix or BlockMatrix
            Matrix in the native format which needs to converted to the standard CSCMatrix format
        """
        super().__init__(native_matrix, duplicate_indices=duplicate_indices)
        
        if self.duplicate_indices:
            cols_rows = np.append([native_matrix.cols], [native_matrix.rows], axis=0).T
            unique_sorted_cols_rows, indices, inverse_duplicate_indices = np.unique(cols_rows, return_index = True, return_inverse = True, axis = 0)

            # requested format
            self.rows = unique_sorted_cols_rows[:, 1]
            final_cols = unique_sorted_cols_rows[:, 0]
            self.ind_ptr = np.insert(np.bincount(final_cols).cumsum(), 0, 0)


            self.inverse_duplicate_indices = inverse_duplicate_indices
            self.data = np.bincount(self.inverse_duplicate_indices, weights=self.native.vals.data)
        else:
            # precomputed fwd permutation matrix, sparse_format = 'csc':
            self.bottom_up_sorting_indices = np.lexsort((native_matrix.rows, native_matrix.cols))
            
            #optimizer requested format
            self.rows = native_matrix.rows[self.bottom_up_sorting_indices]
            final_cols = native_matrix.cols[self.bottom_up_sorting_indices]
            self.ind_ptr = np.insert(np.bincount(final_cols).cumsum(), 0, 0)

            # precomputed reverse permutation matrix
            self.top_down_sorting_indices = np.argsort(self.bottom_up_sorting_indices)

            # Initialize with the data given in the native_format
            self.data = self.native.vals.data[self.bottom_up_sorting_indices]

    def scipy_sparse_array(self):
        return sp.csc_matrix((self.data, self.rows, self.indptr), shape=self.dense_shape)