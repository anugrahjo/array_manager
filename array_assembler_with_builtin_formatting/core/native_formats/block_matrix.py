"""Define the Vector class"""
import numpy as np


class BlockMatrix(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a variable represents the view of that variable which is generated from the self.vals attribute of the Vector object.

    Attributes
    ----------
    vals : np.ndarray
        Concatenated vector of a list of variables 

    """

    def __init__(self, blocks, shape=None, setup_views=False):
        """
        Initialize the Vector object by allocating a zero vector of desired size.

        Parameters
        ----------
        blocks : list or dict
            List of variables that are concatenated
        """
        if type(blocks) == list:
            list_initialize = True
        elif type(blocks) == dict:
            list_initialize = False
            if shape == None:
                raise Exception('Shape of the block matrix is needed when when initializing with a dictionary')
        else:
            raise TypeError('Blocks {} should be input as a list or dict'.format(blocks))
        
        if list_initialize:
            blocks_list = blocks
            for i in range(len(blocks_list)):
                if len(blocks_list[i]) != len(blocks_list[0]):
                    raise ValueError('Number of blocks in each row must be the same')
            shape = (len(blocks_list), len(blocks_list)[0])

            self.sub_matrices = sub_matrices = {
                i, j: blocks_list[i][j]
                for i in range(shape[0])
                for j in range(shape[1])
            }            
        else:
            self.sub_matrices = sub_matrices = dict(blocks)
        
               
        row_start_indices = np.zeros(shape, dtype=int)
        row_end_indices = np.zeros(shape, dtype=int)
        col_start_indices = np.zeros(shape, dtype=int)
        col_end_indices = np.zeros(shape, dtype=int)
        self.num_nonzeros = 0
        self.children = []

        for i in range(shape[0]):
            for j in range(shape[1]):
                if sub_matrices[i, j].dense_shape[0] != sub_matrices[i, 0].dense_shape[0] or sub_matrices[i, j].dense_shape[1] != sub_matrices[0, j].dense_shape[1]:
                    raise ValueError('Given shapes for blocks inside the block matrix are incompatible')
                # Compute start and end indices of each block
                if i >= 1:
                    row_start_indices[i, j] = row_start_indices[i-1, j] + sub_matrices[i-1, j].dense_shape[0]
                    row_end_indices[i-1, j] = row_start_indices[i, j]
                if j >= 1:
                    col_start_indices[i, j] = row_start_indices[i, j-1] + sub_matrices[i, j-1].dense_shape[1]
                    col_end_indices[i, j-1] = col_start_indices[i, j]

                self.num_nonzeros += sub_matrices[i, j].num_nonzeros
                
                else:
                    raise TypeError('Blocks inside the block matrix should be of type Matrix() or BlockMatrix(). Declared block {} is of type {}'.format(sub_matrices[i, j], type(sub_matrices[i, j])))

        row_end_indices[i, j] = row_start_indices[i, j] + sub_matrices[i, j].dense_shape[0]
        col_end_indices[i, j] = col_start_indices[i, j] + sub_matrices[i, j].dense_shape[1]
        self.dense_shape = (row_end_indices[i, j], col_end_indices[i, j])
        self.dense_size = np.prod(self.dense_shape)
        self.density = self.num_nonzeros / self.dense_size
        self.rows = np.zeros(self.num_nonzeros, dtype=int)
        self.cols = np.zeros(self.num_nonzeros, dtype=int)

        # block components dict is not really parallel with MatrixComponentsDict
        vector_components_dict = VectorComponentsDict()

        # This will result in row major ordering of the block matrices
        start_index = 0
        end_index = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                num_nonzeros = len(sub_matrices[i, j])
                end_index = start_index + num_nonzeros
                vector_components_dict[i, j] = dict(shape=(num_nonzeros,))
                
                global_rows = row_start_indices[i, j] + sub_matrices[i, j].rows
                global_cols = col_start_indices[i, j] + sub_matrices[i, j].cols

                self.rows[start_index:end_index] = global_rows
                self.cols[start_index:end_index] = global_cols

                start_index = end_index * 1

        self.vals = Vector(vector_components_dict, setup_views=setup_views)

        # How to compute the above (vals, rows, cols) without allocating/copying vals?

    def allocate(self, copy=False, data=None):
        if data is None and not copy:
            data = np.zeros(self.num_nonzeros)

        self.vals.allocate(data=data, setup_views=True)

        ind1 = 0
        ind2 = 0
        for i, j in self.sub_matrices:
            sub_matrix = self.sub_matrices[i, j]

            ind2 += sub_matrix.num_nonzeros
            sub_matrix.allocate(data=data[ind1:ind2], copy=copy)
            ind1 += sub_matrix.num_nonzeros

    def update_bottom_up(self):
        for i, j in self.sub_matrices:
            sub_matrix = self.sub_matrices[i, j]

            sub_matrix.update_bottom_up()

            self.vals[i, j] = sub_matrix.vals.data

    def update_top_down(self):
        for i, j in self.sub_matrices:
            sub_matrix = self.sub_matrices[i, j]

            sub_matrix.vals.data[:] = self.vals[i, j]

            sub_matrix.update_top_down()

    # len() returns the number of nonzeros in the blockmatrix
    def __len__(self):
        return len(self.vals)