"""Define the Vector class"""
import numpy as np
from array_manager.core.native_formats.vector_components_dict import VectorComponentsDict
from array_manager.core.native_formats.vector import Vector
from array_manager.core.native_formats.matrix import Matrix


class BlockMatrix(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a variable represents the view of that variable which is generated from the self.vals attribute of the Vector object.

    Attributes
    ----------
    vals : Vector
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
            self.shape = shape = (len(blocks_list), len(blocks_list[0]))

            self.sub_matrices = sub_matrices = {
                (i, j) : blocks_list[i][j]
                for i in range(shape[0])
                for j in range(shape[1])
            }            
        else:
            self.shape = shape
            self.sub_matrices = sub_matrices = blocks
        
        
        row_sizes = np.zeros(shape[0], dtype=int)
        col_sizes = np.zeros(shape[1], dtype=int)
        row_start_indices = np.zeros(shape, dtype=int)
        row_end_indices = np.zeros(shape, dtype=int)
        col_start_indices = np.zeros(shape, dtype=int)
        col_end_indices = np.zeros(shape, dtype=int)
        self.num_nonzeros = 0
        
        # # Find a nonzero block
        # for i in range(shape[0]):
        #     for j in range(shape[1]):
        #         # When a block is defined zero
        #         if type(sub_matrices[i,j]) ==int:
        #             if sub_matrices[i,j] == 0:
        #                 continue

        # Find row and column sizes
        for i in range(shape[0]):
            for j in range(shape[1]):
               # When a block is defined zero
                if type(sub_matrices[i,j]) == int:
                    if sub_matrices[i,j] == 0:
                        continue
                    else:
                        raise ValueError('0 is the only scalar value that can be assigned to any block in the declaration')

                if type(sub_matrices[i, j]) not in (BlockMatrix, Matrix):
                    raise TypeError('Blocks inside the block matrix should be of type Matrix() or BlockMatrix(). Declared block {} is of type {}'.format(sub_matrices[i, j], type(sub_matrices[i, j])))
                
                if row_sizes[i] == 0:
                    row_sizes[i] = sub_matrices[i, j].dense_shape[0]
                elif row_sizes[i] != sub_matrices[i, j].dense_shape[0]:
                    raise ValueError('Given shapes for blocks inside the block matrix are incompatible')
                
                if col_sizes[j] == 0:
                    col_sizes[j] = sub_matrices[i, j].dense_shape[1]
                elif col_sizes[j] != sub_matrices[i, j].dense_shape[1]:
                    raise ValueError('Given shapes for blocks inside the block matrix are incompatible')
                
                

        for i in range(shape[0]):
            for j in range(shape[1]):
                # When a block is defined zero
                if type(sub_matrices[i,j]) == int:
                    if sub_matrices[i,j] == 0:
                        continue

                # Compute start and end indices of each block
                if i >= 1:
                    row_start_indices[i, j] = row_start_indices[i-1, j] + row_sizes[i-1]
                    row_end_indices[i-1, j] = row_start_indices[i, j]
                if j >= 1:
                    col_start_indices[i, j] = row_start_indices[i, j-1] + col_sizes[j-1]
                    col_end_indices[i, j-1] = col_start_indices[i, j]

                self.num_nonzeros += sub_matrices[i, j].num_nonzeros

        row_end_indices[i, j] = row_start_indices[i, j] + row_sizes[i]
        col_end_indices[i, j] = col_start_indices[i, j] + col_sizes[j]
        self.dense_shape = (row_end_indices[i, j], col_end_indices[i, j])
        self.dense_size = np.prod(self.dense_shape)
        self.density = float(self.num_nonzeros / self.dense_size)
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

        self.vals = Vector(vector_components_dict)

    def allocate(self, copy=False, data=None):
        # Line 124 is executed only at the top level when we say copy is not needed. Because for lower levels in the hierarchy, data comes from higher levels and data is never None.

        # if data is None and not copy:
        #     data = np.zeros(self.num_nonzeros)

        # New addition
        shape = self.shape
        if data is not None and not copy: 
            pass
        else:
            data = np.zeros(self.num_nonzeros)

        self.vals.allocate(data=data, setup_views=True)

        ind1 = 0
        ind2 = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                sub_matrix = self.sub_matrices[i, j]

                ind2 += sub_matrix.num_nonzeros
                sub_matrix.allocate(data=data[ind1:ind2], copy=copy)
                ind1 += sub_matrix.num_nonzeros
        
        # To test if allocate() works with and without copy=True, run all_in_one.py after commenting out self.update_bottom_up() here. This will give correct results when copy=False and incorrect results when copy=True (only the Matrix objects will contain nonzero values, all BlockMatrix objects' data will be populated with zeros)
        if copy:
            self.update_bottom_up()

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