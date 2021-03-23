"""Define the Vector class"""
import numpy as np


class Matrix(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a variable represents the view of that variable which is generated from the self.vals attribute of the Vector object.

    Attributes
    ----------
    vals : np.ndarray
        Concatenated vector of a list of variables 

    """

    def __init__(self, matrix_components_dict, setup_views=False):
        """
        Initialize the Matrix object by allocating a zero vector of desired size.

        Parameters
        ----------
        variables_list : VariablesList
            List of variables that are concatenated
        """

        self.dense_shape = matrix_components_dict.dense_shape
        self.dense_size = matrix_components_dict.dense_size
        self.num_nonzeros = matrix_components_dict.num_nonzeros
        self.density = self.num_nonzeros / self.dense_size
        self.rows = np.zeros(self.num_nonzeros, dtype=int)
        self.cols = np.zeros(self.num_nonzeros, dtype=int)

        vector_components_dict = VectorComponentsDict()

        for key, component_dict in matrix_components_dict.items():
            shape = component_dict['shape']
            # COO arrays from given CSR or CSC arrays
            if component_dict['ind_ptr']:
                ind_ptr = component_dict['ind_ptr']
                # Compute differences between consecutive elements in the ind_ptr array
                num_repeats = np.ediff1d(ind_ptr)
                if component_dict['cols'] == None:
                    component_dict['cols'] = np.repeat(np.arange(shape[1]), num_repeats)

                elif component_dict['rows'] == None:
                    component_dict['rows'] =  np.repeat(np.arange(shape[0]), num_repeats)

                # need this?
                # del matrix_dict['ind_ptr']

            # Give exception if only one of rows or cols is None at this point
            # Dense matrix component
            elif !(component_dict['rows']) and !(component_dict['cols']):
                component_dict['rows'] = np.repeat(np.arange(shape[0]), shape[1])
                component_dict['cols'] = np.tile(np.arange(shape[1]), shape[0])

            global_rows = component_dict['rows'] + component_dict['row_start_index']
            global_cols = component_dict['cols'] + component_dict['col_start_index']

            start = component_dict['start_index']
            end = component_dict['end_index']
            
            self.rows[start:end] = global_rows
            self.cols[start:end] = global_cols

            # need this?
            # del matrix_dict['rows']
            # del matrix_dict['cols']
            
            vals_shape = component_dict['vals_shape']
            vector_components_dict[key] = dict(shape=vals_shape)

        self.vals = Vector(vector_components_dict, setup_views=setup_views)

        # new addition
        for key, component_dict in matrix_components_dict.items():
            vals = self.component_dict['vals']
            if vals:
                self[key] = vals

    def __getitem__(self, key):
        # return self.vals_vector.dict_[key]
        # return self.vals_vector[key]
        return self.vals[key]

    def __setitem__(self, key, value):
        # self.vals_vector.dict_[key][:] = value
        # self.vals_vector[key] = value
        self.vals[key] = value

    def allocate(self, data=None):
        if data is None:
            data = np.zeros(self.num_nonzeros)

        self.vals.allocate(data=data, setup_views=True)

        ind1 = 0
        ind2 = 0
        for i, j in self.sub_matrices:
            sub_matrix = self.sub_matrices[i, j]

            ind2 += sub_matrix.num_nonzeros
            sub_matrix.allocate(data[ind1:ind2])
            ind1 += sub_matrix.num_nonzeros

    def update_bottom_up(self):
        pass

    def update_top_down(self):
        pass

    # len() returns the number of nonzeros in the matrix
    def __len__(self):
        return len(self.vals)

    def __iadd__(self, other):
        self.vals += other.vals

    def __isub__(self, other):
        self.vals -= other.vals
