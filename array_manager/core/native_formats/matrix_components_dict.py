"""Define the MatrixComponentsDict class"""
import numpy as np
from typing import Dict, Tuple
from array_manager.core.native_formats.vector import Vector



class MatrixComponentsDict(dict):
    """
    Dictionary of dictionaries representing a matrix composed of multiple submatrices. Each dictionary within this dictionary represents data corresponding to a submatrix such as values, row indices, column indices, index pointers, etc.
    For example, this can be a dictionary of Jacobians in which each Jacobian corresponds to the partial derivative of some function with respect to the design variables in an optimization problem.
    Each dictionary in this dictionary (which stores the data pertaining to a subvector) contains values corresponding to six keys, namely,
        1. shape : shape of the subvector
        5. start_index : starting index of the subvector in the concatenated contiguous vector containing all subvectors from the same class
        6. end_index : ending index of the subvector in the concatenated contiguous vector containing all subvectors from the same class

    Attributes
    ----------
    vector_components_dict1 : VectorComponentsDict
        Size of the vector that contains all the subvectors
    vector_components_dict2 : VectorComponentsDict
        Size of the vector that contains all the subvectors
    num_nonzeros : int
        Size of the vector that contains all the subvectors
    dense_shape : tuple
        Size of the vector that contains all the subvectors
    dense_size : int
        Size of the vector that contains all the subvectors
    """
    def __init__(self, vector_components_dict1, vector_components_dict2):
        """
        Initialize a dictionary object with a default value for the vector_size attribute. 
        """
        # Note: In the case of partials, 1 represents output and 2 represents input respectively.
        self.vector_components_dict1 = vector_components_dict1
        self.vector_components_dict2 = vector_components_dict2
        self.num_nonzeros = 0
        self.dense_shape = (self.vector_components_dict1.vector_size, self.vector_components_dict2.vector_size)
        self.dense_size = np.prod(self.dense_shape)
        super().__init__()


    def __setitem__(self, key : Tuple, component_dict: Dict):
        """
        Add/replace a dictionary corresponding to a submatrix in the current list of submatrix dictionaries.

        Parameters
        ----------
        component_dict : dict
            Submatrix dictionary to be added/replaced in self.
        """
        if len(key) != 2:
            raise KeyError('The submatrix key provided {} has only one subvector reference; it should be referenced by two subvector names'.format(key))

        if type(key[0]) != str or type(key[1]) != str:
            raise KeyError('The key referencing a submatrix must be tuple of strings, the key provided is {}'.format(key))

        name1 = key[0]
        name2 = key[1]

        if key in self:
            raise KeyError('A submatrix called {} has already been added'.format(key))

        row_start_index = self.vector_components_dict1[name1]['start_index']
        col_start_index = self.vector_components_dict2[name2]['start_index']
        row_end_index = self.vector_components_dict1[name1]['end_index']
        col_end_index = self.vector_components_dict2[name2]['end_index']

        shape1 = self.vector_components_dict1[name1]['shape']
        shape2 = self.vector_components_dict2[name2]['shape']

        component_dict['row_start_index'] = row_start_index
        component_dict['col_start_index'] = col_start_index
        component_dict['row_end_index'] = row_end_index
        component_dict['col_end_index'] = col_end_index

        size1 = row_end_index - row_start_index
        size2 = col_end_index - col_start_index

        shape = (size1, size2)
        component_dict['shape'] = shape

        if 'rows' in component_dict: 
            rows = component_dict['rows']
        else:
            rows = component_dict['rows'] = None

        if 'cols' in component_dict: 
            cols = component_dict['cols']
        else:
            cols = component_dict['cols'] = None

        if 'vals' in component_dict: 
            vals = component_dict['vals']
        else:
            vals = component_dict['vals'] = None
        
        if 'ind_ptr' in component_dict: 
            ind_ptr = component_dict['ind_ptr']
        else:
            ind_ptr = component_dict['ind_ptr'] = None
        
        # vals_shape is used for setting up views
        if 'vals_shape' in component_dict: 
            vals_shape = component_dict['vals_shape']
        else:
            vals_shape = component_dict['vals_shape'] = None
        
        allowed_shapes = (shape, shape1 + shape2)
        if size1 == 1:
            allowed_shapes = allowed_shapes + (shape2,)
        if size2 == 1:
            allowed_shapes = allowed_shapes + (shape1,)
        allowed_dtypes = (int, float)

        def check_shape(given_shape):
            if given_shape not in allowed_shapes:
                raise ValueError('Given shape {} of dense submatrix cannot be broadcast to match the subvector shapes ({}) and ({})'.format(vals.shape, shape1, shape2))

        def compare_shapes(shape1, shape2, name1, name2):
            if shape1 != shape2:
                raise ValueError('Shapes of {} {} and {} {} '.format(name1, shape1, name2, shape2))

        def check_dtype_vals(given_array):
            if given_array.dtype not in allowed_dtypes:
                raise TypeError('Given vals are not of type "int" or "float"')

        def check_int_array(given_array, name):
            if given_array.dtype != int:
                raise TypeError('Given {} array is not of type "int"'.format(name))

        # If component is dense
        if not(isinstance(rows, np.ndarray)) and not(isinstance(cols, np.ndarray)):
            if isinstance(vals, np.ndarray):
                check_shape(vals.shape)
                check_dtype_vals(vals)
                component_dict['vals_shape'] = vals.shape

            else:
                if isinstance(vals_shape, tuple):
                    check_shape(vals_shape)  
                    component_dict['vals_shape'] = vals_shape
                else:
                    component_dict['vals_shape'] = shape

        # If component is COO
        elif isinstance(rows, np.ndarray) and isinstance(cols, np.ndarray):
            compare_shapes(cols.shape, rows.shape, 'cols', 'rows')
            check_int_array(rows, 'rows')
            check_int_array(cols, 'cols')
            component_dict['vals_shape'] = rows.shape

            # Constant sparse COO matrix component
            if isinstance(vals, np.ndarray):
                compare_shapes(vals.shape, rows.shape, 'vals', 'rows')
                check_dtype_vals(vals)

        # If component is CSC
        elif isinstance(rows, np.ndarray):
            check_int_array(rows, 'rows')
            check_int_array(ind_ptr, 'ind_ptr')
            if ind_ptr.shape != (shape[1] + 1,):
                raise ValueError('Shape of the ind_ptr {} does not match the number of columns {} in the matrix component'.format(ind_ptr.shape, shape[1]))

            component_dict['vals_shape'] = rows.shape
            
            # Constant sparse CSC matrix component
            if isinstance(vals, np.ndarray):
                compare_shapes(vals.shape, rows.shape, 'vals', 'rows')
                check_dtype_vals(vals)

        # If component is CSR
        elif isinstance(cols, np.ndarray):
            check_int_array(cols, 'cols')
            check_int_array(ind_ptr, 'ind_ptr')
            if ind_ptr.shape != (shape[0] + 1,):
                raise ValueError('Shape of the ind_ptr {} does not match the number of rows {} of the component matrix'.format(ind_ptr.shape, shape[0]))
            
            component_dict['vals_shape'] = cols.shape

            # Constant sparse CSR matrix component
            if isinstance(vals, np.ndarray):
                compare_shapes(vals.shape, cols.shape, 'vals', 'cols')
                check_dtype_vals(vals)

        if isinstance(rows, np.ndarray):
            num_nonzeros = rows.size     
        elif isinstance(cols, np.ndarray):
            num_nonzeros = cols.size
        else:
            num_nonzeros = size1 * size2

        component_dict['start_index'] = self.num_nonzeros
        self.num_nonzeros += num_nonzeros     
        component_dict['end_index'] = self.num_nonzeros # - 1

        super().__setitem__((name1, name2), component_dict)






    
    

