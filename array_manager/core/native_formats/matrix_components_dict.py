"""Define the PartialsList class"""
import numpy as np
from vector import Vector


# Deprecated (mostly?)
# 1. Safer to go with compute_pRpx, compute_pRpy, compute_pFpx, compute_pFpy 
#    - e.g., compute_pRpx(self, x, y, pRpx) where pRpx is DerivativesDict.values
#    - e.g., user code looks like pRpx['f', 'x'] = ... (clean, simple)


class MatrixComponentsDict(dict):

    def __init__(self, vector_components_dict1, vector_components_dict2):
        # Note: In the case of partials, 1 represents output and 2 represents input respectively.
        
        self.vector_components_dict1 = vector_components_dict1
        self.vector_components_dict2 = vector_components_dict2
        self.num_nonzeros = 0
        self.dense_shape = (self.vector_components_dict1.vector_size, self.vector_components_dict2.vector_size)
        self.dense_size = np.prod(self.dense_shape)
        super().__init__()


    def __setitem__(self, key : tuple, component_dict: dict):
        
        if len(key) != 2:
            raise KeyError('The submatrix key provided {} has only one subvector reference; it should be referenced by two subvector names'.format(key))

        if type(key[0]) != str or type(key[1]) != str:
            raise KeyError('The key referencing a submatrix must be tuple of strings, the key provided is {}'.format(key))

        name1 = key[1]
        name2 = key[2]

        if key in self:
            raise KeyError('A submatrix called {} has already been added'.format((name1, name2)))

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

        rows = component_dict['rows']
        cols = component_dict['cols']
        vals = component_dict['vals']
        in_ptr = component_dict['ind_ptr']
        # in_shape is used for setting up views
        vals_shape = component_dict['vals_shape']
        
        allowed_shapes = (shape, shape1 + shape2)
        if size1 == 1:
            allowed_shapes = allowed_shapes + (shape2)
        if size2 == 1:
            allowed_shapes = allowed_shapes + (shape1)
        allowed_dtypes = (int, float)

        def check_shape(given_shape):
            if !(given_shape in allowed_shapes):
                raise ValueError('Given shape {} of dense submatrix cannot be broadcast to match the subvector shapes ({}) and ({})'.format(vals.shape, shape1, shape2))

        def check_dtype(given_array):
            if !(type(given_array) in allowed_dtypes):
                # raise error

        def compare_shapes(shape1, shape2):
            pass

        # If component is dense
        if !(rows) and !(cols):
            if vals:
                check_shape(vals.shape)
                check_dtype(vals)
                # if !(vals.shape in allowed_shapes):
                #     raise ValueError('Given shape {} of dense submatrix cannot be broadcast to match the subvector shapes ({}) and ({})'.format(vals.shape, shape1, shape2))
                # if !(type(vals) in allowed_dtypes):
                #     # add new error

                component_dict['vals_shape'] = vals.shape
            else:
                if vals_shape:
                    check_shape(vals.shape)
                    # if !(vals_shape in allowed_shapes):
                    #     raise ValueError('Given shape {} of the submatrix cannot be broadcast to match the subvector shapes ({}) and ({})'.format(vals_shape, shape1, shape2))  
                        
                    component_dict['vals_shape'] = vals_shape  
                else:
                    component_dict['vals_shape'] = shape

        # If component is COO
        elif rows and cols:
            if cols.shape != rows.shape:
                raise ValueError('Shapes of cols {} and rows {} do not match'.format(cols.shape, rows.shape))
            
            component_dict['vals_shape'] = rows.shape
            
            if vals:
                if rows.shape != vals.shape:
                    raise ValueError('Shapes of vals {} and rows {} do not match'.format(vals.shape, rows.shape))

                if !(type(vals) in allowed_dtypes):
                    # raise error

        # If component is CSC
        elif rows:
            if ind_ptr.shape != (shape[1] + 1,)
                raise ValueError('Shape of the ind_ptr {} and does not match the number of columns {} of the component matrix'.format(ind_ptr.shape, shape[1])

            component_dict['vals_shape'] = rows.shape
            
            # Constant sparse (COO or CSC) matrix component
            if vals:
                if rows.shape != vals.shape:
                    raise ValueError('Shapes of vals {} and rows {} do not match'.format(vals.shape, rows.shape))

                if !(type(vals) in allowed_dtypes):
                    # raise error

        # If component is CSR
        elif cols:
            if ind_ptr.shape != (shape[0] + 1,)
                raise ValueError('Shape of the ind_ptr {} and does not match the number of rows {} of the component matrix'.format(ind_ptr.shape, shape[0])
            
            component_dict['vals_shape'] = cols.shape

            # Constant sparse (CSR) matrix component
            if vals:
                if cols.shape != vals.shape:
                    raise ValueError('Shapes of vals {} and cols {} do not match'.format(vals.shape, cols.shape))
                    
                if !(type(vals) in allowed_dtypes):
                    # raise error



        if component_dict['rows']:
            num_nonzeros = component_dict['rows'].size     
        elif component_dict['cols']:
            num_nonzeros = component_dict['cols'].size
        else:
            num_nonzeros = size1 * size2

        component_dict['start_index'] = self.num_nonzeros
        self.num_nonzeros += num_nonzeros     
        component_dict['end_index'] = self.num_nonzeros # - 1

        super()[name1, name2] = component_dict






    
    

