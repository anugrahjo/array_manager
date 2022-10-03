"""Define the Vector class"""
import numpy as np
from array_manager.core.native_formats.vector_components_dict import VectorComponentsDict
from array_manager.core.native_formats.matrix_components_dict import MatrixComponentsDict
from array_manager.core.native_formats.vector import Vector
from array_manager.core.standard_formats.dense_matrix import DenseMatrix
from array_manager.core.standard_formats.coo_matrix import COOMatrix
from array_manager.core.standard_formats.csr_matrix import CSRMatrix
from array_manager.core.standard_formats.csc_matrix import CSCMatrix
import scipy.sparse as sp


class Matrix(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a variable represents the view of that variable which is generated from the self.vals attribute of the Vector object.

    Attributes
    ----------
    vals : Vector
        Concatenated vector of a list of variables 

    """

    def __init__(self, matrix_components_dict, setup_views=False):
        """
        Initialize the Matrix object by allocating a zero vector of desired size.

        Parameters
        ----------
        matrix_components_dict : VariablesList
            List of variables that are concatenated
        """
        self.matrix_components_dict = matrix_components_dict
        self.dense_shape = matrix_components_dict.dense_shape
        self.dense_size = matrix_components_dict.dense_size
        self.num_nonzeros = matrix_components_dict.num_nonzeros

        if (self.num_nonzeros==0) and (self.dense_size==0):
            self.density = None
        else:
            self.density = float(self.num_nonzeros / self.dense_size)
        self.rows = np.zeros(self.num_nonzeros, dtype=int)
        self.cols = np.zeros(self.num_nonzeros, dtype=int)

        vector_components_dict = VectorComponentsDict()

        for key, component_dict in matrix_components_dict.items():
            shape = component_dict['shape']
            # COO arrays from given CSR or CSC arrays
            if isinstance(component_dict['ind_ptr'], np.ndarray):
                ind_ptr = component_dict['ind_ptr']
                # Compute differences between consecutive elements in the ind_ptr array
                num_repeats = np.ediff1d(ind_ptr)
                if isinstance(component_dict['rows'], np.ndarray):
                    component_dict['cols'] = np.repeat(np.arange(shape[1]), num_repeats)

                elif isinstance(component_dict['cols'], np.ndarray):
                    component_dict['rows'] =  np.repeat(np.arange(shape[0]), num_repeats)

                # need this?
                # del matrix_dict['ind_ptr']

            # Dense matrix component
            elif (component_dict['rows'] is None) and (component_dict['cols'] is None):
                # print('Here')
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

        self.vals = Vector(vector_components_dict)

        # for key, component_dict in matrix_components_dict.items():
        #     vals = self.component_dict['vals']
        #     if vals:
        #         self[key] = vals

    def __getitem__(self, key):
        return self.vals[key]

    def __setitem__(self, key, value):
        self.vals[key] = value

    def allocate(self, copy=False, data=None):
        
        if data is not None and not copy: 
            pass
        else:
            data = np.zeros(self.num_nonzeros)

        self.vals.allocate(data=data, setup_views=True)

        for key, component_dict in self.matrix_components_dict.items():
            vals = component_dict['vals']
            if vals is not None:
                self[key] = vals

        # ind1 = 0
        # ind2 = 0
        # for i, j in self.sub_matrices:
        #     sub_matrix = self.sub_matrices[i, j]

        #     ind2 += sub_matrix.num_nonzeros
        #     sub_matrix.allocate(data[ind1:ind2])
        #     ind1 += sub_matrix.num_nonzeros

    def update_bottom_up(self):
        pass

    def update_top_down(self):
        pass

    def transpose(self):
        # Note: Assigning values to the transpose matrix will yield wrong results. Need to be careful while working with transpose matrices.

        new_matrix_components_dict = MatrixComponentsDict(self.matrix_components_dict.vector_components_dict2, self.matrix_components_dict.vector_components_dict1)
        new_matrix = Matrix(new_matrix_components_dict)
        # new_matrix.dense_shape = self.dense_shape[::-1]
        # new_matrix.dense_size = self.dense_size
        new_matrix.num_nonzeros = self.num_nonzeros

        new_matrix.density = self.density
        new_matrix.cols = self.rows
        new_matrix.rows = self.cols
        new_matrix.vals = self.vals

        return new_matrix


    # len() returns the number of nonzeros in the matrix
    def __len__(self):
        return len(self.vals)

    def check_type_and_size_inplace(self, other):
        if isinstance(other, (int, float)):
            pass
        
        elif isinstance(other, Matrix):
            if other.dense_shape != self.dense_shape:
                raise TypeError('Arguments should be objects of the Matrix class with same shapes')
            if len(other) != self.num_nonzeros or other.rows != self.rows or other.cols != self.cols:
                raise TypeError('Arguments should be objects of the Matrix class with same sparsity structure')

        else:
            raise TypeError('Argument should be either an object of the Matrix class or a scalar (int or float)')

    def check_type_and_size(self, other):
        if isinstance(other, (int, float)):
            pass

        # elif isinstance(other, Vector):
        #     if len(other) != self.dense_shape[1]:
        #         raise TypeError('Arguments should be objects of the Vector/Matrix/numpy.ndarray class with compatible shapes')
        
        elif isinstance(other, (Matrix, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix)):
            if other.dense_shape != self.dense_shape:
                raise TypeError('Arguments should be objects of the Matrix/numpy.ndarray class with same shapes')
            # if len(other) != self.num_nonzeros:
            #     raise TypeError('Arguments should be objects of the Vector/Matrix/numpy.ndarray class with compatible shapes')
        elif isinstance(other, (np.ndarray, sp.csr.csr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix)):
            if len(other.shape) != 2:
                raise TypeError('Objects of the numpy.ndarray should be matrices')
            if other.shape != self.dense_shape:
                raise TypeError('Argument should be objects of the numpy/scipy array class with same shapes')
        else:
            raise TypeError('Argument should be either an object of the Matrix/numpy.ndarray class or a scalar (int or float)')

    def scipy_coo(self, native_matrix):
        return sp.coo_matrix((native_matrix.data, (native_matrix.rows, native_matrix.cols)), shape=native_matrix.dense_shape)

    def __iadd__(self, other):
        self.check_type_and_size_inplace(other)
        if isinstance(other, (int, float)):
            raise NotImplementedError('Adding a nonzero scalar to a sparse matrix is not supported')
        else:              # isinstance(other, (int, float))
            self.vals += other.vals

        return self
          
    def __isub__(self, other):
        self.check_type_and_size_inplace(other)
        if isinstance(other, (int, float)):
            raise NotImplementedError('Subtracting a nonzero scalar to a sparse matrix is not supported')
        else:              # isinstance(other, (int, float))
            self.vals -= other.vals

        return self

    def __imul__(self, other):
        self.check_type_and_size_inplace(other)  
        if isinstance(other, Matrix):
            self.vals *= other.vals
        else:
            self.vals *= other
    
        return self

    def __itruediv__(self, other):
        self.check_type_and_size_inplace(other)  
        if isinstance(other, Matrix):
            self.vals /= other.vals
        else:
            self.vals /= other

        return self

    def __ipow__(self, other):
        self.check_type_and_size_inplace(other)  
        if isinstance(other, Matrix):
            self.vals **= other.vals
        else:
            self.vals **= other
        
        return self


    # In-place matrix multiplication is not (yet) supported. Use 'a = a @ b' instead of 'a @= b'.
    
    # def __imatmul__(self, other):
    #     if other.dense_shape
    #     if isinstance(other, Matrix):
    #         self.data @= other.data
    #     else:
    #         raise TypeError('Argument should be either an object of the Vector class or a scalar (int or float)')

    def __add__(self, other):
        self.check_type_and_size(other)
        # Returns Matrix object
        if isinstance(other, (int, float)):
            new_matrix = Matrix(self.matrix_components_dict)
            new_data = self.vals.data + other
            new_matrix.allocate(data=new_data)
            return new_matrix

        elif isinstance(other, Matrix) and len(other) == self.num_nonzeros: 
            # Returns Matrix object
            if other.rows == self.rows and other.cols == self.cols:
                new_matrix = Matrix(self.matrix_components_dict)
                new_data = self.vals.data + other.vals.data
                new_matrix.allocate(data=new_data)
                return new_matrix
            # Returns scipy.coo.coo_matrix object
            else:
                return self.scipy_coo(self) + self.scipy_coo(other)

        # Returns dense np.ndarray object # new Densematrix is not stored?
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self).data + other.data
        elif isinstance(other, COOMatrix):
            scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
            return self.scipy_coo(self) + scipy_matrix
        elif isinstance(other, CSRMatrix):
            scipy_matrix = sp.csr_matrix((other.data, other.cols, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) + scipy_matrix
        elif isinstance(other, CSCMatrix):
            scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) + scipy_matrix

        else: # isinstance(other, (np.ndarray, sp.csr.csr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix))
            return self.scipy_coo(self) + other

    def __sub__(self, other):
        self.check_type_and_size(other)
        # Returns Matrix object
        if isinstance(other, (int, float)):
            new_matrix = Matrix(self.matrix_components_dict)
            new_data = self.vals.data - other
            new_matrix.allocate(data=new_data)
            return new_matrix

        elif isinstance(other, Matrix) and len(other) == self.num_nonzeros: 
            # Returns Matrix object
            if other.rows == self.rows and other.cols == self.cols:
                new_matrix = Matrix(self.matrix_components_dict)
                new_data = self.vals.data - other.vals.data
                new_matrix.allocate(data=new_data)
                return new_matrix
            # Returns scipy.coo.coo_matrix object
            else:
                return self.scipy_coo(self) - self.scipy_coo(other)

        # Returns dense np.ndarray object # new Densematrix is not stored?
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self).data - other.data
        elif isinstance(other, COOMatrix):
            scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
            return self.scipy_coo(self) - scipy_matrix
        elif isinstance(other, CSRMatrix):
            scipy_matrix = sp.csr_matrix((other.data, other.cols, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) - scipy_matrix
        elif isinstance(other, CSCMatrix):
            scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) - scipy_matrix

        else: # isinstance(other, (np.ndarray, sp.csr.csr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix))
            return self.scipy_coo(self) - other

    def __mul__(self, other):
        self.check_type_and_size(other)
        # Returns Matrix object
        if isinstance(other, (int, float)):
            new_matrix = Matrix(self.matrix_components_dict)
            new_data = self.vals.data * other
            new_matrix.allocate(data=new_data)
            return new_matrix

        elif isinstance(other, Matrix) and len(other) == self.num_nonzeros: 
            # Returns Matrix object
            if other.rows == self.rows and other.cols == self.cols:
                new_matrix = Matrix(self.matrix_components_dict)
                new_data = self.vals.data * other.vals.data
                new_matrix.allocate(data=new_data)
                return new_matrix
            # Returns scipy.coo.coo_matrix object
            else:
                return self.scipy_coo(self) * self.scipy_coo(other)

        # Returns dense np.ndarray object # new Densematrix is not stored?
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self).data * other.data
        elif isinstance(other, COOMatrix):
            scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
            return self.scipy_coo(self) * scipy_matrix
        elif isinstance(other, CSRMatrix):
            scipy_matrix = sp.csr_matrix((other.data, other.cols, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) * scipy_matrix
        elif isinstance(other, CSCMatrix):
            scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) * scipy_matrix

        else: # isinstance(other, (np.ndarray, sp.csr.csr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix))
            return self.scipy_coo(self) * other
    
    def __truediv__(self, other):
        self.check_type_and_size(other)
        # Returns Matrix object
        if isinstance(other, (int, float)):
            new_matrix = Matrix(self.matrix_components_dict)
            new_data = self.vals.data / other
            new_matrix.allocate(data=new_data)
            return new_matrix

        elif isinstance(other, Matrix) and len(other) == self.num_nonzeros: 
            # Returns Matrix object
            if other.rows == self.rows and other.cols == self.cols:
                new_matrix = Matrix(self.matrix_components_dict)
                new_data = self.vals.data / other.vals.data
                new_matrix.allocate(data=new_data)
                return new_matrix
            # Returns scipy.coo.coo_matrix object
            else:
                return self.scipy_coo(self) / self.scipy_coo(other)

        # Returns dense np.ndarray object # new Densematrix is not stored?
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self).data / other.data
        elif isinstance(other, COOMatrix):
            scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
            return self.scipy_coo(self) / scipy_matrix
        elif isinstance(other, CSRMatrix):
            scipy_matrix = sp.csr_matrix((other.data, other.cols, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) / scipy_matrix
        elif isinstance(other, CSCMatrix):
            scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) / scipy_matrix

        else: # isinstance(other, (np.ndarray, sp.csr.csr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix))
            return self.scipy_coo(self) / other

    def __pow__(self, other):
        self.check_type_and_size(other)
        # Returns Matrix object
        if isinstance(other, (int, float)):
            new_matrix = Matrix(self.matrix_components_dict)
            new_data = self.vals.data ** other
            new_matrix.allocate(data=new_data)
            return new_matrix

        elif isinstance(other, Matrix) and len(other) == self.num_nonzeros: 
            # Returns Matrix object
            if other.rows == self.rows and other.cols == self.cols:
                new_matrix = Matrix(self.matrix_components_dict)
                new_data = self.vals.data ** other.vals.data
                new_matrix.allocate(data=new_data)
                return new_matrix
            # Returns scipy.coo.coo_matrix object
            else:
                return self.scipy_coo(self) ** self.scipy_coo(other)

        # Returns dense np.ndarray object # new Densematrix is not stored?
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self).data ** other.data
        elif isinstance(other, COOMatrix):
            scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
            return self.scipy_coo(self) ** scipy_matrix
        elif isinstance(other, CSRMatrix):
            scipy_matrix = sp.csr_matrix((other.data, other.cols, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) ** scipy_matrix
        elif isinstance(other, CSCMatrix):
            scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
            return self.scipy_coo(self) ** scipy_matrix

        else: # isinstance(other, (np.ndarray, sp.csr.csr_matrix, sp.csc.csc_matrix, sp.coo.coo_matrix))
            return self.scipy_coo(self) ** other

    def __matmul__(self, other):  # (Note: Vector object is the first argument)
        """
        Returns a scalar, a numpy array (vector), or a Vector object that results from the given matrix multiplication.
        """
        if not(isinstance(other, (Vector, np.ndarray, Matrix, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix, sp.coo.coo_matrix, sp.csr.csr_matrix, sp.csc.csc_matrix))):
            raise TypeError('Arguments should be objects of Vector, np.ndarray, Matrix, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix, sp.coo.coo_matrix,sp.csr.csr_matrix, or sp.csc.csc_matrix  classes')
        
        # Vector inner product
        if isinstance(other, Vector):
            if len(other) != self.dense_shape[1]:
                raise TypeError('Arguments should have compatible shapes')
            else:
                inner_product = self.scipy_coo(self) @ other.data
                
            return inner_product

        # Inner product with np.ndarray 
        elif isinstance(other, np.ndarray):
            if len(other.shape) not in (1, 2):
                raise TypeError('Objects of the numpy.ndarray should be vectors/matrices')
            
            # numpy vector inner product
            if len(other.shape) == 1:
                if len(other) != self.dense_shape[1]:
                    raise TypeError('Arguments should have compatible shapes')
                else:
                    inner_product = self.scipy_coo(self) @ other

            # numpy matrix inner product
            else:                             # len(other.shape) = 2
                if other.shape[0] != self.dense_shape[1] :
                    raise TypeError('Arguments should have compatible shapes')
                else:
                    inner_product = self.scipy_coo(self) @ other

            return inner_product

        # scipy sparse matrix inner product (returns a scipy matrix)
        elif isinstance(other, (sp.coo.coo_matrix, sp.csr.csr_matrix, sp.csc.csc_matrix)):
            if self.dense_shape[1] != other.shape[0] :
                raise TypeError('Arguments should have compatible shapes')
            else:
                inner_product = self.scipy_coo(self) @ other

            return inner_product

        # Matrix inner product (returns a scipy matrix)
        elif isinstance(other, (Matrix, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix)):
            if self.dense_shape[1] != other.dense_shape[0]:
                raise TypeError('Arguments should have compatible shapes')
            else:
                # inner_product = Vector(other.native.matrix_components_dict.vector_components_dict2)
      
                if isinstance(other, DenseMatrix):
                    new_data = self.scipy_coo(self) @ other.data
                elif isinstance(other, COOMatrix):
                    scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
                    new_data = self.scipy_coo(self) @ scipy_matrix
                elif isinstance(other, CSRMatrix):
                    scipy_matrix = sp.csr_matrix((data, other.cols, other.indptr), shape=other.dense_shape)
                    new_data = self.scipy_coo(self) @ scipy_matrix
                else:
                    scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
                    new_data = self.scipy_coo(self) @ scipy_matrix

        # inner_product.allocate(data=new_data, setup_views=other.native.matrix_components_dict.vector_components_dict2.setup_views_)
        return new_data
