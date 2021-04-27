"""Define the Vector class"""
import numpy as np
import scipy.sparse as sp


class BlockVector(object):
    """
    Dictionary which contains views for different variables.
    The value corresponding to the key as the name of a subvector represents the view of that subvector which is generated from the self.data attribute of the Vector object.

    Attributes
    ----------
    data : np.ndarray
        Concatenated vector from the dictionary of subvectors 
    """

    def __init__(self, blocks, shapes=None, setup_views=False):
        """
        Initialize the Vector object by allocating a zero vector of desired size.

        Parameters
        ----------
        vector_components_dict : VectorComponentsDict
            List of variables that are concatenated
        """
        self.vector_components_dict = vector_components_dict

    def allocate(self, data=None, copy=False, setup_views=False):
        # If data is given, no copy is made, only poiners are stored
        # User is never supposed to access data, can use allocate if needed
        if data is not None:
            self.data = data
        else:
            self.data = data = np.zeros(self.vector_components_dict.vector_size)

        self.setup_views_ = setup_views
        if setup_views:
            self.dict_ = self.setup_views(self.vector_components_dict)

        else:
            self.dict_ = None

        # New addition
        if data is not None and not copy: 
            pass
        else:
            data = np.zeros(self.num_nonzeros)

        self.vals.allocate(data=data, setup_views=True)

        ind1 = 0
        ind2 = 0
        for i in range(len(self.vector_components_dict)):
                sub_vector = self.sub_vectors[i]

                ind2 += sub_vector.vector_size
                sub_vector.allocate(data=data[ind1:ind2], copy=copy)
                ind1 += sub_vector.vector_size

        if copy:
            self.update_bottom_up()

    def setup_views(self, vector_components_dict):
        """
        Setup views for variables that are concatenated into a single vector.

        Parameters
        ----------
        variables_set : VariablesSet
            List of variables that are concatenated
        """
        dict_ = {}
        for key, component_dict in vector_components_dict.items():
            shape = component_dict['shape']
            ind1 = component_dict['start_index']
            ind2 = component_dict['end_index']
            dict_[key] = self.data[ind1:ind2].reshape(shape)

        return dict_

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, value):
        self.dict_[key][:] = value

    def __len__(self):
        return self.vector_components_dict.vector_size

    def append(self, other):
        new_vector_components_dict = {**self.vector_components_dict, **other.vector_components_dict}
        new_vector = Vector(new_vector_components_dict)
        new_vector.allocate(data=np.append(self.data, other.data), setup_views=self.setup_views_)
        
        return new_vector

    def check_type_and_size(self, other):
        if isinstance(other, (int, float)):
            pass
        elif isinstance(other, Vector):
            if len(other) != len(self) :
                raise TypeError('Arguments should be objects of the Vector/numpy.ndarray class with equal sizes')
        elif isinstance(other, np.ndarray):
            if len(other.shape) != 1:
                raise TypeError('Objects of the numpy.ndarray should be vectors')
            if len(other) != len(self) :
                raise TypeError('Arguments should be objects of the Vector/numpy.ndarray class with equal sizes')
        else:
            raise TypeError('Argument should be either an object of the Vector/numpy.ndarray class or a scalar (int or float)')

        

    def __iadd__(self, other):
        self.check_type_and_size(other)
        if isinstance(other, Vector):
            self.data += other.data
        else:                    # isinstance(other, (int, float, np.ndarray))
            self.data += other

        return self

    def __isub__(self, other):
        self.check_type_and_size(other)
        if isinstance(other, Vector):
            self.data -= other.data
        else:
            self.data -= other

        return self  

    def __imul__(self, other):
        self.check_type_and_size(other)  
        if isinstance(other, Vector):
            self.data *= other.data
        else:
            self.data *= other
    
        return self

    def __itruediv__(self, other):
        self.check_type_and_size(other)
        if isinstance(other, Vector):
            self.data /= other.data
        else:
            self.data /= other

        return self

    def __ipow__(self, other):
        self.check_type_and_size(other)
        if isinstance(other, Vector):
            self.data **= other.data
        else:
            self.data **= other
        
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
        new_vector = Vector(self.vector_components_dict)

        if isinstance(other, Vector):
            new_data = self.data + other.data
        else:                   # isinstance(other, (int, float, np.ndarray))
            new_data = self.data + other
        
        new_vector.allocate(data=new_data, setup_views=self.setup_views_)
        return new_vector

    def __sub__(self, other):
        self.check_type_and_size(other)
        new_vector = Vector(self.vector_components_dict)
        
        if isinstance(other, Vector):
            new_data = self.data - other.data
        else:
            new_data = self.data - other

        new_vector.allocate(data=new_data, setup_views=self.setup_views_)
        return new_vector

    def __mul__(self, other):
        self.check_type_and_size(other)  
        new_vector = Vector(self.vector_components_dict)
        
        if isinstance(other, Vector):
            new_data = self.data * other.data
        else:
            new_data = self.data * other

        new_vector.allocate(data=new_data, setup_views=self.setup_views_)
        return new_vector
    
    def __truediv__(self, other):
        self.check_type_and_size(other)
        new_vector = Vector(self.vector_components_dict)
        
        if isinstance(other, Vector):
            new_data = self.data / other.data
        else:
            new_data = self.data / other

        new_vector.allocate(data=new_data, setup_views=self.setup_views_)
        return new_vector

    def __pow__(self, other):
        self.check_type_and_size(other)
        new_vector = Vector(self.vector_components_dict)
        
        if isinstance(other, Vector):
            new_data = self.data ** other.data
        else:
            new_data = self.data ** other

        new_vector.allocate(data=new_data, setup_views=self.setup_views_)
        return new_vector

    def __matmul__(self, other):  # (Note: Vector object is the first argument)
        """
        Returns a scalar, a numpy array (vector), or a Vector object that results from the given matrix multiplication.
        """
        if not(isinstance(other, (Vector, Matrix, np.ndarray, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix, sp.coo.coo_matrix, sp.csr.csr_matrix, sp.csc.csc_matrix))):
            raise TypeError('Arguments should be objects of Vector, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix, sp.coo.coo_matrix,sp.csr.csr_matrix, or sp.csc.csc_matrix  classes')
        
        # Vector inner product
        if isinstance(other, Vector):
            if len(other) != len(self):
                raise TypeError('Arguments should be objects of the Vector/numpy.ndarray class with equal sizes')
            else:
                inner_product = self.data @ other.data
                
            return inner_product

        # Inner product with np.ndarray 
        elif isinstance(other, np.ndarray):
            if len(other.shape) not in (1, 2):
                raise TypeError('Objects of the numpy.ndarray should be vectors/matrices')
            
            # numpy vector inner product
            if len(other.shape) == 1:
                if len(other) != len(self) :
                    raise TypeError('Arguments should be objects of the Vector/numpy.ndarray class with equal sizes')
                else:
                    inner_product = self.data @ other

            # numpy matrix inner product
            else:                             # len(other.shape) = 2
                if len(self) != other.shape[0] :
                    raise TypeError('Arguments should have compatible shapes')
                else:
                    inner_product = self.data @ other

            return inner_product

        # scipy sparse matrix inner product
        elif isinstance(other, (sp.coo.coo_matrix, sp.csr.csr_matrix, sp.csc.csc_matrix)):
            if len(self) != other.shape[0] :
                raise TypeError('Arguments should have compatible shapes')
            else:
                inner_product = self.data @ other

            return inner_product

        # Matrix inner product (only case that returns a Vector object)
        elif isinstance(other, (Matrix, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix)):
            if other.dense_shape[0] != len(self):
                raise TypeError('Arguments should have compatible shapes')
            else:
                if isinstance(other, Matrix): 
                    inner_product = Vector(other.matrix_components_dict.vector_components_dict2)
                else:
                    inner_product = Vector(other.native.matrix_components_dict.vector_components_dict2)
      
                if isinstance(other, DenseMatrix):
                    new_data = self.data @ other.data
                elif isinstance(other, Matrix):
                    scipy_matrix = sp.coo_matrix((other.vals.data, (other.rows, other.cols)), shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix
                elif isinstance(other, COOMatrix):
                    scipy_matrix = sp.coo_matrix((other.data, (other.rows, other.cols)), shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix
                elif isinstance(other, CSRMatrix):
                    scipy_matrix = sp.csr_matrix((other.data, other.cols, other.indptr), shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix
                else:
                    scipy_matrix = sp.csc_matrix((other.data, other.rows, other.indptr), shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix

        if isinstance(other, Matrix): 
            inner_product.allocate(data=new_data, setup_views=other.matrix_components_dict.vector_components_dict2.setup_views_)
        else:
            inner_product.allocate(data=new_data, setup_views=other.native.matrix_components_dict.vector_components_dict2.setup_views_)

        return inner_product

        # vec == other
        # vec.set(other)
        # vec *= alpha
        # vec += other
        # ...
        # vec.set_const(0.)
