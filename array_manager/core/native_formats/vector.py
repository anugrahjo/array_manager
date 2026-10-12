"""Define the Vector class"""
import numpy as np
import scipy.sparse as sp


class Vector(object):
    """
    Store a vector with optional views into its named components.

    Call ``allocate(setup_views=True)`` to enable access by component name.
    Without explicit data, allocation shares the component dictionary's initial-value array.
    The value for a key (subvector name) is a view of the subvector
    generated from the `self.data` attribute.

    Attributes
    ----------
    data : np.ndarray
        Contiguous values of the concatenated components after allocation.

    Methods
    -------
    allocate(data=None, setup_views=False)
        Set the vector's backing array and optionally create named component views.
    """
    def __init__(self, vector_components_dict):
        """
        Initialize the Vector object by providing a VectorComponentsDict object.

        Parameters
        ----------
        vector_components_dict : VectorComponentsDict
            Dictionary defining the subvector components of the Vector object.
        """
        self.vector_components_dict = vector_components_dict

    def allocate(self, data=None, setup_views=False):
        """
        Set the vector's backing array and optionally create component views.

        Parameters
        ----------
        data : np.ndarray or None, optional
            One-dimensional backing array of length ``self.vector_components_dict.vector_size``.
            The array is used directly without copying. If `None` (the default),
            use ``self.vector_components_dict.vals``, which contains the component values
            supplied at definition time, defaulting to zero where no initial values were supplied.
        setup_views : bool, optional
            If `True`, create views into the backing array, reshaped to each component's
            declared shape, enabling access and assignment through ``vector[name]``.
            If `False` (the default), named component access is disabled.

        Notes
        -----
        This method does not allocate a new array or reset existing values.
        Changes to the vector also affect the supplied array, or the component
        dictionary's ``vals`` array when ``data`` is None.
        Pass a copy as ``data`` when independent storage is needed.

        Calling this method again replaces the backing-array reference and
        recreates or disables named views according to ``setup_views``.
        The supplied array's type and length are not explicitly validated.
        """
        # If data is given, no copy is made, only pointers are stored
        # User is never supposed to access data, can use allocate if needed
        if data is not None:
            self.data = data

        else:
            # New addition
            self.data = data = self.vector_components_dict.vals

            # Previous way
            # self.data = data = np.zeros(
            #     self.vector_components_dict.vector_size)

        self.setup_views_ = setup_views
        if setup_views:
            self.dict_ = self.setup_views()

        else:
            self.dict_ = None

    def setup_views(self):
        """
        Set up views for subvector components that are concatenated into the vector.
        """
        dict_ = {}
        for key, component_dict in self.vector_components_dict.items():
            shape = component_dict['shape']
            ind1 = component_dict['start_index']
            ind2 = component_dict['end_index']
            dict_[key] = self.data[ind1:ind2].reshape(shape)

        return dict_

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, value):
        self.dict_[key][:] = value

    def get_data(self, ):
        return self.data.copy()

    def set_data(self, data):
        self.data[:] = data

    def __len__(self):
        return self.vector_components_dict.vector_size

    def append(self, other):
        new_vector_components_dict = {
            **self.vector_components_dict,
            **other.vector_components_dict
        }
        new_vector = Vector(new_vector_components_dict)
        new_vector.allocate(data=np.append(self.data, other.data),
                            setup_views=self.setup_views_)

        return new_vector

    def check_type_and_size(self, other):
        if isinstance(other, (int, float)):
            pass
        elif isinstance(other, Vector):
            if len(other) != len(self):
                raise TypeError(
                    'Arguments should be objects of the Vector/numpy.ndarray class with equal sizes'
                )
        elif isinstance(other, np.ndarray):
            if len(other.shape) != 1:
                raise TypeError(
                    'Objects of the numpy.ndarray should be vectors')
            if len(other) != len(self):
                raise TypeError(
                    'Arguments should be objects of the Vector/numpy.ndarray class with equal sizes'
                )
        else:
            raise TypeError(
                'Argument should be either an object of the Vector/numpy.ndarray class or a scalar (int or float)'
            )

    def __iadd__(self, other):
        self.check_type_and_size(other)
        if isinstance(other, Vector):
            self.data += other.data
        else:  # isinstance(other, (int, float, np.ndarray))
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
        else:  # isinstance(other, (int, float, np.ndarray))
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
            new_data = self.data**other.data
        else:
            new_data = self.data**other

        new_vector.allocate(data=new_data, setup_views=self.setup_views_)
        return new_vector

    # TODO: The following functionality was never tested or validated
    def __matmul__(self, other):  # (Note: Vector object is the first argument)
        """
        Returns a scalar, a numpy array (vector), or a Vector object that results from the given matrix multiplication.
        """
        if not (isinstance(other,
                           (Vector, Matrix, np.ndarray, DenseMatrix, COOMatrix,
                            CSRMatrix, CSCMatrix, sp.coo.coo_matrix,
                            sp.csr.csr_matrix, sp.csc.csc_matrix))):
            raise TypeError(
                'Arguments should be objects of Vector, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix, sp.coo.coo_matrix,sp.csr.csr_matrix, or sp.csc.csc_matrix  classes'
            )

        # Vector inner product
        if isinstance(other, Vector):
            if len(other) != len(self):
                raise TypeError(
                    'Arguments should be objects of the Vector/numpy.ndarray class with equal sizes'
                )
            else:
                inner_product = self.data @ other.data

            return inner_product

        # Inner product with np.ndarray
        elif isinstance(other, np.ndarray):
            if len(other.shape) not in (1, 2):
                raise TypeError(
                    'Objects of the numpy.ndarray should be vectors/matrices')

            # numpy vector inner product
            if len(other.shape) == 1:
                if len(other) != len(self):
                    raise TypeError(
                        'Arguments should be objects of the Vector/numpy.ndarray class with equal sizes'
                    )
                else:
                    inner_product = self.data @ other

            # numpy matrix inner product
            else:  # len(other.shape) = 2
                if len(self) != other.shape[0]:
                    raise TypeError('Arguments should have compatible shapes')
                else:
                    inner_product = self.data @ other

            return inner_product

        # scipy sparse matrix inner product
        elif isinstance(
                other,
            (sp.coo.coo_matrix, sp.csr.csr_matrix, sp.csc.csc_matrix)):
            if len(self) != other.shape[0]:
                raise TypeError('Arguments should have compatible shapes')
            else:
                inner_product = self.data @ other

            return inner_product

        # Matrix inner product (only case that returns a Vector object)
        elif isinstance(
                other, (Matrix, DenseMatrix, COOMatrix, CSRMatrix, CSCMatrix)):
            if other.dense_shape[0] != len(self):
                raise TypeError('Arguments should have compatible shapes')
            else:
                if isinstance(other, Matrix):
                    inner_product = Vector(
                        other.matrix_components_dict.vector_components_dict2)
                else:
                    inner_product = Vector(other.native.matrix_components_dict.
                                           vector_components_dict2)

                if isinstance(other, DenseMatrix):
                    new_data = self.data @ other.data
                elif isinstance(other, Matrix):
                    scipy_matrix = sp.coo_matrix(
                        (other.vals.data, (other.rows, other.cols)),
                        shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix
                elif isinstance(other, COOMatrix):
                    scipy_matrix = sp.coo_matrix(
                        (other.data, (other.rows, other.cols)),
                        shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix
                elif isinstance(other, CSRMatrix):
                    scipy_matrix = sp.csr_matrix(
                        (other.data, other.cols, other.indptr),
                        shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix
                else:
                    scipy_matrix = sp.csc_matrix(
                        (other.data, other.rows, other.indptr),
                        shape=other.dense_shape)
                    new_data = self.data @ scipy_matrix

        if isinstance(other, Matrix):
            inner_product.allocate(data=new_data,
                                   setup_views=other.matrix_components_dict.
                                   vector_components_dict2.setup_views_)
        else:
            inner_product.allocate(
                data=new_data,
                setup_views=other.native.matrix_components_dict.
                vector_components_dict2.setup_views_)

        return inner_product

        # vec == other
        # vec.set(other)
        # vec *= alpha
        # vec += other
        # ...
        # vec.set_const(0.)
