Standard Formats
================

These are the standard output formats that could be requested by the user. These formats would be generated from objects of Matrix / BlockMatrix class.

Ex. x = COOMatrix(X) generates a sparse matrix object x in the coo format from an already defined matrix in the native format (either a Matrix object or a BlockMatrix object) 

.. autoclass:: array_manager.core.standard_formats.dense_matrix.DenseMatrix

.. autoclass:: array_manager.core.standard_formats.coo_matrix.COOMatrix

.. autoclass:: array_manager.core.standard_formats.csr_matrix.CSRMatrix

.. autoclass:: array_manager.core.standard_formats.csc_matrix.CSCMatrix