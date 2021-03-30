Native Formats
==============

Users should define their vectors / matrices using the VectorComponentsDict / MatrixComponentsDict class.
Block matrices can be constructed from Matrix/BlockMatrix objects.
Only the topmost blockmatrix in the hierarchy is allocated memory and all the matrices down the hierarchy stores views to the top blockmatrix.

.. autoclass:: array_manager.core.native_formats.vector_components_dict.VectorComponentsDict
.. autoclass:: array_manager.core.native_formats.vector.Vector
.. autoclass:: array_manager.core.native_formats.matrix_components_dict.MatrixComponentsDict
.. autoclass:: array_manager.core.native_formats.matrix.Matrix
.. autoclass:: array_manager.core.native_formats.block_matrix.BlockMatrix