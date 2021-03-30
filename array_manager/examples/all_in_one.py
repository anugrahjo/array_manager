from array_manager.api import VectorComponentsDict, Vector
from array_manager.api import MatrixComponentsDict, Matrix, BlockMatrix
from array_manager.api import DenseMatrix
from array_manager.api import COOMatrix, CSRMatrix, CSCMatrix

import numpy as np
from guppy import hpy

h = hpy()


vec1_dict = VectorComponentsDict()
vec2_dict = VectorComponentsDict()
vec3_dict = VectorComponentsDict()
vec4_dict = VectorComponentsDict()

vec1_dict['f0'] = dict(shape = (3,))
vec1_dict['f1'] = dict(shape = (2,))
vec2_dict['xo'] = dict(shape = (2,))
vec2_dict['x1'] = dict(shape = (2,))

vec3_dict['g0'] = dict(shape = (1,))
vec3_dict['g1'] = dict(shape = (2,))
vec4_dict['y0'] = dict(shape = (1,))
vec4_dict['y1'] = dict(shape = (1,))


vec1 = Vector(vec1_dict)
vec2 = Vector(vec2_dict)
vec3 = Vector(vec3_dict)
vec4 = Vector(vec4_dict)

mat11_dict = MatrixComponentsDict(vec1_dict, vec2_dict)
mat11_dict['f0', 'x1'] = dict(vals=np.arange(6).reshape(3,2))
mat12_dict = MatrixComponentsDict(vec1_dict, vec4_dict)
mat21_dict = MatrixComponentsDict(vec3_dict, vec2_dict)
mat22_dict = MatrixComponentsDict(vec3_dict, vec4_dict)
mat22_dict['g1', 'y0'] = dict(vals=np.array([5, 6]))


mat11 = Matrix(mat11_dict)
mat12 = Matrix(mat12_dict)
mat21 = Matrix(mat21_dict)
mat22 = Matrix(mat22_dict)

block1 = BlockMatrix([[mat11, mat12]])
block2 = BlockMatrix([[mat21, mat22]])

block = BlockMatrix([[block1], [block2]])

h.setrelheap()

# block.allocate()

# To test allocate() works with and without copy=True, run after commenting out self.update_bottom_up() in the block_matrix.py. This will give correct results when copy=False and incorrect result when copy=True (only the Matrix objects will contain nonzero values, all BlockMatrix objects' data will be populated with zeros)

block.allocate(copy=True)
# block.update_bottom_up()
# block.update_top_down()

mem = h.heap()

print(mem)

dense = DenseMatrix(block)
coo = COOMatrix(block)
csr = CSRMatrix(block)
csc = CSCMatrix(block)

print(mat11.vals.data)
print(mat12.vals.data)
print(mat21.vals.data)
print(mat22.vals.data)
print(block1.vals.data)
print(block2.vals.data)
print(block.vals.data)

print(dense.data)
print(coo.data, coo.rows, coo.cols)
print(csr.data, csr.ind_ptr, csr.cols)
print(csc.data, csc.rows, csc.ind_ptr)