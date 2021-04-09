from array_manager.api import VectorComponentsDict, Vector
from array_manager.api import MatrixComponentsDict, Matrix, BlockMatrix
from array_manager.api import DenseMatrix
from array_manager.api import COOMatrix, CSRMatrix, CSCMatrix

import numpy as np

# Objective function: F(x, y, z) = 
# Design variable vector: v = [x, y, z].T

x_vals = np.array([[1, 2], [3, 4]])
y_vals = np.array([[1], [2]])
# y_vals = np.array([[1, 2]])
z_vals = np.array([1, 2])

lag_mult_vals = np.ndarray([5, 5, 5])

v_dict = VectorComponentsDict()

v_dict['x'] = dict(shape = (2,2))
v_dict['y'] = dict(shape = (2,1))
# v_dict['y'] = dict(shape = (1,2))
v_dict['z'] = dict(shape = (2,))

# Design variable vector v consists of x, y, and z vectors
v = Vector(v_dict)
v.allocate(setup_views=True)
x = v['x'] = x_vals
y = v['y'] = y_vals
z = v['z'] = z_vals

# f = x[0,0] ** 2 + x[0,0] * x[0,1] + x[0,2] * np.sin(x[1,0]) + x[1,1] + np.log(y[0,0]) + y[1,0] ** 3 + np.exp(z[0] * z[1]) + z[0] * x[1,1] ** 2

# Constraint vector c consists of x, y, and z vectors
c_dict = VectorComponentsDict()
c_dict['C1'] = dict(shape = (1,))
c_dict['C2'] = dict(shape = (1,))
c_dict['C3'] = dict(shape = (1,))

c = Vector(c_dict)
c.allocate(setup_views=True)

lag_mult = Vector(c_dict)
lag_mult.allocate(setup_views=True)
lag_mult.vals = lag_mult_vals

# Objective gradient (gradient of objective function F with respect to (wrt) v)
pFpv = Vector(v_dict)
pFpv.allocate(setup_views=True)
pFpv['x'] = np.array([[2 * x[0, 0] + x[0, 1], x[0, 0] + np.sin(x[1, 0])], [x[0, 1] * np.cos(x[1, 0]), 2 * z[0] * x[1,1]]])
pFpv['y'] = np.array([[1 / y[0, 0] ], [3 * y[1, 0] ** 2]])
pFpv['z'] = np.array([z[1] * np.exp(z[0] * z[1]) + x[1,1] ** 2, z[0] * np.exp(z[0] * z[1])])

# Constraint gradients (gradients of constraint functions C_i with respect to v)
pCpv_dict = MatrixComponentsDict(c_dict, v_dict)
pCpv_dict['C1','x'] = dict(vals=np.array([2 * x[0, 0], 2 * x[0, 1]]), rows=np.array([0, 0]), cols=np.array([0, 1]))
pCpv_dict['C2','y'] = dict(vals=np.array([[1], [-1]])) # Note: Issue with shape
pCpv_dict['C3','z'] = dict(vals=np.array([-np.cos(z[0]), 1]))

pCpv = Matrix(pCpv_dict)



# # Gradients of C1 with respect to v
# pC1pv = Vector(v_dict)
# data1 = np.zeros(v_dict.vector_size)
# np.put(data1, [0, 1], [2 * x[0, 0], 2 * x[0, 1]])
# pC1pv.allocate(data=data1, setup_views=True)

# # Gradients of C2 with respect to v
# pC2pv = Vector(v_dict)
# data2 = np.zeros(v_dict.vector_size)
# np.put(data2, [5, 6], [1, -1])
# pC2pv.allocate(data=data2, setup_views=True)

# # Gradients of C3 with respect to v
# pC3pv = Vector(v_dict)
# pC3pv.allocate(setup_views=True)
# pC3pv['z'] = np.array([-np.cos(z[0]), 1])


# Hessian of the objective wrt v
p2Fpvv_dict = MatrixComponentsDict(v_dict, v_dict)
p2Fpvv_dict['x', 'x'] = dict(rows=np.array([0, 0, 1, 1, 2, 2, 3]), cols=np.array([0, 1, 0, 2, 1, 2, 3]), vals=np.array([2, 1, 1, np.cos(x[1,0]), np.cos(x[1,0]), -x[0, 1] * np.sin(x[1,0]), 2 * z[0]]))
p2Fpvv_dict['y', 'y'] = dict(rows=np.array([0, 0]), cols=np.array([1, 1]), vals=np.array([-(1 / y[0,0] ** 2), 6 * y[1,0]]))
p2Fpvv_dict['z', 'z'] = dict(vals=np.array([[z[1]**2 * np.exp(z[0] * z[1]), (1 + z[0] * z[1]) * np.exp(z[0] * z[1])], [(1 + z[0] * z[1]) * np.exp(z[0] * z[1]), z[0]**2 * np.exp(z[0] * z[1])]]))
p2Fpvv_dict['x', 'z'] = dict(rows=np.array([3,]), cols=np.array([0,]) ,vals=np.array([2 * x[1,1], ]))
p2Fpvv_dict['z', 'x'] = dict(rows=np.array([0,]), cols=np.array([3,]) ,vals=np.array([2 * x[1,1], ]))

p2Fpvv = Matrix(p2Fpvv_dict)
# p2Fpvv.allocate(setup_views=True)
# p2Fpvv['x', 'x'] = np.array([])
# p2Fpvv['y', 'y'] = np.array([])
# p2Fpvv['z', 'z'] = np.array([[, ], [, ]])

# Hessian of the constraints wrt v
p2Cpvv = MatrixComponentsDict(c_dict, v_dict)

# Hessian of C1 wrt v
p2C1pvv_dict = MatrixComponentsDict(v_dict, v_dict)
p2C1pvv_dict['x', 'x'] = dict(rows=np.array([0, 1]), cols=np.array([0, 1]), vals=np.array([2, 2]))
p2C1pvv = Matrix(p2C1pvv_dict)


# Hessian of C2 wrt v : need not be declared since it contains only zeros
# p2C2pvv_dict = MatrixComponentsDict(v_dict, v_dict)
# p2C2pvv = Matrix(p2C2pvv_dict)
# p2C2pvv

# Hessian of C3 wrt v
p2C3pvv_dict = MatrixComponentsDict(v_dict, v_dict)
p2C3pvv_dict['z', 'z'] = dict(rows=np.array([0,]), cols=np.array([0,]),  vals=np.array([np.sin(z[0]),]))
p2C3pvv = Matrix(p2C3pvv_dict)
# p2C3pvv['z', 'z'] = np.array([np.sin(z[0])]) # not possible since matrix is not allocated yet

# Gradient of the Lagrangian wrt v
pLpv = pFpv + pCpv.transpose() @ lag_mult # transpose(pCpv) doesn't work?

# Hessian of the Lagrangian wrt v
p2Lpvv = p2fpvv + lag_mult['C1'] * p2C1pvv + lag_mult['C2'] * p2C2pvv + lag_mult['C3'] * p2C3pvv

# KKT system for this problem

KKT_matrix = BlockMatrix([[p2Lpvv, transpose(pCpv)], [p2Cpvv, 0]])
rhs_vector = pLpv.append(c)





