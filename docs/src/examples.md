# Examples

## A KKT system with mixed matrix formats

Minimize `0.5 * x.T @ H @ x` subject to `A @ x = b`. The optimality
conditions give the KKT system `[[H, A.T], [A, 0]] @ [x, lambda] = [0, b]`.
Here, the Hessian is dense, the constraint Jacobian is CSR, and its transpose is CSC. 
Pass each sparse block's values, indices, and index pointers to `MatrixComponentsDict`.
Omitted blocks are assumed to be zeros (e.g., the lower-right block here).

```{testcode}

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from array_manager.api import VectorComponentsDict, MatrixComponentsDict, Matrix, CSRMatrix

H = np.diag([2.0, 4.0])
A = csr_matrix([[1.0, 1.0]])
AT = A.T  # CSC format
b = np.array([1.0])

variables = VectorComponentsDict()
variables["x"] = {"shape": (2,)}
variables["lambda"] = {"shape": (1,)}

blocks = MatrixComponentsDict(variables, variables)
blocks["x", "x"] = {"vals": H}  # Dense block
blocks["lambda", "x"] = {  # CSR block
    "vals": A.data, "cols": A.indices, "ind_ptr": A.indptr,
}
blocks["x", "lambda"] = {  # CSC block
    "vals": AT.data, "rows": AT.indices, "ind_ptr": AT.indptr,
}

kkt = Matrix(blocks)
kkt.allocate()
kkt_csr = CSRMatrix(kkt).get_std_array()
solution = spsolve(kkt_csr, np.concatenate([np.zeros(2), b]))

np.testing.assert_allclose(solution, [2 / 3, 1 / 3, -4 / 3])
np.testing.assert_allclose(A @ solution[:2], b)
print("x     :", solution[:2])  # x     : [0.66666667 0.33333333]
print("lambda:", solution[2:])  # lambda: [-1.33333333]
```

```{testoutput}

x     : [0.66666667 0.33333333]
lambda: [-1.33333333]
```

## More examples

The repository's `examples/` directory contains additional scripts. 
The vector example is included below directly from its source:

```{literalinclude} ../../examples/using_vectors.py
:language: python
```
