# array_manager

Tools for assembling vectors and block matrices from named components, with
conversion to dense and sparse (COO, CSR, CSC) matrix formats. Built on NumPy
and SciPy.

## Installation

From the repository root:

```sh
python -m pip install -e .
```

## Usage

```python
from array_manager.api import Vector, VectorComponentsDict

components = VectorComponentsDict()
components["position"] = {"shape": (3,)}
components["velocity"] = {"shape": (3,)}

vector = Vector(components)
vector.allocate(setup_views=True)
vector["position"] = [1.0, 2.0, 3.0]

print(vector.data)  # [1. 2. 3. 0. 0. 0.]
```

## Example: a KKT system with mixed matrix formats

Minimize `0.5 * x.T @ H @ x` subject to `A @ x = b`. The optimality
conditions give the KKT system `[[H, A.T], [A, 0]] @ [x, lambda] = [0, b]`.
Here, the Hessian is dense, the constraint matrix is CSR, and its transpose
is CSC. Pass each sparse block's values, indices, and index pointers to
`MatrixComponentsDict`; omitted blocks are zero.

```python
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

print(solution[:2])  # x: [0.66666667 0.33333333]
print(solution[2:])  # lambda: [-1.33333333]
```

See [array_manager/examples](array_manager/examples) for more examples.