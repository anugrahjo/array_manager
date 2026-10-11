# array_manager

Tools for assembling vectors and block matrices from named components, with
conversion to dense and sparse (COO, CSR, CSC) matrix formats. 
Built on NumPy and SciPy.

## Installation

To install the latest commit from the main branch, run the following command in the terminal:
```sh
pip install git+https://github.com/anugrahjo/array_manager.git
```

To uninstall, run:
```sh
pip uninstall array_manager
```

To upgrade to the latest commit, uninstall and then reinstall using:
```sh
pip uninstall array_manager
pip install git+https://github.com/anugrahjo/array_manager.git
```

To install array_manager in development mode, clone the repository and install it using:
```sh
git clone https://github.com/anugrahjo/array_manager.git
pip install -e ./array_manager
```
The `-e` flag installs the package in editable mode, 
allowing you to modify the source code without needing to reinstall it.

To upgrade to the latest commit in development mode, navigate to the repository root and run:
```sh
git pull
```

## Simplest use case
Create a vector composed of two subvectors or vector components.

```python
from array_manager.api import Vector, VectorComponentsDict

# Define vector components with their names and tuple shapes
# Components are initialized to zero unless initial values are supplied
components = VectorComponentsDict()
components["position"] = {"shape": (3,)}
components["velocity"] = {"shape": (3,)}

# Create a vector defined by the components
vector = Vector(components)

# Allocate memory and enable views to access components later by name
vector.allocate(setup_views=True)

# Set the values for a vector component to overwrite its initial value
vector["position"] = [1.0, 2.0, 3.0]

print(vector.data)  # [1. 2. 3. 0. 0. 0.]
```

## Example: a KKT system with mixed matrix formats

Minimize `0.5 * x.T @ H @ x` subject to `A @ x = b`. The optimality
conditions give the KKT system `[[H, A.T], [A, 0]] @ [x, lambda] = [0, b]`.
Here, the Hessian is dense, the constraint Jacobian is CSR, and its transpose is CSC. 
Pass each sparse block's values, indices, and index pointers to `MatrixComponentsDict`.
Omitted blocks are assumed to be zeros (e.g., the lower-right block here).

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

print("x     :", solution[:2])  # x     : [0.66666667 0.33333333]
print("lambda:", solution[2:])  # lambda: [-1.33333333]
```

See [examples](examples/) for more examples.
