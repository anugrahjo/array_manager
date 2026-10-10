"""Define the VectorComponentsDict class"""
import numpy as np
from typing import Dict


class VectorComponentsDict(dict):
    """
    Dictionary of dictionaries representing a vector composed of multiple subvectors. 
    Each dictionary within this dictionary represents the data corresponding to a subvector.
    For example, an object of VectorComponentsDict can be a dictionary of variables
    in which each variable is a design variable or a constraint vector
    in an optimization problem.
    Each dictionary in this dictionary (which stores the data pertaining to a subvector) 
    contains values corresponding to nine keys, namely,
    - shape       : shape of the subvector
    - size        : size (number of entries) of the subvector
    - start_index : starting index of the subvector in the concatenated contiguous vector containing all subvectors from the VectorComponentsDict object
    - end_index   : ending index of the subvector in the concatenated contiguous vector containing all subvectors from the VectorComponentsDict object
    - vals        : an initial value for the subvector
    - upper       : upper limit on the values inside the subvector
    - lower       : lower limit on the values inside the subvector
    - equals      : equality constraint on the values inside the subvector
    - scaler      : a scaling parameter to be applied on the subvector when passing the values to another client, e.g. optimizer.
    
    Attributes
    ----------
    vector_size : int
        Size of the vector that contains all the subvectors
    vals : np.ndarray
        A concatenated vector containing the initial values for all the subvectors.
        Default values of 0.0 are concatenated for subvector dictionaries with no 'vals'
    upper : np.ndarray
        A concatenated vector containing the upper limits for all the subvectors.
        Default values of np.inf are concatenated for subvector dictionaries with no 'upper' or' equals'
    lower : np.ndarray
        A concatenated vector containing the lower limits for all the subvectors.
        Default values of -np.inf are concatenated for subvector dictionaries with no 'lower' or' equals'
    scaler : np.ndarray
        A concatenated vector containing the scalers for all the subvectors.
        Default values of 1.0 are concatenated for subvector dictionaries with no 'scaler'
    """

    def __init__(self):
        """
        Initialize a dictionary object with a default value 0 for the vector_size attribute. 
        """
        self.vector_size = 0

        self.vals = np.array([])
        self.upper = np.array([])
        self.lower = np.array([])
        self.scaler = np.array([])
        
        super().__init__()

    def __setitem__(self, key, component_dict: Dict):
        """
        Add a dictionary representing a subvector to the current dictionary of subvector dictionaries.

        Parameters
        ----------
        key : str (or any immutable object)
            Name for referencing the subvector to be added.
        component_dict : dict
            Subvector dictionary to be added in self.
        """
        if key in self:
            raise KeyError('A subvector called {} is already added before.'.format(key))
        if 'shape' not in component_dict:
            raise KeyError('Shape needs to be specified for all subvectors.')
        if not isinstance(component_dict['shape'], tuple):
            raise TypeError('Shape should be specified as a tuple') 

        size = np.prod(component_dict['shape'])
        component_dict['size'] = size
        component_dict['start_index'] = self.vector_size
        self.vector_size += size
        component_dict['end_index'] = self.vector_size

        # Additional attributes for design variables or constraints
        # Appending initial values
        if 'vals' in component_dict:
            if component_dict['vals'] is not None:
                if component_dict['vals'].shape==component_dict['shape']:
                    self.vals = np.append(self.vals, component_dict['vals'].flatten())
                else:
                    raise ValueError(f'Shape of initial values {component_dict["vals"].shape} provided'
                    f'does not match the declared shape of the subvector {component_dict["shape"]}.')
            else:
                self.vals = np.append(self.vals, np.zeros((size,)))
        else:
            self.vals = np.append(self.vals, np.zeros((size,)))

        # Appending scalers
        if 'scaler' in component_dict:
            if component_dict['scaler'] is not None:
                if np.isscalar(component_dict['scaler']):
                    self.scaler = np.append(self.scaler, component_dict['scaler'] * np.ones((size,)))
                elif component_dict['scaler'].shape==component_dict['shape']:
                    self.scaler = np.append(self.scaler, component_dict['scaler'].flatten())
                else:
                    raise ValueError(f'Shape of scalers {component_dict["scaler"].shape} provided'
                    f'does not match the declared shape of the subvector {component_dict["shape"]}.')
            else:
                self.scaler = np.append(self.scaler, np.ones((size,)))
        else:
            self.scaler = np.append(self.scaler, np.ones((size,)))
        
        # Appending upper and lower limit vectors
        # Note: Upper (and lower and equals) key MUST be defined (maybe with None value) in component_dict
        #  if the full vector (not the subvector here) is a constrained vector
        # Reason: Vectors are also used to create matrices
        if 'upper' in component_dict:
            if component_dict['equals'] is not None:
                if np.isscalar(component_dict['equals']):
                    self.lower = np.append(self.lower, component_dict['equals'] * np.ones((size,)))
                    self.upper = np.append(self.upper, component_dict['equals'] * np.ones((size,)))
                elif component_dict['equals'].shape==component_dict['shape']:
                    self.lower = np.append(self.lower, component_dict['equals'].flatten())
                    self.upper = np.append(self.upper, component_dict['equals'].flatten())
                else:
                    raise ValueError(f'Shape of equals {component_dict["equals"].shape} provided'
                    f'does not match the declared shape of the subvector {component_dict["shape"]}.')
            
            else:
                if component_dict['upper'] is not None:
                    if np.isscalar(component_dict['upper']):
                        self.upper = np.append(self.upper, component_dict['upper'] * np.ones((size,)))
                    elif component_dict['upper'].shape==component_dict['shape']:
                        self.upper = np.append(self.upper, component_dict['upper'].flatten())
                    else:
                        raise ValueError(f'Shape of upper {component_dict["upper"].shape} provided'
                        f'does not match the declared shape of the subvector {component_dict["shape"]}.')
                else:
                    self.upper = np.append(self.upper, np.full((size,), np.inf))
                
                if component_dict['lower'] is not None:
                    if np.isscalar(component_dict['lower']):
                        self.lower = np.append(self.lower, component_dict['lower'] * np.ones((size,)))
                    elif component_dict['lower'].shape==component_dict['shape']:
                        self.lower = np.append(self.lower, component_dict['lower'].flatten())
                    else:
                        raise ValueError(f'Shape of lower {component_dict["lower"].shape} provided'
                        f'does not match the declared shape of the subvector {component_dict["shape"]}.')
                else:
                    self.lower = np.append(self.lower, np.full((size,), -np.inf))

        super().__setitem__(key, component_dict)