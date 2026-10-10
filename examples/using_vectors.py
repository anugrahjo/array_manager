'''
Example file for Vector class
'''

from array_manager.api import VectorComponentsDict, Vector

import numpy as np

vect1_dict = VectorComponentsDict()

nx = 1
ny = 1

# Naming dictionary
vect1_dict['wing'] = dict(shape=(nx*ny,))
vect1_dict['tail'] = dict(shape=(nx*ny,))
vect1_dict['fuse'] = dict(shape=(nx*ny,))

vect1 = Vector(vect1_dict)  #creates Vector

# allocating memory
vect1.allocate(setup_views=True)    
# Note: need setup_views=True if not passing data to a subvector
# Note: if all data is already available, can use vect1.allocate(data=known_data)

# assign values
my_array = np.array([3])
vect1['wing'] = my_array

# Creating another Vector with the same structure
vect2 = Vector(vect1_dict)

vect2.allocate(setup_views=True)    

my_array = np.array([2])
vect2['fuse'] = my_array

print('vect1: ', vect1.data, 'vect2: ', vect2.data)