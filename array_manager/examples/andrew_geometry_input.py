'''
Geometry input file
'''

import pyiges
import numpy as np

from array_manager.api import VectorComponentsDict, Vector


# geometry_dict import/read
path_name = 'CAD/'

file_name = 'test_tail_converted.IGS'
import_file_path = path_name + file_name

# print('Importing ', import_file_path)
iges = pyiges.Iges(import_file_path)
wing_surfaces = iges.bspline_surfaces()
# print(wing_surfaces[0])
# print('Number of b-spline surfaces: ', len(wing_surfaces))

# mesh = iges.bspline_surfaces(as_vtk=True, merge=True)
# mesh.plot()

# mesh = iges.to_vtk(bsplines=False, surfaces=True, merge=True, delta=0.05)
# mesh.plot(color='w', smooth_shading=True)


file_name = 'test_tail_converted.IGS'
import_file_path = path_name + file_name
iges = pyiges.read(import_file_path)
tail_surfaces = iges.bspline_surfaces()


file_name = 'test_fuse_converted.IGS'
import_file_path = path_name + file_name
iges = pyiges.read(import_file_path)
fuse_surfaces = iges.bspline_surfaces()


# file_name = 'test_disk_converted.IGS'
# import_file_path = path_name + file_name
# iges = pyiges.read(import_file_path)
# prop_surfaces = iges.bspline_surfaces()


# file_name = 'test_duct_converted.IGS'
# import_file_path = path_name + file_name
# iges = pyiges.read(import_file_path)
# torus_surfaces = iges.bspline_surfaces()


geometry_dict = VectorComponentsDict()

geo_cps = np.zeros((1,3))
wing_cps = wing_surfaces[0]._cp[0]
for i in range(len(wing_surfaces)):
    wing_surf_cps = wing_surfaces[i]._cp[0]
    for j in range(len(wing_surfaces[i]._cp)):
        if j == 0:
                pass
        else:  
            wing_surf_cps =  np.vstack((wing_surf_cps, wing_surfaces[i]._cp[j]))
    string_name = 'wing_{}'.format(i)
    geometry_dict[string_name] = dict(shape=(len(wing_surf_cps),3))
    wing_cps = np.vstack((wing_cps, wing_surf_cps))

geo_cps = np.vstack((geo_cps, wing_cps))

geo_cps = geo_cps[1:,:]

tail_cps = tail_surfaces[0]._cp[0]
for i in range(len(tail_surfaces)):
    tail_surf_cps = tail_surfaces[i]._cp[0]
    for j in range(len(tail_surfaces[i]._cp)):
        if j == 0:
                pass
        else:  
            tail_surf_cps =  np.vstack((tail_surf_cps, tail_surfaces[i]._cp[j]))
    string_name = 'tail_{}'.format(i)
    geometry_dict[string_name] = dict(shape=(len(tail_surf_cps),3))
    tail_cps = np.vstack((tail_cps, tail_surf_cps))
geo_cps = np.vstack((geo_cps, tail_cps))

fuse_cps = fuse_surfaces[0]._cp[0]
for i in range(len(fuse_surfaces)):
    fuse_surf_cps = fuse_surfaces[i]._cp[0]
    for j in range(len(fuse_surfaces[i]._cp)):
        if j == 0:
                pass
        else:  
            fuse_surf_cps =  np.vstack((fuse_surf_cps, fuse_surfaces[i]._cp[j]))
    string_name = 'fuse_{}'.format(i)
    geometry_dict[string_name] = dict(shape=(len(fuse_surf_cps),3))
    fuse_cps = np.vstack((fuse_cps, fuse_surf_cps))
geo_cps = np.vstack((geo_cps, fuse_cps))

geometry = Vector(geometry_dict)  #creates Vector

# allocating memory

geometry.allocate(setup_views=True, data=geo_cps.flatten())
print(geometry.data.shape)
print(geo_cps.shape)
# geometry.allocate(setup_views=True, data=geo_cps)    
# geometry.allocate(setup_views=True)    

# geometry['wing_1']

cutoff = 0
for i in range(len(wing_surfaces)):
    string_name = 'wing_{}'.format(i)
    geometry[string_name] = geo_cps[cutoff:cutoff+geometry[string_name].shape[0], :]
    cutoff += len(geometry[string_name])

for i in range(len(tail_surfaces)):
    string_name = 'tail_{}'.format(i)
    geometry[string_name] = geo_cps[cutoff:cutoff+len(geometry[string_name]), :]
    cutoff += len(geometry[string_name])

for i in range(len(fuse_surfaces)):
    string_name = 'fuse_{}'.format(i)
    geometry[string_name] = geo_cps[cutoff:cutoff+len(geometry[string_name]), :]
    cutoff += len(geometry[string_name])


print('initialized_cps: ', geometry.data)

# ERROR: Do not make modifications to data, data is only for accessing the vector
# geometry.data  = geometry.data*2  # data points to a different memory location. Here, geometry.data is reassigned

geometry.set_data(geometry.get_data() * 2)

# Following options will all work, but the one above leads to errors

# geometry.data[:]  = geometry.data*2

# geometry.data  *= 2           # data points to the same memory location

# geometry.allocate(setup_views=True, data=geometry.data*2)

# geometry *= 2                   # recommended way: treat Vector objects like numpy vectors

# geometry = geometry * 2       # recommended way: treat Vector objects like numpy vectors


print('scaled cps', geometry.data)
geometry['wing_0'] = geometry['wing_0']*-10000
print('geo_cps', geo_cps)
print('shifted wing_0', geometry['wing_0'])
print('geo', geometry.data)
print(np.min(geometry.data))

# surfaces = np.array([wing_surfaces, tail_surfaces, fuse_surfaces, prop_surfaces, torus_surfaces])

# geometry_dict parameterization settings
#   num_cntrl_pts (default?) (maybe a density?)
# cp_dens_default = 50
# allow specification of cp or cp_dens based on type of primitive
cp_dens_lift_surf = 200
cp_dens_body = 50
cp_dens_rotor_duct = 50

#   reference_axis_settings
# ref_axis_loc_default = 0.25     # quarter chord
ref_axis_loc_lift_surf = 0.25     # quarter chord
ref_axis_loc_body = 0.25     # quarter chord
ref_axis_loc_rotor_duct = 0.25     # quarter chord

geo_settings = np.array([cp_dens_lift_surf, cp_dens_body, cp_dens_rotor_duct, ref_axis_loc_lift_surf, ref_axis_loc_body, ref_axis_loc_rotor_duct])