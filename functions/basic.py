import numpy as np
import pandas as pd


# constant values
rc = 4.625
rp = 4.375


'''
First idea: read file and save results in pandas dataframe. Index = atom index, three columns for x, y, z coordinates.
Way too slow when it comes to finding neighbours, as an iteration over rows becomes necessary.
'''


def read_file_pandas(filename):
    atoms_position = pd.read_csv(filename, sep='\s+',header=0,names=['x','y','z'])
    return atoms_position


def find_neighbours_pandas(atoms_position,rc):
    neighbours = []
    for ix,row in enumerate(atoms_position.to_numpy()):
        this_neighbours = []
        for ix2,row2 in enumerate(atoms_position.to_numpy()):
            if (ix != ix2) & (np.sqrt(np.sum((np.abs(row-row2)**2)))<rc):
                this_neighbours.append(ix2)
        neighbours.append(this_neighbours)
    return neighbours


'''
Numpy approach. A bit tricky to understand, but way faster. On the same PC:
    Pandas approach computing time (find_neighbours): approx 0.8 seconds on average
    Numpy approach computing time (find_neighbours): approx 0.015 seconds on average (53x faster)
'''


def read_file(filename):
    x,y,z = np.genfromtxt(filename, unpack=True)
    sx = x[0]
    sy = y[0]
    sz = z[0]
    x = x[1:]
    y = y[1:]
    z = z[1:]
    n_atoms = len(x)
    return n_atoms,sx,sy,sz,x,y,z
    

def find_neighbours(n_atoms,sx,sy,sz,x,y,z,PBC=False):
    coord_x =  np.column_stack((x,np.zeros(n_atoms),np.zeros(n_atoms)))
    coord_y =  np.column_stack((np.zeros(n_atoms),y,np.zeros(n_atoms)))
    coord_z =  np.column_stack((np.zeros(n_atoms),np.zeros(n_atoms),z))
    x_distances = np.sum(coord_x[:,None,:] - coord_x, axis=-1)
    y_distances = np.sum(coord_y[:,None,:] - coord_y, axis=-1)        
    z_distances = np.sum(coord_z[:,None,:] - coord_z, axis=-1)    
    if PBC:
        mask_x_plus = x_distances>(0.5*sx)
        mask_y_plus = y_distances>(0.5*sy)
        mask_x_minus = x_distances<(-0.5*sx)
        mask_y_minus = y_distances<(-0.5*sy)
        x_distances = x_distances - sx*mask_x_plus + sx*mask_x_minus
        y_distances = y_distances - sy*mask_y_plus + sy*mask_y_minus
    distances = np.sqrt((x_distances**2 + y_distances**2 + z_distances**2))
    mask1 = distances<rc
    mask2 = distances>0
    mask = mask1 * mask2
    return mask,distances
