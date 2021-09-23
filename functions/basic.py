import numpy as np
import pandas as pd


# constant values
rc = 4.625
rp = 4.375


def read_file_pandas(filename):
    atoms_position = pd.read_csv(filename, sep='\s+',header=0,names=['x','y','z'])
    return atoms_position


def find_neighbours_pandas(atoms_position,rc):
    neighbours = []
    for ix,row in tqdm(enumerate(atoms_position.to_numpy())):
        this_neighbours = []
        for ix2,row2 in enumerate(atoms_position.to_numpy()):
            if (ix != ix2) & (np.sqrt(np.sum((np.abs(row-row2)**2)))<rc):
                this_neighbours.append(ix2)
        neighbours.append(this_neighbours)
    return neighbours


############################################################################################################################


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
    

def find_neighbours(nat,sx,sy,sz,x,y,z,PBC=False):
    fake_x = np.zeros(nat)
    fake_y = np.zeros(nat)
    fake_z = np.zeros(nat)
    coord_x =  np.column_stack((x,fake_y,fake_z))
    coord_y =  np.column_stack((fake_x,y,fake_z))
    coord_z =  np.column_stack((fake_x,fake_y,z))
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
    neighbours_distances = np.sqrt((x_distances**2 + y_distances**2 + z_distances**2))
    mask1 = neighbours_distances<rc
    mask2 = neighbours_distances>0
    mask = mask1 * mask2
    return mask,neighbours_distances
