import numpy as np
import pandas as pd


# constant values
rc = 4.625
rp = 4.375
kb = 1/11603
m_ag = 108*1.66e-27/16

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
    '''
    returns a boolean mask to tell which atoms can be consideres as "neighbours" wrt to rp and rc values and distances between atoms in the lattice;

    arguments:
        n_atoms(int): number of atoms in the lattice;
        sx, sy, sz: lattice span in x,y,z dimension;
        x, y, z: numpy arrays containing x,y,z coordinates of the atoms
        PBC: boolean flag to select whether to active Periodic Boundary Conditions or not; default: False

    returns:
        mask: (n_atoms,n_atoms) ndarray; j-th position in i-th tells if atoms i and j are considered as neighbours
        distances: (n_atoms,n_atoms) ndarray; j-th position in i-th row contains the distances between atoms i and j
    '''
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


def initialize_speed(n_atoms,x,y,z,T,remove_translation=True):
    c = np.sqrt(3*kb*T/m_ag)
    vx = c*(2*np.random.rand(n_atoms))
    vy = c*(2*np.random.rand(n_atoms))
    vz = c*(2*np.random.rand(n_atoms))
    vx[n_atoms-1] = 0
    vy[n_atoms-1] = 0
    vz[n_atoms-1] = 0
    if remove_translation:
        vx = np.asarray([v - np.mean(vx) for v in vx])
        vy = np.asarray([v - np.mean(vy) for v in vy])
        vz = np.asarray([v - np.mean(vz) for v in vz])
    v2 = vx**2 + vy**2  + vz**2
    Ekin = 0.5*m_ag*np.sum(v2)
    Tkin = 2*Ekin/(3*n_atoms*kb)     
    vx = vx*np.sqrt(T/Tkin)
    vy = vy*np.sqrt(T/Tkin)
    vz = vz*np.sqrt(T/Tkin)
    return vx,vy,vz
