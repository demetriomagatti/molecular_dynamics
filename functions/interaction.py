import numpy as np
from scipy.spatial.distance import cdist
import scipy
import pylab as pp


# constant values 
rc = 4.625
rp = 4.375
sigma = 2.644
epsilon = 0.345
kb = 1/11603
Temp = 100
m_ag = 108*1.66e-27/16


# Lennard Jones poly7 approximation coefficients
Apoli7 = (1/((rc - rp)**7*rp**12))*4*epsilon*rc**4*sigma**6*(2* rp**6 *(-42* rc**3 + 182* rc**2* rp - 273* rc* rp**2 + 143* rp**3) + (455* rc**3 - 1729* rc**2* rp + 2223* rc* rp**2 - 969* rp**3)*sigma**6)
Bpoli7 = (1/((rc - rp)**7* rp**13))*16* epsilon* rc**3* sigma**6*(rp**6* (54* rc**4 - 154* rc**3* rp + 351* rc *rp**3 - 286* rp**4) + (-315* rc**4 + 749 *rc**3 * rp + 171 *rc**2* rp**2 - 1539* rc* rp**3 + 969* rp**4)* sigma**6)
Cpoli7 = (1/((rc - rp)**7* rp**14))*12* epsilon* rc**2* sigma**6* (rp**6* (-63* rc**5 - 7* rc**4 *rp + 665* rc**3 *rp**2 - 975* rc**2* rp**3 - 52* rc* rp**4 + 572* rp**5) +  2 *(195* rc**5 + 91* rc**4* rp - 1781* rc**3* rp**2 + 1995 *rc**2* rp**3 + 399* rc* rp**4 - 969* rp**5)* sigma**6)
Dpoli7 = (1/((rc - rp)**7* rp**15))*16* epsilon* sigma**6*(rc* rp**6* (14* rc**6 + 126* rc**5* rp - 420* rc**4* rp**2 - 90* rc**3* rp**3 + 1105* rc**2* rp**4 - 624* rc* rp**5 - 286 *rp**6) + rc* (-91* rc**6 - 819* rc**5* rp + 2145 * rc**4 * rp**2 + 1125* rc**3* rp**3 - 5035* rc**2* rp**4 + 1881* rc* rp**5 + 969* rp**6)* sigma**6)
Epoli7 = (1/((rc - rp)**7* rp**15))*4* epsilon* sigma**6*(2* rp**6* (-112* rc**6 - 63* rc**5* rp + 1305* rc**4* rp**2 - 1625* rc**3* rp**3 - 585* rc**2* rp**4 +  1287 *rc* rp**5 + 143* rp**6) + (1456*rc**6 +1404*rc**5* rp - 14580 *rc**4* rp**2 + 13015* rc**3* rp**3 + 7695* rc**2* rp**4 - 8721 *rc* rp**5 - 969* rp**6)* sigma**6)
Fpoli7 = (1/((rc - rp)**7* rp**15))*48* epsilon* sigma**6*(-rp**6* (-28* rc**5 + 63* rc**4* rp + 65* rc**3* rp**2 - 247* rc**2* rp**3 + 117* rc* rp**4 + 65* rp**5) + (-182* rc**5 + 312* rc**4* rp + 475* rc**3* rp**2 - 1140* rc**2* rp**3 + 342* rc* rp**4 + 228* rp**5)* sigma**6)
Gpoli7 = (1/((rc - rp)**7* rp**15))*4* epsilon* sigma**6* (rp**6* (-224* rc**4 + 819* rc**3* rp - 741* rc**2* rp**2 - 429* rc* rp**3 + 715* rp**4) + 2 *(728* rc**4 - 2223* rc**3* rp + 1425* rc**2* rp**2 + 1292* rc* rp**3 - 1292* rp**4)* sigma**6)
Hpoli7 = (1/((rc - rp)**7* rp**15))*16* epsilon* sigma**6* (rp**6*(14* rc**3 - 63* rc**2* rp + 99* rc* rp**2 - 55* rp**3) + (-91* rc**3 + 351* rc**2* rp - 459* rc* rp**2 + 204* rp**3)* sigma**6)


# Total potential energy (approx)
def lennard_jones_approx(distances):
    '''
    returns total potential energy value;
    approximation: 
        true LJ for distance<rp
        poly7 function for rp<distance<rc
        zero interaction for distance>rc

    arguments:
        distances: (n_atoms,n_atoms) ndarray; j-th position in i-th row contains the distances between atoms i and j
    '''
    mask_rp = distances<rp
    mask_rc = distances<rc
    mask_rc = mask_rc ^ mask_rp
    distances[np.where(distances==0)] = np.infty
    Epot_rp = 4*epsilon*((sigma/(mask_rp*distances))**12 - (sigma/(mask_rp*distances))**6)
    Epot_rc = Apoli7*mask_rc + Bpoli7*mask_rc*distances + Cpoli7*((mask_rc*distances)**2) + Dpoli7*((mask_rc*distances)**3) + Epoli7*((mask_rc*distances)**4) + Fpoli7*((mask_rc*distances)**5) + Gpoli7*((mask_rc*distances)**6) + Hpoli7*((mask_rc*distances)**7)
    return 0.5*(np.nansum(Epot_rp) + np.nansum(Epot_rc))


# Total potential energy
def lennard_jones(distances):
    '''
    returns total potential energy value

    arguments:
        distances: (n_atoms,n_atoms) ndarray; j-th position in i-th row contains the distances between atoms i and j
    '''
    distances[np.where(distances==0)] = np.infty
    Epot = 4*epsilon*((sigma/(distances))**12 - (sigma/(distances))**6)
    return 0.5*np.nansum(Epot)


# Force experienced by each atom (approx)
def calc_force_approx(n_atoms,sx,sy,sz,x,y,z,distances,PBC=False):
    '''
    arguments:
        n_atoms(int): number of atoms in the lattice;
        sx, sy, sz: lattice span in x,y,z dimension;
        x, y, z: numpy arrays containing x,y,z coordinates of the atoms;
        distances: (n_atoms,n_atoms) ndarray; j-th position in i-th row contains the distances between atoms i and j;
        PBC: boolean flag to select whether to active Periodic Boundary Conditions or not; default: False

    returns:
        Fx,Fy,Fz: n_atoms arrays with x,y,z components of the force experienced by each atom. Fx[i],Fy[i],Fz[i] describes the total force felt by i-th atom
    '''
    mask_rp = distances<rp
    mask_rc = distances<rc
    mask_rc = mask_rc ^ mask_rp
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
    distances[np.where(distances==0)] = np.infty
    Fx_rp = 24*epsilon*( 2*sigma**12*(mask_rp*distances)**(-14) - sigma**6*(mask_rp*distances)**(-8) )*(mask_rp*x_distances)
    Fx_rc = - 2*(mask_rc*x_distances)*( 0.5*Bpoli7/(mask_rc*distances) + Cpoli7 + 1.5*Dpoli7*(mask_rc*distances) + 2*Epoli7*(mask_rc*distances)**2 + 2.5*Fpoli7*(mask_rc*distances)**3 + 3*Gpoli7*(mask_rc*distances)**4 + 3.5*Hpoli7*(mask_rc*distances)**5)
    Fx_rp = np.nansum(Fx_rp,axis=1)
    Fx_rc = np.nansum(Fx_rc,axis=1)
    Fx = Fx_rp + Fx_rc
    Fy_rp = 24*epsilon*( 2*sigma**12*(mask_rp*distances)**(-14) - sigma**6*(mask_rp*distances)**(-8) )*(mask_rp*y_distances)
    Fy_rc = - 2*(mask_rc*y_distances)*( 0.5*Bpoli7/(mask_rc*distances) + Cpoli7 + 1.5*Dpoli7*(mask_rc*distances) + 2*Epoli7*(mask_rc*distances)**2 + 2.5*Fpoli7*(mask_rc*distances)**3 + 3*Gpoli7*(mask_rc*distances)**4 + 3.5*Hpoli7*(mask_rc*distances)**5)
    Fy_rp = np.nansum(Fy_rp,axis=1)
    Fy_rc = np.nansum(Fy_rc,axis=1)
    Fy = Fy_rp + Fy_rc
    Fz_rp = 24*epsilon*( 2*sigma**12*(mask_rp*distances)**(-14) - sigma**6*(mask_rp*distances)**(-8) )*(mask_rp*z_distances)    
    Fz_rc = - 2*(mask_rc*z_distances)*( 0.5*Bpoli7/(mask_rc*distances) + Cpoli7 + 1.5*Dpoli7*(mask_rc*distances) + 2*Epoli7*(mask_rc*distances)**2 + 2.5*Fpoli7*(mask_rc*distances)**3 + 3*Gpoli7*(mask_rc*distances)**4 + 3.5*Hpoli7*(mask_rc*distances)**5)  
    Fz_rp = np.nansum(Fz_rp,axis=1)
    Fz_rc = np.nansum(Fz_rc,axis=1)
    Fz = Fz_rp + Fz_rc
    return Fx,Fy,Fz
    

# Force experienced by each atom
def calc_force(n_atoms,sx,sy,sz,x,y,z,distances,PBC=False):
    '''
    arguments:
        n_atoms(int): number of atoms in the lattice;
        sx, sy, sz: lattice span in x,y,z dimension;
        x, y, z: numpy arrays containing x,y,z coordinates of the atoms;
        distances: (n_atoms,n_atoms) ndarray; j-th position in i-th row contains the distances between atoms i and j;
        PBC: boolean flag to select whether to active Periodic Boundary Conditions or not; default: False

    returns:
        Fx,Fy,Fz: n_atoms arrays with x,y,z components of the force experienced by each atom. Fx[i],Fy[i],Fz[i] describes the total force felt by i-th atom
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
    distances[np.where(distances==0)] = np.infty
    Fx = 24*epsilon*( 2*sigma**12*(distances)**(-14) - sigma**6*(distances)**(-8) )*(x_distances)
    Fx = np.nansum(Fx,axis=1)
    Fy = 24*epsilon*( 2*sigma**12*(distances)**(-14) - sigma**6*(distances)**(-8) )*(y_distances)
    Fy = np.nansum(Fy,axis=1)
    Fz = 24*epsilon*( 2*sigma**12*(distances)**(-14) - sigma**6*(distances)**(-8) )*(z_distances)    
    Fz = np.nansum(Fz,axis=1)
    return Fx,Fy,Fz
