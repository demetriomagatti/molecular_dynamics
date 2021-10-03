import numpy as np


# Calculating positions after a fixed timestep
def make_a_move(x,y,z,vx,vy,vz,ax,ay,az,timestep):
    '''
    arguments:
        x, y, z: numpy arrays containing x,y,z coordinates of the atomsat time = t0;
        vx, vy, vz: numpy arrays containing x,y,z velocities of the atoms at time = t0;
        ax, ay, az: numpy arrays containing x,y,z accelerations of the atoms at time = t0;
        timestep: finite time difference

    returns:
        new_x, new_y, new_z: x,y,z coordinates of the atoms at time = t0 + timestep;
    '''
    new_x = x + vx*timestep + 0.5*ax*timestep**2
    new_y = y + vy*timestep + 0.5*ay*timestep**2
    new_z = z + vz*timestep + 0.5*az*timestep**2
    return new_x,new_y,new_z


# Calculating velocities after a fixed timestep
def update_velocity(vx,vy,vz,ax,ay,az,new_ax,new_ay,new_az,timestep):
    '''
    arguments:
        vx, vy, vz: numpy arrays containing x,y,z velocities of the atoms;
        ax, ay, az: numpy arrays containing x,y,z accelerations of the atoms at time = t0;
        new_ax, new_ay, new_az: numpy arrays containing x,y,z accelerations of the atoms at time = t0 + timestep;
        timestep: finite time difference

    returns:
        new_vx, new_vy, new_vz: x,y,z coordinates of the atoms at time = t0 + timestep
    '''
    new_vx = vx + (ax + new_ax)*0.5*timestep
    new_vy = vy + (ay + new_ay)*0.5*timestep
    new_vz = vz + (az + new_az)*0.5*timestep
    return new_vx,new_vy,new_vz