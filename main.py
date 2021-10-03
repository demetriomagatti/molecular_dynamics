import numpy as np
from functions import basic 
from functions import interaction
from functions import evolution
from tqdm.auto import tqdm


# constant values
kb = 1/11603
m_ag = 108*1.66e-27/16


# Executing a simulation
def make_simulation(filename,T,timelength,timestep,PBC=False,approx=False):
    '''
    executes a full simulation
    '''
    lattice = basic.read_file(filename)
    n_atoms,sx,sy,sz,x,y,z = lattice
    mask, distance = basic.find_neighbours(*lattice, PBC=PBC)
    vx,vy,vz = basic.initialize_speed(n_atoms,x,y,z,T,remove_translation=True)
    if approx:
        Fx,Fy,Fz = interaction.calc_force_approx(*lattice,distance,PBC=PBC)
    else:
        Fx,Fy,Fz = interaction.calc_force(*lattice,distance,PBC=PBC)
    ax,ay,az = interaction.calc_acceleration(Fx,Fy,Fz)
    v2 = vx**2 + vy**2  + vz**2
    Ekin = 0.5*m_ag*np.sum(v2)
    Tkin = 2*Ekin/(3*n_atoms*kb)  
    print('Translation removal check:')
    print(f'    mean vx: {np.mean(vx)}')
    print(f'    mean vy: {np.mean(vy)}')
    print(f'    mean vz: {np.mean(vz)}\n')
    print(f'Set temperature: {T:.2f}K; real temperature: {Tkin:.2f}K \n')
    steps = int(timelength/timestep)
    time_array = np.arange(timestep,timelength,timestep)
    # output arrays
    Temp_array = []
    energy_array = []
    all_x = []
    all_y = []
    all_z = []
    all_vz = []
    all_az = []
    # cicling
    for i in tqdm(range(0,steps)):
        new_x,new_y,new_z = evolution.make_a_move(x,y,z,vx,vy,vz,ax,ay,az,timestep)
        lattice = n_atoms,sx,sy,sz,new_x,new_y,new_z
        mask, distance = basic.find_neighbours(*lattice, PBC=PBC)
        if approx:
            new_Fx,new_Fy,new_Fz = interaction.calc_force_approx(n_atoms,sx,sy,sz,new_x,new_y,new_z,distance,PBC=PBC)
            E_pot = interaction.lennard_jones_approx(distance)
        else:
            new_Fx,new_Fy,new_Fz = interaction.calc_force(n_atoms,sx,sy,sz,new_x,new_y,new_z,distance,PBC=PBC)
            E_pot = interaction.lennard_jones(distance)
        new_ax,new_ay,new_az = interaction.calc_acceleration(new_Fx,new_Fy,new_Fz)
        new_vx,new_vy,new_vz = evolution.update_velocity(vx,vy,vz,ax,ay,az,new_ax,new_ay,new_az,timestep)
        v2 = new_vx**2 + new_vy**2  + new_vz**2
        E_kin = 0.5*m_ag*np.sum(v2)
        T_kin = 2*E_kin/(3*n_atoms*kb)  
        E_tot = E_kin + E_pot
        ###
        Temp_array.append(T_kin)
        energy_array.append(E_tot)
        ###
        x,y,z = new_x,new_y,new_z
        vx,vy,vz = new_vx,new_vy,new_vz
        ax,ay,az = new_ax,new_ay,new_az
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
    Temp_array = np.asarray(Temp_array)
    energy_array = np.asarray(energy_array)
    all_x = np.transpose(all_x)
    all_y = np.transpose(all_y)
    all_z = np.transpose(all_z)
    return time_array,all_x,all_y,all_z,Temp_array,energy_array