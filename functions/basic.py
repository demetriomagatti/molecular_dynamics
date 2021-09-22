import numpy as np
import pandas as pd


def read_file(filename):
    atoms_position = pd.read_csv(filename, sep='\s+',header=0,names=['x','y','z'])
    return atoms_position


def find_neighbours(atom_position):
    return 