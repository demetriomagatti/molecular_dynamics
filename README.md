# Molecular dynamics
Simulation of the behaviour of a crystal lattice in different temperature conditions. Brief description following.

## Project steps
<ul>
<li>Definition of initial conditions
<li>Definition of finite difference method for time evolution of the system
<li>Definition of atom-atom interaction in the lattice
<li>Introduction of periodic boundary conditions
</ul>

## Files and structure
<ul>
<li>fcc100a256.txt: source text file containing perfect crystal lattice position for a 256 atoms fcc-crystal. row 0: lattice dimension; rows 1-257: atoms positions; everything is expressed in nanometers
<li>functions: folder containing source files
<li>results: folder with simulation results in .xlsx extension
<li>config.json: json file to define conditions for different simulations in a row
<li>molecular_dynamics.ipynb: jupyter notebook for code execution
</ul>

## Notes on the project and on the jupyter notebook
<ul>
<li>This project is a re-writing of some bad-shaped code I wrote for a master's course. The exam had several requests which are not fullfilled here, but the existing code should provide the means to provide all the required answers. My intent was actually to kill some free time, re-write some code in a better shape and starting to fill my git repository list, so it's not in my interest to do all required simulations and provide results
<li>The jupyter notebook can be exectued cell-by-cell to get a graps of what different implemented functions do; executing the first three cells (basic imports, a couple of tweaks) and the three ones in the "More simulations in a row, saving results" section actually allows to run simulations without the need to execute code cells inbetween; new configs can be added in the configs.json file
<li>The seems to be some problem with the rendering of the molecular_dynamics.ipynb file; I uploaded an html version, which should be easier to download/view
</ul>