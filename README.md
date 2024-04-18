# IPM_gpu
GPU solver for the Incompressible Porous Medium (IPM) equations using pseudo spectral method on a doubly periodic domain. 

The main branch solves the limit case of infinite anisotropy while the second branch is dedicated to the classical IPM equations. 

## Initial condition : 
+-1 on the vertical direction. The interface at the bottom/top boundaries (due to periodicity) is smoothen in order to avoid any instability there. 
The center interface is initialy straigth with random (+-1) values.

## Dissipation : 
On the base version, a diffusion ($\nu \Delta \theta$)is added to prevent numerical instabilities. It can be removed by changing the "reg_type" parameter to something else in Use_Me.jl. Also, hyper-diffusion can be used by changing "hp" (not working yet but will be implemented soon). 

## Output :
The scalar field is saved every "nout" time step. 

## Remarks :
Few other options have also been implemented (forcing, lagrangian tracers, etc ...) but not the purpose here. 
Please do not hesitate to contact me if there is any question : nicolas.valade@inria.fr

