# IPM_gpu
GPU solver for the Incompressible Porous Medium (IPM) equations using pseudo spectral method on a doubly periodic domain. The main branch solves the limit case of infinite anisotropy while the second branch is dedicated to the classical IPM equations. 

Required packages : HDF5, CUDA, FFTW, AbstractFFTs: fftfreq, rfftfreq , LinearAlgebra: mul!, ldiv!, TickTock, Random, KernelAbstractions. 
(hints: open Julia REPL, do: crtl + ] ,then write: add *package_name* ).

To use : run the Use_Me.jl file with julia on a terminal
(hint: julia -e 'include("/home/nvalade/Yulia/Digital/sqg_gpu/src/Use_Me.jl")'  ).

## Initial condition: 
+-1 on the vertical direction. The interface at the bottom/top boundaries (due to periodicity) is smoothen in order to avoid any instability there. 
The center interface is initialy straigth with random (+-1) values.
It is possible to restart a simulation from existing data (previous run). However, the new resolution $N$ and the previous $N_{old}$ have to be define such that $N=N_{old}$ or $N=2*N_{old}$

## Dissipation: 
On the base version, a diffusion ( $Eq + \nu \Delta \theta$ )is added to prevent numerical instabilities. It can be removed by changing the "reg_type" parameter to something else in Use_Me.jl. Also, hyper-diffusion can be used by changing "hp" (not working yet but will be implemented soon). 

## Output/Input:
The scalar field is saved every "nout" time step in "path" as "ffields[time].h5" in h5 format. 

## Additional remarks:
Few other options have also been implemented (forcing, lagrangian tracers, etc ...) but not the purpose here. 
Please do not hesitate to contact me if there is any question : nicolas.valade@inria.fr

