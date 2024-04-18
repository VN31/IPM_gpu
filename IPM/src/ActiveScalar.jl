#######################################################################
#################### 2D Active scalar Solver  #########################
#######################################################################

# standard pseudo-spectral solver using the PencilFFTs.jl 
#            library (ref. Juan Ignacio Polanco).

# This branch is made to study SQG's statisticaly steady state. 
# The code runs  on mono-GPU.
# The time is adapted upon the CFL condition.
# Passive tracers are also available.

# Remarks : 1 - For performance reasons, the vorticity field
#          is the only output when GPU is used. Otherwise, energy
#          spectrum and macroscopic quantities are also computed.
#           2 - Fields are saved in .h5 files except if you run the
#          code on the NEF cluster. Then it is .jld2
 
using HDF5, CUDA
using FFTW
using AbstractFFTs: fftfreq, rfftfreq
using LinearAlgebra: mul!, ldiv!
using TickTock
using Random
using KernelAbstractions

include("Equations.jl")    # Generalized SQG equations
include("Time_Stepper.jl") # RK stepper's methods and adaptative time step
include("IO.jl")           # Save scalar, spectrum and 2D fields
include("utilities.jl")    # grids, and fields creation functions
include("Tracers.jl")      # Passive tracers dynamics with MPI and GPU

if Base.find_package("CUDA") !== nothing
    using CUDA
    using CUDA.CUDAKernels
    const backend = CUDABackend()
    CUDA.allowscalar(false)
else
    const backend = CPU()
end
# ------------------------------------------------------------
# Main functions

# ActiveScalar() : 
#    - Solve the SQ-g active scalar equation (see Equation.jl)
function ActiveScalar(params)
    t, rep    = params.t0, 0
    N         = params.N  
    t,rep     = resolving(N,t,rep,params)    
    return t,rep
end
 
# resolving() : - Create the fields and run the simulation.  
function resolving(N,t,rep,prm)

    # -----------------------------------------------
    # partitionning         
    Ns  = N # (N,N)
    Ls  = prm.Ls

    # Active Scalar
    theta  = prm.TA(zeros(Float64,Ns[1],Ns[2])) 

    # FFT plan : Real Fourier transform
    plan   = plan_rfft(theta)
    thetah = plan * theta

    # grids (physical and Fourier) 
    grid              = real_grids(Ns,Ls)
    grid_fourier, ks  = spec_grids(Ns,Ls)

    # usefull constants
    prm = (;prm..., Ns           = Ns,   
                    ks           = ks, 
                    plan         = plan,
                    grid         = grid ,
                    grid_fourier = grid_fourier,
                    scalefactor  = Ns[1]^Ns[2]
            ) 

    #----------------------------------------------------
    # Initialising

    # Lagrangian problem
    prm, particles = init_Lagrangian(prm)

    # fix some constant for the forcing : See utilities.jl
    prm = init_forcing(prm,thetah)
    
    #intermediate fields for pseudo-spectral methods
    rhs_RK4          = intermediate_RK4(thetah)
    w,wh             = inter_fields(plan,theta)
    Fourier_Matrices = matrices(prm)

    # Intial Eulerian fields : see IO.jl
    init_Eulerian!(theta,thetah,prm,Fourier_Matrices)

    # Initial Diagnostics  
    output!(t,thetah,theta,prm) 
 
    # --------------------------------------------------------------------
    # Running

    # solving until exit condition 
    t,rep = run(thetah,theta,particles,rhs_RK4,t,rep,prm,w,wh,Fourier_Matrices)

    return t,rep     
end

# run() : 
#     - Time stepping until the end.
#     - See Time_Stepper.jl for the exit loop condition.
#     - Outputs are made each params.nout steps.
function run(thetahs,thetas,particles,rhs_RK4,t,rep,param,w,wh,F_M)
    dt = param.dt
    RUN = true
    
    while RUN
        tick()
        t, rep = advance!(thetahs,thetas,particles,rhs_RK4,t,rep,dt,param,w,wh,F_M)
        dt     = minimum([param.dt,timestep(thetahs,w,wh,F_M[3],F_M[4],param)]) 
        @show dt
	RUN    = exit_loop(rep,t,thetahs,param)		    
        println(t)
        output!(t,thetahs,thetas,param) 
        tock()
    end
    
    println("Time loop's exit : t=$t")
    return t,rep
end

# advance!() :
# Time advancement beetween two outputs.
# Different scheme can be used (see Time_Stepper.jl)
#   -> change the intermediate fields accordingly
function advance!(thetahs,thetas,particles,rhs_RK4,t,rep,dt,params,w,wh,F_M)
    (;nout) = params

    for i in 1:nout
        dt = minimum([params.dt,timestep(thetahs,w,wh,F_M[3],F_M[4],params)])
        if particles === nothing
            t,rep = EDTRK4!(thetahs,thetas,dt,rhs_RK4,w,wh,t,rep,rhs_inv!,an_visco!,F_M,params)
        else 
            t,rep = L_EDTRK4!(thetahs,thetas,dt,rhs_RK4,particles,w,wh,t,rep,rhs_inv!,an_visco!,F_M,params)
       	        if mod(rep,100)==0
		            lname = lagrangian_output_multi!(particles,params,t,w,wh,thetahs,F_M,0)
          		        save_Lagrangian_multi(lname,particles,t)
            end
    	    end
        #Forcing!(thetahs,dt,params,F_M)
    end 
    return t,rep
end 

# Forcing!() : - Large scale forcing
#              - See Equations.jl for the functions 
function Forcing!(thetahs,dt,params,F_M)
    (; type, prmf) = params.forcing
    if type =="TypeI"
        gaussian_forcing!(thetahs,dt,prmf,params.scalefactor,F_M[1],F_M[2],N)
        linear_damping!(thetahs,dt,prmf)
    end

end
