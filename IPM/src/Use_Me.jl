#                          User's file : Yulia version
#   It : 
#     - sets the parameters
#     - runs the code 
#
# See "ActiveScalar_MPI.jl" for the description of what the code does.
# Mono CPU/GPU
using CUDA
TA      =  CuArray     # Mono CPU -> Array  / GPU -> CuArray


# Numerical Parameters 
N       = (512,2048)      # grid resolution    
Ls      = (2π, 8π)     # Physical grid       
dt      = 1e-3         # maximal time step 
Ccfl    = 0.3          # CFL based adaptative time step

nout    = 100         # saving interval
rep_max = Int(1e7)    # maximal steps number
tmax    = 30          # maximal time 

# Regularisation : Laplacian / Hidden-Symetric 
reg_type = "Laplacian" # "Laplacian"
nu       =  2e-2       # Kinematic viscosity
hp       =  1.         # Hyperviscosity order


# Files and folders
code_rep      = ""
path          = "/workspace/sqg/Yulia_and_Sergey/Anisotrop_N_512_2048/"     # WRITE YOUR REPOSITORY HERE
ffields       = "scalar_t"  # files name with eulerian data
flag          = "tracers"   # files name with lagrangian ones

### Initial scalar field
restart  = false # true/false
t0       = 0.0 

# if restart == true : old file // WARNING : can only be restarted if Nold = N or N/2
old_file  = "/workspace/sqg/Yulia/test8_32/scalar_t0.524798.h5"
prm_IC    = (; o_file  = old_file,
               o_resol = (512,2048), # old file resolution
            )


# else if : Initial condition definition 
# Digital instability : +- 1 on vertical direction

function f_init!(thetas,thetahs,plan,x,y,FM,cste_init,TA,prm)
  #+- 1 
  @. thetas = sign(y' - (4pi+4pi/N[2])) 

  # We need to smooth the top and bottom boundaries (periodic) to only focus on the middle interface  
  y_0_L_smoothing_size = Int(round(N[2]/128))
  reg = zeros(N[1],N[2])
  reg[1:end,1:y_0_L_smoothing_size] .= (atan.((1 .- (1:y_0_L_smoothing_size)) ./ (y_0_L_smoothing_size .- (1:y_0_L_smoothing_size)))/(pi/2) .+ 1)'
  reg[1:end,end-y_0_L_smoothing_size+1:end] .= (atan.(abs.(1 .- (1:y_0_L_smoothing_size)) ./ (y_0_L_smoothing_size .- (1:y_0_L_smoothing_size)))/(pi/2) .- 1)'  
  reg = prm.TA(reg)
  @. thetas += reg
  reg = nothing

  # Perturbation of the center interface
  for i in 1:N[1]
      CUDA.@allowscalar thetas[i,Int(N[2]/2)] = rand((-1,1))
  end

  # Load the field in Fourier
  mul!(thetahs,plan,thetas)
end


# parameters used for the initial condition  (for now useless)
prm_IC  = (; type    = "Digital Instability",
             cte_IC  = (;)
          )

# Forcing (here useless)
forcing = (; type = "TypeI", # Exemple : White in time, Amplitude F,
                             #           Gaussian over the shells in Fourier space,
                             #           centered in |k|=kf, width sigma,
                             #           + Linear Damping (alpha) for |k|<k_d
             prmf = (; kf    = 15  ,
                       F     = 0   , # <- 0 == no forcing
                       sigma = 1   ,
                       k_d   = 2   ,
                       alpha = 0   , # <- 0 == no dissipation
                    )
           )

# Particules (here useless)
part     = (; n_part   = Int(0),
              meth     = "cubic",
              type     = "multi points",
              restart  = false,
           )


# Summ up of the parameters
params = (; Ls      = Ls,
            N       = N,
            dt      = dt,
            r_type  = reg_type, 
            nu      = nu,  
            hp      = hp,
            kc      = kc,
            part    = part,
            rep_max = rep_max, 
            tmax    = tmax,   
            path    = path,                   
            ffields = ffields,     
            flag    = flag,
            Ccfl    = Ccfl,
            nout    = nout,
            restart = restart,
	    t0      = t0,
            prm_IC  = prm_IC,
            forc_IC = forcing,
            TA      = TA,
        ) 

# import all the functions  
include(code_rep*"ActiveScalar.jl")

# --------------------------------------------------------------------------
#                           ! RUN !
# --------------------------------------------------------------------------
t,rep  =  ActiveScalar(params)
println("FINISHED : rep = $rep and time = $t ")
