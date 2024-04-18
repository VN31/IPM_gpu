#######################################################
######### IO and fields creation functions ############

#   I - Field saving functions
#  II - Outputs saving
# III - Restart with higher resolution


### I - Save 

# fields
function save_vorticity(name,thetas)
    h5open(name*".h5", "w") do ff
        ff["theta"] = Array(thetas)
    end
end

# Lagrangian
function save_Lagrangian_multi(name,particles,t)
    h5open(name*".h5", "cw") do ff
        println("time save multi lagrangian : $t")
        # absolute time 
	    ff["t"] = t

	        # positions 
        ff["X"] = Array(particles[1])
        ff["Y"] = Array(particles[2])

        # velocity
        ff["Ux"] =  Array(particles[4][1])
        ff["Uy"]     = Array(particles[6])
	
        # relative time
        ff["tau_theta"] = Array(particles[5])
        ff["eps_theta"] =  Array(particles[3][1])
    end
end

# ------------------------------------------------------------------

### II - Output

function output!(t,thetahs,thetas,p)

    ldiv!(thetas, p.plan, thetahs)
    fname = p.path*p.ffields*"$(round(t,digits=6))"
    save_vorticity(fname,thetas)

end


# --------------------------------------------------------------

### III - restarting from previous and less resolved fields
function init_Eulerian!(theta,thetah,prm,FM)
    if prm.restart == false 
        new_fields!(theta,thetah,prm,FM)
    else
        restart!(theta,thetah,prm)
    end 
end

# Open a .h5 for file and load theta
function open_vorticity!(name,thetas,TA)
    h5open(name,"r") do ff
        A = TA(read(ff["theta"]))
        @. thetas = A 
    end
end

# open an existing .h5 file with a (N,N) or (2N,2N) theta Field
# and instantiate the (2N,2N) thetas and thetahs pencil arrays
function restart!(thetas,thetahs,params)
    (; plan , prm_IC,Ns)  = params
    (;o_file,o_resol)   = prm_IC

    if o_resol == Ns
    println("meme dimension")
        open_vorticity!(o_file,thetas,params.TA)
        mul!(thetahs,plan,thetas)
	
    elseif 2 .* o_resol == Ns
        old_theta  = params.TA(zeros(Float64,o_resol))
        old_plan   = plan_rfft(old_theta)
        old_thetah = old_plan * old_theta

        open_vorticity!(o_file,old_theta,params.TA)
	    mul!(old_thetah,old_plan,old_theta)

        CUDA.@allowscalar  thetahs[1:Int(Ns[1]/4)+1,1:Int(Ns[2]/4)]          .= old_thetah[1:end,1:Int(Ns[2]/4)]
	    CUDA.@allowscalar  thetahs[1:Int(Ns[1]/4)+1,Int(3/4*Ns[2]+1):Int(N)] .= old_thetah[1:end,Int(Ns[2]/4)+1:Int(Ns[2]/2)]
        @. thetahs *= 4  # *4 for the fourier scale factor 

	    ldiv!(thetas,plan,thetahs)	      
    else
        error(1)
    end
    println("Restarted from "*old_file)
end

# Directly instantiate the initial fields 
function new_fields!(thetas,thetahs,params,FM)
    (;plan ,plan, grid,prm_IC,TA) = params

    f_init!(thetas,thetahs,plan,grid[1],grid[2],FM,prm_IC.cte_IC,TA,params)

    println(prm_IC.type)
    println("Theta : $(summary(thetas))")
end
