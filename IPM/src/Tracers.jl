#######################################################
############# Passive Tracers dynamics ################

include("Interpolate.jl")

#   I - Initialisation
#  II - Evolution
# III - Correct repartition 

#---------------------------------------------
# I - Initialisation

function init_Lagrangian(prm)
    (;TA,part) = prm

    if part.n_part == 0 
        particles = nothing
    else
        particles = create_part(prm,part.n_part,TA)
        if part.restart == false
            h5open(prm.path*prm.flag*"_t$(prm.t0).h5", "w") do ff
                create_group(ff, "$(prm.t0)")
                g = ff["$(prm.t0)"]
                g["X"] = Array(particles[1])
                g["Y"] = Array(particles[2])
            end
        else
            h5open(prm.path*prm.flag*"$(prm.t0).h5", "r") do ff
                px = prm.TA(read(ff["X"]))
                py = prm.TA(read(ff["Y"]))
                @. particles[1] = px
                @. particles[2] = py
            end
        end
        

        method = nothing
        if part.meth == "linear"
            method = linear_interpolation
        else
            method = cubic_interpolation
        end
        part =(;part...,method)
        prm  = (;prm..., part=part)
        
    end 
    return prm, particles
end

# init_part() : - create an array with n_part/Nproc particles
#               - their position are uniformly randomly set
#               - also create their associated x,y-velocity array
function create_part(prm,n_part,TA)

    part_x  = TA(rand(prm.grid[1],n_part))
    part_y  = TA(rand(prm.grid[2],n_part))

    # RK4 method 
    part_ux1 = similar(part_x)
    part_ux2 = similar(part_x)
    part_ux3 = similar(part_x)
    part_ux4 = similar(part_x)

    part_uy1 = similar(part_y)
    part_uy2 = similar(part_y)
    part_uy3 = similar(part_y)
    part_uy4 = similar(part_y)

    part_x_save = similar(part_x)
    part_y_save = similar(part_y)

    part_ux_RK4 = [part_ux1,part_ux2,part_ux3,part_ux4,part_x_save]
    part_uy_RK4 = [part_uy1,part_uy2,part_uy3,part_uy4,part_y_save]

  #  part_int_dissip = similar(part_x) 
  #  part_int_dissip_u = similar(part_x)

    return (part_x , part_y, part_ux_RK4,part_uy_RK4)#,part_int_dissip,part_int_dissip_u)
end

# -------------------------------------------
# II - Evolution 

@inline function Lag_RK4_sub_step!(particles,dt,i,thetahs,F_M,wh,w,p)
    # compute Uxi(x(t))
    @. wh = F_M[3] * thetahs
    ldiv!(w,p.plan,wh)
    if particles !== nothing
    field_particles!(particles[1],particles[2],particles[3][i], w,p,p.part.method)
    end

    # compute Uyi(x(t))
    @. wh = F_M[4] * thetahs
    ldiv!(w,p.plan,wh)
    if particles !== nothing
    field_particles!(particles[1],particles[2],particles[4][i], w,p,p.part.method)
    end

    # x^(n+i/4) ( final step is different )
    if i<4
        if particles !== nothing 
            @. particles[1] = particles[3][5] + dt * particles[3][i]
            @. particles[2] = particles[4][5] + dt * particles[4][i]
        end
	end
    
end

@inline function Lag_RK4_final_step!(particles,dt)
    # x^(n+i)
    if particles !== nothing
        @. particles[1] = particles[3][5] + dt/6 * (particles[3][1] + 2 * particles[3][2] + 2 * particles[3][3] + particles[3][4] )
        @. particles[2] = particles[4][5] + dt/6 * (particles[4][1] + 2 * particles[4][2] + 2 * particles[4][3] + particles[4][4] )
    end
end

# Lagrangian - Eulerian  RK4 with analytic viscosity treatment
@inline function L_EDTRK4!(thetahs,thetas,dt,rhs_RK4,particles,w,wh,t,rep,fNL,fL,F_M,p)    
    dt1 = dt/2
    dt2 = dt/2
    dt3 = dt

    # previous field and position saved
    @. rhs_RK4[5] = thetahs           
    if particles !== nothing
       @. particles[3][5] = particles[1] 
       @. particles[4][5] = particles[2]
    end

    # 1st step RK4 Lag-Eul
    Lag_RK4_sub_step!(particles,dt1,1,thetahs,F_M,wh,w,p) 
    fNL(rhs_RK4[1],thetahs,thetas,p,t,w,wh,F_M)
    @. thetahs = thetahs + dt1*rhs_RK4[1]
    fL(thetahs,dt1,F_M[5],p)

     # 2nd step
    Lag_RK4_sub_step!(particles,dt2,2,thetahs,F_M,wh,w,p)
    fNL(rhs_RK4[2],thetahs,thetas,p,t+dt1,w,wh,F_M)
    @. thetahs = rhs_RK4[5] + dt2*rhs_RK4[2]
    fL(thetahs,dt2,F_M[5],p) 

    # 3rd step
    Lag_RK4_sub_step!(particles,dt3,3,thetahs,F_M,wh,w,p)  
    fNL(rhs_RK4[3],thetahs,thetas,p,t+dt2,w,wh,F_M)
    @. thetahs = rhs_RK4[5] + dt3*rhs_RK4[3]
    fL(thetahs,dt3,F_M[5],p) 

    # 4th step
    Lag_RK4_sub_step!(particles,dt,4,thetahs,F_M,wh,w,p)
    fNL(rhs_RK4[4],thetahs,thetas,p,t+dt3,w,wh,F_M)

    fL(rhs_RK4[2],-dt1,F_M[5],p)
    fL(rhs_RK4[3],-dt2,F_M[5],p)
    fL(rhs_RK4[4],-dt3,F_M[5],p)

    @. thetahs = rhs_RK4[5] + dt/6.0*(rhs_RK4[1]+2.0*rhs_RK4[2]+2.0*rhs_RK4[3]+rhs_RK4[4])
    fL(thetahs,dt,F_M[5],p)
    Lag_RK4_final_step!(particles,dt)

    t   = t + dt 
    rep = rep + 1
    return t,rep
end


## III)  Outputs

function lagrangian_output_multi!(particles,p,t,w,wh,thetahs,F_M,i) 
    # u
    @. wh = F_M[3]* thetahs #ux
    ldiv!(w,p.plan,wh)
    field_particles!(particles[1],particles[2],particles[4][1], w,p,p.part.method)

    @. wh = F_M[4]* thetahs #uy
    ldiv!(w,p.plan,wh)
    field_particles!(particles[1],particles[2],particles[6], w,p,p.part.method)

    if i != 0
       lname = p.path*"i$(i)_"*p.flag*"$(round(t,digits=6))"
    else
       lname = p.path*p.flag*"$(round(t,digits=6))"
    end
    return lname
end