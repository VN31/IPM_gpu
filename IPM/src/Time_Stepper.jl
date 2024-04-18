
########      TIME STEPPER    #########
#######################################

#  I  -  Numerical schemes 
#  II -  Adaptative time step 
# III -  Time loop exit's condition

# ------------------------------------------------------------
### I - Numerical schemes : Only RK4 here

# EDTRK4
function EDTRK4!(thetahs,thetas,dt,rhs_RK4,w,wh,t,rep,fNL,fL,F_M,p)    
    dt1 = dt/2
    dt2 = dt/2
    dt3 = dt

    @. rhs_RK4[5] = thetahs

    fNL(rhs_RK4[1],thetahs,thetas,p,t,w,wh,F_M)
    @. thetahs = thetahs + dt1*rhs_RK4[1]
    fL(thetahs,dt1,F_M[5],p)

    fNL(rhs_RK4[2],thetahs,thetas,p,t+dt1,w,wh,F_M)
    @. thetahs = rhs_RK4[5] + dt2*rhs_RK4[2]
    fL(thetahs,dt2,F_M[5],p) 

    fNL(rhs_RK4[3],thetahs,thetas,p,t+dt2,w,wh,F_M)
    @. thetahs = rhs_RK4[5] + dt3*rhs_RK4[3]
    fL(thetahs,dt3,F_M[5],p) 

    fNL(rhs_RK4[4],thetahs,thetas,p,t+dt3,w,wh,F_M)

    fL(rhs_RK4[2],-dt1,F_M[5],p)
    fL(rhs_RK4[3],-dt2,F_M[5],p)
    fL(rhs_RK4[4],-dt3,F_M[5],p)

    @. thetahs = rhs_RK4[5] + dt/6.0*(rhs_RK4[1]+2.0*rhs_RK4[2]+2.0*rhs_RK4[3]+rhs_RK4[4])
    fL(thetahs,dt,F_M[5],p)

    t   = t + dt 
    rep = rep + 1
    return t,rep
end
# --------------------------------------------------------------

### II - Adaptative time step 

# cfl() :
#  - return the maximal time step respecting cfl condition.
#  - the constant is defined on Use_Me.jl
function cfl(thetahs,w,wh,Mux,Muy,p)
    grid = p.grid
    dx = grid[1][2]-grid[1][1]
    C  = p.Ccfl
    velocity_x!(wh,thetahs,Mux)
    ldiv!(w,p.plan,wh)
    ux_max = maximum(w)
    @show ux_max
    velocity_y!(wh,thetahs,Muy)
    ldiv!(w,p.plan,wh)
    uy_max = maximum(w)
    @show uy_max

    u_max = maximum((ux_max,uy_max))
    dt = C * dx / u_max
    return dt
end

function check_cfl(thetahs,w,wh,Mux,Muy,prm)
    dt = prm.dt
    dt_cfl =  cfl(thetahs,w,wh,Mux,Muy,prm)

    if dt>dt_cfl 
        println( "ERROR :  time step > dt_cfl : dt = $dt and dt_cfl = $dt_cfl")
        error(1)
    else
        println( "Time step < dt_cfl : dt = $dt and dt_cfl = $dt_cfl")
    end
end

# timestep() :
#  - return the minimum dt between current / max cfl / max BKM.
function timestep(thetahs,w,wh,Mux,Muy,p)
    return cfl(thetahs,w,wh,Mux,Muy,p)
end

# -------------------------------------------------------------

### III - time loop exit's condition 

# ARTUNG : Remove check_good_resol on GPU
function exit_loop(nrep,time,thetahs,p)
    run = true
    if time>p.tmax || nrep>p.rep_max || isnan(time) || time<0.0
        run = false
    end
    println("time>tmax : $(time>p.tmax), nrep>rep_max : $(nrep>p.rep_max), isnan(time) : $(isnan(time)), neg(time) : $(time<0) ") 
    return run
end

# ----------------------------------------------------------