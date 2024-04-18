###########################################################
########               Fields Equations       #############
###########################################################
#   I - Pseudo spectral SQg
#  II - Forcing's functions
# III - Perturbation's functions


# matrices : Main used fourier arrays
@inline function matrices(params)
    (;Ns,grid_fourier,hp,TA) = params

    Kx =  grid_fourier[1]  .* ones(Ns[2])'
    Ky =  ones(Int(Ns[1]/2+1)) .* Array(grid_fourier[2])'

    Mux = zeros(ComplexF64,size(Kx))
    Muy = zeros(ComplexF64,size(Kx))
    Md  = similar(Kx)

    Kx  = TA(Kx)
    Ky  = TA(Ky)
    Mux = TA(Mux)
    Muy = TA(Muy)

    @. Mux =    Kx*Ky / (Kx^2+Ky^2)^(2/2)
    @. Muy =    -Kx^2 / (Kx^2+Ky^2)^(2/2)

    if params.r_type=="Laplacian"
        Mk2    =  similar(Kx)
        @. Mk2 = (Ky^2)^(hp)
    end

    CUDA.@allowscalar  Mux[1,1:end] .= Mux[1,1] = 0.0 
    CUDA.@allowscalar  Muy[1,1:end] .= Muy[1,1] = 0.0 

    # 2/3 dealiasing filter 
    ks_lim_x = Ns[1]/3
    ks_lim_y = Ns[2]/3

    @inbounds for (n, I) in enumerate(CartesianIndices(Md)) 
        i,j = Tuple(I) 

        kv = (grid_fourier[1][i],grid_fourier[2][j])
        if any(abs(kv[1]) > ks_lim_x || abs(kv[2]) > ks_lim_y) 
            Md[n] = 0
        else 
            Md[n] = 1
        end
    end

    Md = TA(Md)

    if params.r_type=="Laplacian"
        return [Kx,Ky,Mux,Muy,Mk2,Md]
    else 
        return [Kx,Ky,Mux,Muy,Md]
    end        
end

# ---------------------------------------------------------
# I - Pseudo spectral SQg equations 
# u(theta)

@inline function velocity_y!(uyhs,thetahs,Maty)
    @. uyhs = Maty * thetahs
end

@inline function velocity_x!(uxhs,thetahs,Matx)
     @. uxhs = Matx * thetahs
end

@inline function scalar_advection!( 
    Fhs, thetahs,thetas, plan,w,wh,Kx,Ky,Mux,Muy)

    velocity_x!(wh,thetahs,Mux)
    ldiv!(w,plan,wh)

    @. w = thetas * w    # w = theta * v_l in physical space
    mul!(wh, plan, w)      # same in Fourier space
    @. Fhs = - im *Kx * wh

    velocity_y!(wh,thetahs,Muy)
    ldiv!(w,plan,wh)

    @. w = thetas * w  # w = theta * v_l in physical space
    mul!(wh, plan, w)   

    @. Fhs += - im * Ky * wh
end

@inline function dealias_twothirds!(whs, Md)
  @.  whs = Md * whs 
end

# all together but whitout viscosity 
@inline function rhs_inv!(dthetahs, thetahs,thetas, p,t,w,wh,F_M) #Kx,Ky,Mux,Muy,Md

    (; plan) = p

    ## 1. Get the fields in Fourier and real spaces
    ldiv!(thetas, plan, thetahs)

    ## 2. Compute non-linear term and dealias it
    scalar_advection!(dthetahs,thetahs, thetas,plan,w,wh,F_M[1],F_M[2],F_M[3],F_M[4])
    dealias_twothirds!(dthetahs,F_M[end])
end

@inline function zero_if_inf(a)
        if a > 10^80
           return 0
        else
            return a
        end
end

@inline function an_visco!(dthetahs,dt,Mk2,p)
    dtnu = dt*p.nu
    @. dthetahs = dthetahs * exp(-dtnu*Mk2)
end
	 

# ---------------------------------------------------------------------------------------
## II - Forcing : it supposes that klim<N/Nproc

function init_forcing(prm,thetahs)
    (; type, prmf) = prm.forc_IC
    println("Forcing type : $type")

    if type == "TypeI"
        (; k_d) = prmf
        Random.seed!(N[1])  
        damped_modes = select_damped_modes(prm,k_d)
        prmf = (; prmf... , damped_modes)
    end

    forcing = (;type=type,prmf=prmf)
    prm = (;prm..., forcing=forcing)
    return prm
end

function select_damped_modes(params,klim)
    Ns = params.Ns
    n = 1
    A = []
    @inbounds for i in 1:Int(Ns[1]/2)+1
        @inbounds for j in 1:Ns[2]
            kv  = (params.grid_fourier[1][i],params.grid_fourier[2][j])
            k   = sqrt(kv[1]^2+kv[2]^2)
            if k <= klim
                push!(A,[i,j])
                n += 1
            end
        end
    end
    @show A 
    na  = length(A)
    d_m = zeros(na,2)
    for i in 1:na
        d_m[i,1] = A[i][1]
        d_m[i,2] = A[i][2]
    end
    @show d_m
    return d_m
end

function gaussian_forcing!(thetahs,dt,prmf,scale_factor,Kx,Ky,N)
    (;F,kf,sigma) = prmf
    @. thetahs += sqrt(dt*2*(F/(2pi*kf)))  * 1/(2pi*sqrt(Kx^2+Ky^2)*sigma*sqrt(2pi))^(0.5)* exp(-(sqrt(Kx^2+Ky^2)-kf)^2/(4*sigma^2)) * randn(ComplexF64) * scale_factor

    for i in 2:Int(N/2)
       CUDA.@allowscalar thetahs[1,i] = conj(thetahs[1,N+2-i]) # Hermitian symmetry !!
    end
end

function linear_damping!(thetahs,dt,prmf)
    (;damped_modes,alpha) = prmf
    for i in size(damped_modes)[1] # linear damping
    kx , ky = damped_modes[i,:]
    CUDA.@allowscalar thetahs[Int(kx) , Int(ky)] = thetahs[Int(kx) ,Int(ky)]*exp(-alpha*dt)
    end
end


# ---------------------------------------- Not used in this version
# simple Kernel Abstraction for the matrice point wise multiplication

@kernel function point_mul_kernel!(a, b)
    p = @index(Global)

    a[p] = a[p] * b[p]

end

# Creating a wrapper kernel for launching with error checks
function point_mul!(a, b)
    kernel! = point_mul_kernel!(device(a), 256)
    kernel!(a, b) 
end

# same for velocity from kx, ky matrices

@kernel function velocityH_x_kernel!(uhx, thetah,Kx,Ky,prm)
    p = @index(Global)

    k_gamma = (Ky[p]^2 + Kx[p]^2)^prm.gamma
    uhx[p] = - thetah[p] * im * Ky[p] / k_gamma

end

@kernel function velocityH_y_kernel!(uhy, thetah,Kx,Ky,prm)
    p = @index(Global)

    k_gamma = (Ky[p]^2 + Kx[p]^2)^prm.gamma
    uhy[p] =  thetah[p] * im * Kx[p] / k_gamma

end

function velocityH_x!(uhx, thetah,Kx,Ky,prm)
    kernel! = velocityH_x_kernel!(device(thetah), 256)
    kernel!(uhx, thetah,Kx,Ky,prm) 
end

function velocityH_y!(uhy, thetah,Kx,Ky,prm)
    kernel! = velocityH_y_kernel!(device(thetah), 256)
    kernel!(uhy, thetah,Kx,Ky,prm) 
end

function velocity_x_KA!(w,wh,thetah,Kx,Ky,prm)
    velocityH_x!(wh, thetah,Kx,Ky,prm)
    ldiv!(w,prm.plan,wh)
end

function velocity_y_KA!(w,wh,thetah,Kx,Ky,prm)
    velocityH_y!(wh, thetah,Kx,Ky,prm)
    ldiv!(w,prm.plan,wh)
end

