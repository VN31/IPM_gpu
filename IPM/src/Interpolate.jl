### Lagrangian Solver sub-routine : Interpolation.

using KernelAbstractions

#if Base.find_package("CUDA") !== nothing
#    using CUDA
#    using CUDA.CUDAKernels
#    const backend = CUDABackend()
#    CUDA.allowscalar(false)
#else
#    const backend = CPU()
#end

"""
I -  Interpolation : 
Interpolate `field` to the physical point `(x, y, z)`
Note that this is a lower-level method defined for use in CPU/GPU kernels.

"""
#fractional_indices(x,y,prm)
#   - Convert the coordinates `(x, y)` to _fractional_ 
#     indices on a regular rectilinear grid.
@inline function fractional_indices(x,y,Ns,ls)

    xi = mod(x,ls[1])
    yi = mod(y,ls[2])

    i = xi * Ns[1] / ls[1]
    j = yi * Ns[2] / ls[2]

    return (i,j)
end

@inline function modN(i,N)
    if i == N
        return N 
    else 
        return mod(i,N)
    end
end

# a) bi-Linear Interpolation.
@inline ϕ₁(ξ, η) = (1 - ξ) * (1 - η) 
@inline ϕ₂(ξ, η) = (1 - ξ) *      η 
@inline ϕ₃(ξ, η) =      ξ  * (1 - η)
@inline ϕ₄(ξ, η) =      ξ  *      η

@inline _linear_interpolation(field, ξ, η, i, j,Ns) =
    @inbounds (  ϕ₁(ξ, η) * field[i,   j   ]
               + ϕ₂(ξ, η) * field[i,   modN(j+1,Ns[2]) ]
               + ϕ₃(ξ, η) * field[modN(i+1,Ns[1]), j   ]
               + ϕ₄(ξ, η) * field[modN(i+1,Ns[1]), modN(j+1,Ns[2]) ] )

@inline function linear_interpolation(field, x, y, Ns,ls)
    i, j = fractional_indices(x, y, Ns,ls)

    # Convert fractional indices to unit cell coordinates 0 <= (ξ, η) <=1
    # and integer indices (with 0-based indexing).
    # For why we use Base.unsafe_trunc instead of trunc see: Oceananigans
    ξ, i = mod(i, 1), Base.unsafe_trunc(Int, i)
    η, j = mod(j, 1), Base.unsafe_trunc(Int, j)

    # Convert indices to proper integers and shift to 1-based indexing.
    return _linear_interpolation(field, ξ, η, Int(i+1), Int(j+1),Ns)
end

# b) bi-Cubic Interpolation

@inline ϕ(p0,p1,p2,p3,ξ) = p1 + ξ *( (p2-p0)/2.0 + ξ* ((p0-5.0/2.0*p1+2.0*p2-p3/2.0) + ξ*(-p0+3.0*p1-3.0*p2+p3)/2.0 )) 

@inline _cubic_interpolation(field, ξ, η, i, j,Ns) =
    @inbounds (  ϕ(  ϕ(field[modN(i-1,Ns[1]),modN(j-1,Ns[2])],field[modN(i-1,Ns[1]),j],field[modN(i-1,Ns[1]),modN(j+1,Ns[2])],field[modN(i-1,Ns[1]),modN(j+2,Ns[2])],η) ,
                     ϕ(field[i,modN(j-1,Ns[2])],field[i,j],field[i,modN(j+1,Ns[2])],field[i,modN(j+2,Ns[2])],η),
                     ϕ(field[modN(i+1,Ns[1]),modN(j-1,Ns[2])],field[modN(i+1,Ns[1]),j],field[modN(i+1,Ns[1]),modN(j+1,Ns[2])],field[modN(i+1,Ns[1]),modN(j+2,Ns[2])],η),
                     ϕ(field[modN(i+2,Ns[1]),modN(j-1,Ns[2])],field[modN(i+2,Ns[1]),j],field[modN(i+2,Ns[1]),modN(j+1,Ns[2])],field[modN(i+2,Ns[1]),modN(j+2,Ns[2])],η),
                     ξ
                  )
               )

@inline function cubic_interpolation(field, x, y, Ns,ls)
    i, j = fractional_indices(x, y, Ns,ls)

    ξ, i = mod(i, 1), Base.unsafe_trunc(Int, i)
    η, j = mod(j, 1), Base.unsafe_trunc(Int, j)

    # Convert indices to proper integers and shift to 1-based indexing.
    return _cubic_interpolation(field, ξ, η, Int(i+1), Int(j+1),Ns)
end

# c) field's values at the positions of the particles
# low level GPU method using KernelAbstractions.jl 

# one particle
@inline function field_particle((x, y),field,method,Ns,ls)
    fx   = method(field, x, y, Ns,ls)

    return fx
end

# GPU/CPU kernel for particles array
@kernel function _field_particles!(posX,posY, field_A, field,method,Ns,ls) 
    p = @index(Global)

    @inbounds begin
        x = posX[p]
        y = posY[p]
    end

    fx = field_particle((x,y), field,method,Ns,ls) 

    @inbounds begin
        field_A[p] = fx 
    end
end

# Creating a wrapper kernel for launching
function field_particles!(posX,posY, field_A, field,prm,method)
    if prm.TA == Array
        field_particles_kernel! = _field_particles!(CPU(),2)
    else 
        field_particles_kernel! = _field_particles!(backend)
    end

    (;Ns,Ls,part) = prm
    field_particles_kernel!(posX,posY,field_A, field,method,Ns,Ls,ndrange=size(posX))

    return nothing
end
