############ Utilities #################

# I   - Grids Construction : local/global, fourier/real
# II  - Intermediate fields building
# III - Useful functions 

# ---------------------------------------------------------------
# I - Grids
function real_grids(Ns,Ls)
    # Last point egals the first one
    grid = map((N, L) -> range(0, L; length = N+1), Ns, Ls)
    return (grid[1][1:end-1],grid[2][1:end-1]) 
end

function spec_grids(Ns,Ls)

    grid_fourier = (
        rfftfreq(Ns[1], 2π * Ns[1] / Ls[1]), # kx | real-to-complex
        fftfreq(Ns[2], 2π * Ns[2] / Ls[2]),  # ky | complex-to-complex
            )
    ks = rfftfreq(Ns[1], 2π * Ns[1] / Ls[1]) # 1D for spectral diagnostics

    return grid_fourier,ks
end


# ----------------------------------------------------------------
# II - Intermediate fields
function intermediate_RK4(thetahs)
    K1        = similar(thetahs)
    K2        = similar(thetahs)
    K3        = similar(thetahs)
    K4        = similar(thetahs) 
    thetaSave = deepcopy(thetahs)
    return [K1,K2,K3,K4,thetaSave]
end

function inter_fields(p,theta)
    w  = similar(theta)
    wh = p * w
    return w,wh
end

# ---------------------------------------------------------------------
# III - useful functions

function deletefields!(list)
    for f in list
        f = nothing
    end
end