using Plots
using HDF5

# Specify the full path to the .h5 file
filepath = "D:\\SQG_FD\\IPM\\Instability\\first_run\\inf_nonux/"
name     = "scalar_t14.212733.h5"

N     = (512,2048)


theta = zeros(N[1],N[2])

# Read data from .h5 file
file = h5open(filepath*name, "r")
theta = read(file["theta"])
close(file)


# Plot
x = LinRange(0,2pi,N[1])
y = LinRange(0,8pi,N[2])

println(maximum(theta))
println(minimum(theta))

heatmap_plot  = heatmap(x,y[Int(round(N[2]/16)):Int(round(15N[2]/16))],transpose(theta)[Int(round(N[2]/16)):Int(round(15N[2]/16)),1:end], color=:ice,xticks=nothing,yticks=nothing,dpi=300,clims=(-1,1))
plot!(size=(400,600))

savefig(heatmap_plot,filepath*name[1:end-2]*"png" )












  