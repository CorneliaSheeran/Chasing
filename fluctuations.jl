
using LinearAlgebra
using Plots
using MAT
include("haplotype_model.jl")
using MAT
using Noise
using Random
using Distributions
using HDF5
using Statistics



function laplacian_2d(x, delta_r1, delta_r2)
    """
    Compute the Laplacian of a function x(r1, r2) using finite differences on a 2D grid.
    
    Parameters:
        x (2D array-like): Array containing values of x sampled on a 2D grid.
        delta_r1 (float): Grid spacing along the r1 direction.
        delta_r2 (float): Grid spacing along the r2 direction.
    
    Returns:
        2D array-like: Array containing the Laplacian of x at each grid point.
    """
    N1, N2 = size(x)
    laplacian = zeros(N1, N2)
    
    for i in 2:N1-1
        for j in 2:N2-1
            laplacian[i, j] = (x[i-1, j] - 2*x[i, j] + x[i+1, j]) / delta_r1^2 +
                              (x[i, j-1] - 2*x[i, j] + x[i, j+1]) / delta_r2^2
        end
    end
    
    # Boundary conditions (assuming Neumann boundary conditions)
    # For simplicity, assuming a square grid and constant grid spacing
    laplacian[1, :]   .= laplacian[2, :]
    laplacian[end, :] .= laplacian[end-1, :]
    laplacian[:, 1]   .= laplacian[:, 2]
    laplacian[:, end] .= laplacian[:, end-1]
    
    return laplacian
end



# Discrete Model Constants
epsilon = 1                # Efficiency of drive cleavage per target site
dt = 0.05 # Time step for continous time model
N  = 35


# Function to add positive Gaussian white noise to a signal
function add_gaussian_noise(mean, std_dev)
    dist = Normal(mean, std_dev)
    noise = rand(dist, (N, N)) #rand(Poisson(std_dev), (N, N)) #
    return noise
end


#size(population_size)
function continuous_time(Cc, dt, T, Di, SD)

    timed = 0:dt:T;

    N_all       = zeros(size(timed)[1], N, N, 3);


    N_all[1, 2:5, 2:5, 3] = zeros(4, 4) .+ 1
    N_all[1, 2:5, 2:5, 1] = zeros(4, 4) .+ 1
    N_all[1, 2:5, 2:5, 2] = zeros(4, 4) .+ 1
    # N_all[1, :, 10:end, 3] = zeros(N, N-9)

    ind = 1;

    #Cc = Cc;

    for t in dt:dt:T

        t = dt*ind
    
        #println(t)

        ind = round(Int, ind + 1);
        SC = sqrt(Cc)

        xd = add_gaussian_noise(0, SC)
        yd = add_gaussian_noise(0, SC)
        zd = add_gaussian_noise(0, SC)

        dNwwdt = N_all[ind-1, :, :, 1].*(sqrt(Cc).*(36*N_all[ind-1, :, :, 1] .+ zd) .- Cc )/Cc  #- N_all[ind-1, :, :, 1]*td/sqrt(Cc)
        dNwwdt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) #.- 2*Di*(N_all[ind-1, :, :, 1].*laplacian_2d(zd, 1, 1))/SC
        
        dNwddt = (61.2*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 2] .+ 36*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 3] .+ N_all[ind-1, :, :, 2].*yd/SC .- N_all[ind-1, :, :, 2]) #- N_all[ind-1, :, :, 2]*td/sqrt(Cc)
        dNwddt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) #.- 2*Di*(N_all[ind-1, :, :, 2].*laplacian_2d(yd, 1, 1))/SC
        
        dNdddt = -(sqrt(Cc)*(-25.2*SC*N_all[ind-1, :, :, 2]^2 .- 25.2*SC*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :, 3] .- N_all[ind-1, :, :, 3].*xd) .+ Cc*N_all[ind-1, :, :, 3])/Cc  #- N_all[ind-1, :, :, 3]*td/sqrt(Cc)
        dNdddt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1) #.- 2*Di*(N_all[ind-1, :, :, 3].*laplacian_2d(xd, 1, 1))/SC
        

        N_all[ind, :, :, 1] = N_all[ind-1, :, :, 1] + dt.*dNwwdt #+ rand(N, N).*10
        N_all[ind, :, :, 2] = N_all[ind-1, :, :, 2] + dt.*dNwddt #+  rand(N, N).*10
        N_all[ind, :, :, 3] = N_all[ind-1, :, :, 3] + dt.*dNdddt #+  rand(N, N).*1




    end
    

    return N_all
end




dd = 0.1:1:7
cc = 10:100:1000

PD = zeros(size(dd)[1], size(cc)[1])
for i=1:size(dd)[1]
    println(i)
    for j=1:size(cc)[1]

        N_all      = continuous_time(cc[j], dt, 7, dd[i], 0)
        #println(N_all[round(Int, 5*(1/dt)+1), 5, 5, 1])
        if isnan(N_all[round(Int, 5*(1/dt)+1), 5, 5, 3])
            #println("here!")
            PD[i, j] = 1
        end
    end
end


plot1 = heatmap(cc, dd, PD, c=:viridis, ylabel="D", xlabel="Carrying Capacity") #, clims=(0, Cc))
display(plot1)