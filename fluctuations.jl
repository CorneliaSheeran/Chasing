
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
s       = 0.95                  # Homozgous fitness cost of drive 
h       = 0.3                   # h*s = fitness cost of heterozygous W/D individual
Wf      = [1 1-h*s; 1-h*s 1-s]  # Female fitness matrix
Wm      = [1 1; 1 1]            # Male fitness matrix
R       = 6                     # Growth rate
f0      = [0.99, 0.01]           # Total starting frequency for W and D alleles
N0      = 5 #1e6 #200 #1e6 #1e6;                  # Total starting population
#Cc      = 1400 #1e6 #200 #1e6;                  # Carrying capacity
#alpha2  = Cc/(R-1)              # Discrete model carrying capacity


# Rate-based Model Constants
A = 1
B = 1
C = 1
D = 1-h*s
E = 1-h*s
F = 1-h*s

alpha = (1-s)
beta  = (1-s)
gamma = (1-s)

dt = 0.01 # Time step for continous time model
#T  = 100   # Number of generations simulated for
N  = 35
#Di =  10 #5 #0.05

# Function to add positive Gaussian white noise to a signal
function add_gaussian_noise(mean, std_dev)
    dist = Normal(mean, std_dev)
    noise = rand(dist, (N, N)) #rand(Poisson(std_dev), (N, N)) #
    return noise
end


#size(population_size)
function continuous_time(N0, f0, Cc, dt, T, Di, SD)

    timed = 0:dt:T;

    N_all       = zeros(size(timed)[1], N, N, 3);
    z_all       = zeros(size(timed)[1], N, N, 3) .+ 1;

    N_all[1, :, :, 1] = zeros(N, N) #.+ N0*f0[1]^2 .+ N0*2*f0[1]*f0[2]  #- rand(-10:10, N, N).*(N0*f0[1]^2/200));
    N_all[1, :, :, 2] = zeros(N, N) #.+ N0*2*f0[1]*f0[2] #- rand(-10:10, N, N).*(N0*2*f0[1]*f0[2]/20));
    N_all[1, :, :, 3] = zeros(N, N) #.+ N0*f0[2]^2) #- rand(-10:10, N, N).*(N0*f0[2]^2/50);
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
        td = add_gaussian_noise(0, SC)

        #println(zd)
        


        # dNwwdt = -(2*N_all[ind-1, :, :, 1] + N_all[ind-1, :, :, 2] + N_all[ind-1, :, :, 3] + (-72*N_all[ind-1, :, :, 1]^2/Cc - 61.2.*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 2]/Cc - 36*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 3]/Cc - 2*N_all[ind-1, :, :, 1].*zd + N_all[ind-1, :, :, 1].*(-xd - yd - zd)
        # - N_all[ind-1, :, :, 1].*(2*xd + 2*yd + 2*zd)./2 - 25.2*N_all[ind-1, :, :, 2]^2/Cc - 25.2*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :, 3]/Cc - N_all[ind-1, :, :, 2].*yd - N_all[ind-1, :, :, 2].*(2*xd + 2*yd + 2*zd)./2 - N_all[ind-1, :, :, 3].*xd - N_all[ind-1, :, :, 3].*(2*xd + 2*yd + 2*zd)./2))
        # dNwwdt = dNwwdt + Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) - 2*Di*(N_all[ind-1, :, :, 1].*laplacian_2d(zd, 1, 1))
        
        # dNwddt = -(N_all[ind-1, :, :, 1] + 2*N_all[ind-1, :, :, 2] + N_all[ind-1, :, :, 3] + (-36*N_all[ind-1, :, :, 1]^2/Cc - 122.4*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 2]/Cc - 72*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 3]/Cc - N_all[ind-1, :, :, 1].*zd - N_all[ind-1, :, :, 1].*(2*xd + 2*yd + 2*zd)./2 - 25.2*N_all[ind-1, :, :, 2]^2/Cc - 25.2*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :, 3]/Cc 
        # - 2*N_all[ind-1, :, :, 2].*yd + N_all[ind-1, :, :, 2].*(-xd - yd - zd) - N_all[ind-1, :, :, 2].*(2*xd + 2*yd + 2*zd)./2 - N_all[ind-1, :, :, 3].*xd - N_all[ind-1, :, :, 3].*(2*xd + 2*yd + 2*zd)./2)) 
        # dNwddt = dNwwdt  + Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) - 2*Di*(N_all[ind-1, :, :, 2].*laplacian_2d(yd, 1, 1))
        
        # dNdddt = -(N_all[ind-1, :, :, 1] + N_all[ind-1, :, :, 2] + 2*N_all[ind-1, :, :, 3] + (-36*N_all[ind-1, :, :, 1]^2/Cc - 61.2*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 2]/Cc - 36*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 3]/Cc - N_all[ind-1, :, :, 1].*zd - N_all[ind-1, :, :, 1].*(2*xd + 2*yd + 2*zd)./2 - 50.4*N_all[ind-1, :, :, 2]^2/Cc
        # - 50.4*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :, 3]/Cc - N_all[ind-1, :, :, 2].*yd - N_all[ind-1, :, :, 2].*(2*xd + 2*yd + 2*zd)/2 - 2*N_all[ind-1, :, :, 3].*xd + N_all[ind-1, :, :, 3].*(-xd - yd - zd) - N_all[ind-1, :, :, 3].*(2*xd + 2*yd + 2*zd)/2)) 
        # dNdddt = dNwwdt + Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) - 2*Di*(N_all[ind-1, :, :, 3].*laplacian_2d(xd, 1, 1))




        dNwwdt = N_all[ind-1, :, :, 1].*(sqrt(Cc).*(36*N_all[ind-1, :, :, 1] .+ zd) .- Cc )/Cc  #- N_all[ind-1, :, :, 1]*td/sqrt(Cc)
        dNwwdt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) #.- 2*Di*(N_all[ind-1, :, :, 1].*laplacian_2d(zd, 1, 1))/SC
        
        dNwddt = (61.2*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 2] .+ 36*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 3] .+ N_all[ind-1, :, :, 2].*yd/SC .- N_all[ind-1, :, :, 2]) #- N_all[ind-1, :, :, 2]*td/sqrt(Cc)
        dNwddt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) #.- 2*Di*(N_all[ind-1, :, :, 2].*laplacian_2d(yd, 1, 1))/SC
        
        dNdddt = -(sqrt(Cc)*(-25.2*SC*N_all[ind-1, :, :, 2]^2 .- 25.2*SC*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :, 3] .- N_all[ind-1, :, :, 3].*xd) .+ Cc*N_all[ind-1, :, :, 3])/Cc  #- N_all[ind-1, :, :, 3]*td/sqrt(Cc)
        dNdddt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1) #.- 2*Di*(N_all[ind-1, :, :, 3].*laplacian_2d(xd, 1, 1))/SC
        


        # dNwwdt = N_all[ind-1, :, :, 1].*(sqrt(Cc).*(36*N_all[ind-1, :, :, 1] .+ z_all[ind, :, :, 1]) .- Cc )/Cc -
        # dNwwdt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1)*4*(-Di * log(0.01))*pi*2/(3* sqrt(3)) .- 2*Di*(N_all[ind-1, :, :, 1].*laplacian_2d(z_all[ind, :, :, 1], 1, 1))*4*(-Di * log(0.01))*pi*2/(3* sqrt(3))/SC
        
        # dNwddt = (61.2*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 2] .+ 36*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 3] .+ N_all[ind-1, :, :, 2].*z_all[ind, :, :, 2]/SC .- N_all[ind-1, :, :, 2]) 
        # dNwddt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1)*4*(-Di * log(0.01))*pi*2/(3* sqrt(3)) .- 2*Di*(N_all[ind-1, :, :, 2].*laplacian_2d(z_all[ind, :, :, 2], 1, 1))*4*(-Di * log(0.01))*pi*2/(3* sqrt(3))/SC
        
        # dNdddt = -(sqrt(Cc)*(-25.2*SC*N_all[ind-1, :, :, 2]^2 .- 25.2*SC*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :, 3] .- N_all[ind-1, :, :, 3].*z_all[ind, :, :, 3]) .+ Cc*N_all[ind-1, :, :, 3])/Cc 
        # dNdddt = dNwwdt .+ Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1)*4*(-Di * log(0.01))*pi*2/(3* sqrt(3)) .- 2*Di*(N_all[ind-1, :, :, 3].*laplacian_2d(z_all[ind, :, :, 3], 1, 1))*4*(-Di * log(0.01))*pi*2/(3* sqrt(3))/SC
        

        # dZdt = (sqrt(Cc)*(sqrt(Cc)*(- 144*N_all[ind, :, :, 1]*z_all[ind, :, :, 1] - 122.4*N_all[ind, :, :, 2]*z_all[ind, :, :, 2] - 72*N_all[ind, :, :, 3]*z_all[ind, :, :, 2]) - Cc*z_all[ind, :, :, 1]^2) + 2*Cc^2*(z_all[ind, :, :, 1]))/(2*Cc^2) 
        # dYdt = (sqrt(Cc)*(sqrt(Cc)*(- 122.4*N_all[ind, :, :, 1]*z_all[ind, :, :, 2] - 100.8*N_all[ind, :, :, 2]*z_all[ind, :, :, 3] - 50.4*N_all[ind, :, :, 3]*z_all[ind, :, :, 3]) - Cc*(z_all[ind, :, :, 2]^2)) + 2*Cc^2*(z_all[ind, :, :, 2]))/(2*Cc^2) 
        # dXdt = (sqrt(Cc)*(sqrt(Cc)*(- 72*N_all[ind, :, :, 1]*z_all[ind, :, :, 2] - 50.4*N_all[ind, :, :, 2]*z_all[ind, :, :, 3]) - Cc*(z_all[ind, :, :, 3]^2)) + 2*Cc^2*(z_all[ind, :, :, 3]))/(2*Cc^2) 



        N_all[ind, :, :, 1] = N_all[ind-1, :, :, 1] + dt.*dNwwdt #+ rand(N, N).*10
        N_all[ind, :, :, 2] = N_all[ind-1, :, :, 2] + dt.*dNwddt #+  rand(N, N).*10
        N_all[ind, :, :, 3] = N_all[ind-1, :, :, 3] + dt.*dNdddt #+  rand(N, N).*1

        # z_all[ind, :, :, 1] = z_all[ind-1, :, :, 1] + dt.*dZdt #+ rand(N, N).*10
        # z_all[ind, :, :, 2] = z_all[ind-1, :, :, 2] + dt.*dYdt #+  rand(N, N).*10
        # z_all[ind, :, :, 3] = z_all[ind-1, :, :, 3] + dt.*dXdt #+  rand(N, N).*1



    end
    

    return N_all
end



# N_all      = continuous_time(N0, f0, Cc, dt, 150, 2)
# h5write("Nalldat3.h5", "dats", N_all)



# ####CONTINUOUS
# N_all      = continuous_time(N0, f0, Cc, dt, 12, 0.05,)

# ###COMPARISION

# time = 0 #round(Int, 45*(1/dt)) 

# m=10

# plot1 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW") #, clims=(0, Cc))
# plot2 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m") #, clims=(0, Cc))
# plot3 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD") #, clims=(0, Cc))

# total1 = plot(plot1, plot2, plot3, layout=(2, 2))
# display(total1)

dd = 0.1:0.05:2
cc = 10:100:1000
sd = 0:1

pmig = exp.(-(dd.*pi*4*dt).^(-1));



#for k=1:size(sd)[1]
PD = zeros(size(dd)[1], size(cc)[1])
for i=1:size(dd)[1]
    println(i)
    for j=1:size(cc)[1]

        N_all      = continuous_time(N0, f0, cc[j], dt, 7, dd[i], 0)
        #println(N_all[round(Int, 5*(1/dt)+1), 5, 5, 1])
        if isnan(N_all[round(Int, 5*(1/dt)+1), 5, 5, 3])
            #println("here!")
            PD[i, j] = 1
        end
    end
end


plot1 = heatmap(cc, pmig, PD, c=:viridis, ylabel="Pmig", xlabel="Carrying Capacity") #, clims=(0, Cc))
display(plot1)
#end

# for m=0:10:150

#     plot1 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW") #, clims=(0, Cc))
#     plot2 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m") #, clims=(0, Cc))
#     plot3 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD") #, clims=(0, Cc))

#     # plot1 = heatmap(ceil.(N_all[round(Int, m+1), :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW") #, clims=(0, Cc))
#     # plot2 = heatmap(ceil.(N_all[round(Int, m+1), :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m") #, clims=(0, Cc))
#     # plot3 = heatmap(ceil.(N_all[round(Int, m+1), :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD") #, clims=(0, Cc))

#     # Combine the plots into a single plot with a layout
#     total1 = plot(plot1, plot2, plot3, layout=(2, 2))
#     display(total1)
# end