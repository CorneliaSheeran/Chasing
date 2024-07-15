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
epsilon = 0.7                # Efficiency of drive cleavage per target site
s       = 0.95                  # Homozgous fitness cost of drive 
h       = 0.3                   # h*s = fitness cost of heterozygous W/D individual
Wf      = [1 1-h*s; 1-h*s 1-s]  # Female fitness matrix
Wm      = [1 1; 1 1]            # Male fitness matrix
R       = 6                     # Growth rate
f0      = [0.9, 0.1]           # Total starting frequency for W and D alleles
N0      = 1e3 #1e6 #200 #1e6 #1e6;                  # Total starting population
Cc      = 1e3 #1e6 #200 #1e6;                  # Carrying capacity
alpha2  = Cc/(R-1)              # Discrete model carrying capacity


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

dt = 0.05 # Time step for continous time model
T  = 100   # Number of generations simulated for
N  = 35
Di = 0.05 #5 #0.05

Dtry = 0.1

#hexmig_TSR(1, s, h, 0.00, 0.00, 6, 100*N^2,   0.0, 0.01, epsilon, 0.0, 0.0,0.0,0.0,N, N, 0.05, 0.0, 50, 1000, 0, 0) #


dd = 0.01:0.01:1
cc = 10:50:1000

#pmig = exp.(-(dd.*pi*4*dt).^(-1));

#for k=1:size(sd)[1]
PD = zeros(size(dd)[1], size(cc)[1]) .+ 1;
for i=1:size(dd)[1]
    println(i)
    for j=1:size(cc)[1]

        fem_freq, male_freq, population_size, ~, ~ =  hexmig_TSR(1, 1, h, 0.0, 0.0, 6, cc[j]*N^2, 0.0, 0.01, 0.995, 0.0, 0.0, 0.0, 0.0, 50, 50, dd[i], 0.0, 101, 1000, 0, 0.0); #hexmig_TSR(1, 1, 0.3, 0.02, 0.0, 6, 500, 0.0, 0.0, 0.995, 0.0, 0.0, 0.0, 0.0, 30, 30, 0.025, 0.0, 50, 1000, 0, 0.0);  

        wild_total = population_size[1, :, :, :].*( 2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 4, :].^2) + population_size[2, :, :, :].*(2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 4, :].^2);
        drive_total = population_size[1, :, :, :].*( 2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 1, :].^2) + population_size[2, :, :, :].*(2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 1, :].^2);
        
        if drive_total[5, 5, 100] <=0
            #println("here!")
            PD[i, j] = 0
        end
    end
end

01213539713





fem_freq, male_freq, population_size, ~, ~ =  hexmig_TSR(1, 1, h, 0.0, 0.0, 6, 100*N^2, 0.0, 0.01, 0.995, 0.0, 0.0, 0.0, 0.0, 50, 50, 0.2, 0.0, 150, 1000, 0, 0.0); #hexmig_TSR(1, 1, 0.3, 0.02, 0.0, 6, 500, 0.0, 0.0, 0.995, 0.0, 0.0, 0.0, 0.0, 30, 30, 0.025, 0.0, 50, 1000, 0, 0.0);  

wild_total = population_size[1, :, :, :].*( 2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 4, :].^2) + population_size[2, :, :, :].*(2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 4, :].^2);
drive_total = population_size[1, :, :, :].*( 2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 1, :].^2) + population_size[2, :, :, :].*(2 .*male_freq[:, :, 4, :].*male_freq[:, :, 1, :] .+ male_freq[:, :, 1, :].^2);

plot1 = heatmap(drive_total[:, :, 1], c=:viridis, xlabel="x", ylabel="y", title="Time 1")
display(plot1)
plot1 = heatmap(wild_total[:, :, 140], c=:viridis, xlabel="x", ylabel="y", title="Time 140")
display(plot1)
plot1 = heatmap(drive_total[:, :, 140], c=:viridis, xlabel="x", ylabel="y", title="Time 140")
display(plot1)








# Define a truncated normal distribution to ensure noise is positive
function create_truncated_normal(mean, std_dev)
    return Normal(mean, std_dev) #Truncated(Normal(mean, std_dev), 0, Inf)
end

# Function to add positive Gaussian white noise to a signal
function add_positive_gaussian_noise(signal, mean, std_dev)
    dist = create_truncated_normal(mean, std_dev)
    noise = rand(dist, size(signal))
    #println(noise)
    return noise #signal .+ noise
end

function binarize_matrix(matrix, threshold)
    binarized_matrix = zeros(Int, size(matrix))
    binarized_matrix .= matrix .>= threshold
    binarized_matrix = Int.(binarized_matrix)
    
    return binarized_matrix
end

#println(binarized_matrix(N_all[10, :, :, 1], 5))


#size(population_size)
function continuous_time(N0, f0, Cc, dt, T, sigma)

    timed = 0:dt:T;

    N_all       = zeros(size(timed)[1], N, N, 3);
    N_all[1, :, :, 1] = zeros(N, N) .+ N0*f0[1]^2 .+ N0*2*f0[1]*f0[2]  #- rand(-10:10, N, N).*(N0*f0[1]^2/200));
    N_all[1, :, :, 2] = zeros(N, N) #.+ N0*2*f0[1]*f0[2] #- rand(-10:10, N, N).*(N0*2*f0[1]*f0[2]/20));
    N_all[1, :, :, 3] = (zeros(N, N) .+ N0*f0[2]^2) #- rand(-10:10, N, N).*(N0*f0[2]^2/50);
    N_all[1, 10:end, :, 3] = zeros(N-9, N)
    N_all[1, :, 10:end, 3] = zeros(N, N-9)

    ind = 1;

    Cc = Cc;

    tWW = zeros(N, N)
    tWD = zeros(N, N)
    tDD = zeros(N, N)

    for t in dt:dt:T

        t = dt*ind
    
        #println(t)

        ind = round(Int, ind + 1);

        bb = R^2
        df = 1

        Es = ( (-sum(N_all[ind-1, :, :, :], dims=3))./Cc .+ 1)

        if sigma == 0


            dNwwdt = bb.*(0.5.*A.*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 1].*Es + (1-epsilon).*0.25.*B.*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  2].*Es + (1-epsilon).*0.25.*D.*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  2].*Es + (1-epsilon)^2*E.*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  2].*Es/8)/Cc 
            dNwwdt = dNwwdt - df*N_all[ind-1, :, :,  1] + Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) #Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1).*(-N_all[ind-1, :, :,  2]./Cc - N_all[ind-1, :, :,  3]./Cc .+ 1) + Di*N_all[ind-1, :, :,  1].*(laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) + laplacian_2d(N_all[ind-1, :, :,  3], 1, 1))./Cc 
    
            dNwddt = bb.*((1+epsilon)*0.25*B*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  2].*Es + 0.5*C*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  3].*Es + (1+epsilon)*0.25*D*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  1].*Es + (1-epsilon)*(1+epsilon)*E*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  2].*Es/4 + (1-epsilon)*0.25*F*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  3].*Es + (1-epsilon)*0.25*beta*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  2].*Es + 0.5*gamma*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  1].*Es)/Cc 
            dNwddt = dNwddt -df*N_all[ind-1, :, :,  2] + Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) #Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1).*(-N_all[ind-1, :, :,  1]./Cc - N_all[ind-1, :, :,  3]./Cc .+ 1) + Di*N_all[ind-1, :, :,  2].*(laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) + laplacian_2d(N_all[ind-1, :, :,  3], 1, 1))./Cc 
    
            dNdddt = bb.*((1+epsilon)^2*E*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :,  2].*Es/8 + (1+epsilon)*0.25*F*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  3].*Es + 0.5*alpha*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  3].*Es + (1+epsilon)*0.25*beta*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  2].*Es)/Cc 
            dNdddt = dNdddt - df*N_all[ind-1, :, :,  3] + Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1) #Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1).*(-N_all[ind-1, :, :,  2]./Cc - N_all[ind-1, :, :,  1]./Cc .+ 1) + Di*N_all[ind-1, :, :,  3].*(laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) + laplacian_2d(N_all[ind-1, :, :,  1], 1, 1))./Cc +  rand(N, N).*2

        else

            dNwwdt = bb.*(0.5.*A.*N_all[ind-1, :, :, 1].*N_all[ind-1, :, :, 1].*Es + (1-epsilon).*0.25.*B.*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  2].*Es + (1-epsilon).*0.25.*D.*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  2].*Es + (1-epsilon)^2*E.*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  2].*Es/8)/Cc 
            dNwwdt = dNwwdt - df*N_all[ind-1, :, :,  1] + Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) #Di*laplacian_2d(N_all[ind-1, :, :,  1], 1, 1).*(-N_all[ind-1, :, :,  2]./Cc - N_all[ind-1, :, :,  3]./Cc .+ 1) + Di*N_all[ind-1, :, :,  1].*(laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) + laplacian_2d(N_all[ind-1, :, :,  3], 1, 1))./Cc 
            dNwwdt = dNwwdt + sigma*add_positive_gaussian_noise(dNwwdt, 0, sigma).*(N_all[ind-1, :, :, 1]) #N_all[ind-1, :, :, 1])

            dNwddt = bb.*((1+epsilon)*0.25*B*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  2].*Es + 0.5*C*N_all[ind-1, :, :,  1].*N_all[ind-1, :, :,  3].*Es + (1+epsilon)*0.25*D*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  1].*Es + (1-epsilon)*(1+epsilon)*E*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  2].*Es/4 + (1-epsilon)*0.25*F*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  3].*Es + (1-epsilon)*0.25*beta*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  2].*Es + 0.5*gamma*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  1].*Es)/Cc 
            dNwddt = dNwddt -df*N_all[ind-1, :, :,  2] + Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) #Di*laplacian_2d(N_all[ind-1, :, :,  2], 1, 1).*(-N_all[ind-1, :, :,  1]./Cc - N_all[ind-1, :, :,  3]./Cc .+ 1) + Di*N_all[ind-1, :, :,  2].*(laplacian_2d(N_all[ind-1, :, :,  1], 1, 1) + laplacian_2d(N_all[ind-1, :, :,  3], 1, 1))./Cc 
            dNwddt = dNwddt + sigma*add_positive_gaussian_noise(dNwddt, 0, sigma).*(N_all[ind-1, :, :, 2]) #N_all[ind-1, :, :, 2])

            dNdddt = bb.*((1+epsilon)^2*E*N_all[ind-1, :, :, 2].*N_all[ind-1, :, :,  2].*Es/8 + (1+epsilon)*0.25*F*N_all[ind-1, :, :,  2].*N_all[ind-1, :, :,  3].*Es + 0.5*alpha*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  3].*Es + (1+epsilon)*0.25*beta*N_all[ind-1, :, :,  3].*N_all[ind-1, :, :,  2].*Es)/Cc 
            dNdddt = dNdddt - df*N_all[ind-1, :, :,  3] + Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1) #Di*laplacian_2d(N_all[ind-1, :, :,  3], 1, 1).*(-N_all[ind-1, :, :,  2]./Cc - N_all[ind-1, :, :,  1]./Cc .+ 1) + Di*N_all[ind-1, :, :,  3].*(laplacian_2d(N_all[ind-1, :, :,  2], 1, 1) + laplacian_2d(N_all[ind-1, :, :,  1], 1, 1))./Cc +  rand(N, N).*2
            dNdddt = dNdddt + sigma*add_positive_gaussian_noise(dNdddt, 0, sigma).*(N_all[ind-1, :, :, 3])#N_all[ind-1, :, :, 3])
        end
        # Euler's method

        N_all[ind, :, :, 1] = N_all[ind-1, :, :, 1] + dt.*dNwwdt #+ rand(N, N).*10
        N_all[ind, :, :, 2] = N_all[ind-1, :, :, 2] + dt.*dNwddt #+  rand(N, N).*10
        N_all[ind, :, :, 3] = N_all[ind-1, :, :, 3] + dt.*dNdddt #+  rand(N, N).*1


        for i in 1:N
            for j in 1:N

                if N_all[ind, i, j, 1] < 0
                    N_all[ind, i, j, 1] = 0
                end

                if N_all[ind, i, j, 2] < 0
                    N_all[ind, i, j, 2] = 0
                end

                if N_all[ind, i, j, 3] < 0
                    N_all[ind, i, j, 3] = 0
                end

                # if sum(N_all[ind, i, j, :]) > Cc
                #     N_all[ind, i, j, :] = N_all[ind-1, i, j, :]
                # end
            end 
        end

        for n in 1:N
            for m in 1:N

                
                if 0.1 < N_all[ind, n, m, 1] < 1 && tWW[n, m] == 0

                    tWW[n, m] = 1

                    for k in 1:N
                        for l in 1:N
                            r = (n-k)^2 + (m-l)^2

                            for Tt=ind:size(timed)[1]
                                
                                N_all[Tt, k, l, 1] = N_all[Tt, k, l, 1] + 0.9*sqrt(Cc)*exp(-df*(Tt*dt-ind*dt+dt))*exp(-r/(4*(Tt*dt-ind+dt+dt)))/sqrt(2*(Tt*dt-ind*dt+dt))
                            end

                        end
                    end
                end
    

                if 0.1 < N_all[ind, n, m, 2] < 1 && tWD[n, m] == 0 

                    for k in 1:N
                        for l in 1:N
                            r = (n-k)^2 + (m-l)^2

                            for Tt=ind:size(timed)[1]
                                
                                N_all[Tt, k, l, 2] = N_all[Tt, k, l, 2] + 0.1*sqrt(Cc)*exp(-df*(Tt*dt-ind*dt+dt))*exp(-r/(4*(Tt*dt-ind+dt+dt)))/sqrt(2*(Tt*dt-ind*dt+dt))
                            end

                        end
                    end
                    tWD[n , m] = 1
                end


                if 0.1 < N_all[ind, n, m, 3] < 1 && tDD[n, m] == 0

                    tDD[n, m] = 1

                    for k in 1:N
                        for l in 1:N
                            r = (n-k)^2 + (m-l)^2

                            for Tt=ind:size(timed)[1]
                                
                                N_all[Tt, k, l, 3] = N_all[ind, k, l, 3] + 0.1*sqrt(Cc)*exp(-df*(Tt*dt-ind*dt+dt))*exp(-r/(4*(Tt*dt-ind*dt+dt)))/sqrt(2*(Tt*dt-ind*dt+dt))
                            end

                        end
                    end
                end
    
            end
        end

        # println(tWW)
        # println(tWD)
        # println(tDD)


    end
    

    return N_all
end



# N_all      = continuous_time(N0, f0, Cc, dt, 150, 2)
# h5write("Nalldat3.h5", "dats", N_all)



####CONTINUOUS
N_all      = continuous_time(N0, f0, Cc, dt, 150, 0)

###COMPARISION

time = 0 #round(Int, 45*(1/dt)) 

for m=0:1:10

    plot1 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW")
    plot2 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m")
    plot3 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD")

    # Combine the plots into a single plot with a layout
    total1 = plot(plot1, plot2, plot3, layout=(2, 2))
    display(total1)
end
# m = 4
# plot4 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW", clims=(0, Cc))
# plot5 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m", clims=(0, Cc))
# plot6 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD", clims=(0, Cc))

# # Combine the plots into a single plot with a layout
# total2 = plot(plot4, plot5, plot6, layout=(2, 2))
# display(total2)

# m = 10
# plot4 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW", clims=(0, Cc))
# plot5 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m", clims=(0, Cc))
# plot6 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD", clims=(0, Cc))

# # Combine the plots into a single plot with a layout
# total2 = plot(plot4, plot5, plot6, layout=(2, 2))
# display(total2)

# m = 20
# plot4 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 1]), c=:viridis, xlabel="x", ylabel="y", title="WW", clims=(0, Cc))
# plot5 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 2]), c=:viridis, xlabel="x", ylabel="y", title="WD, T=$m", clims=(0, Cc))
# plot6 = heatmap(ceil.(N_all[round(Int, m*(1/dt))+1, :, :, 3]), c=:viridis, xlabel="x", ylabel="y", title="DD", clims=(0, Cc))

# # Combine the plots into a single plot with a layout
# total2 = plot(plot4, plot5, plot6, layout=(2, 2))
# display(total2)


#sleep(1)


#h5write("Nalldat.h5", "dats", N_all)
