using LinearAlgebra
using Plots
using MAT
using Noise
using Random
using Distributions

# Discrete Model Constants
epsilon = 0.995                # Efficiency of drive cleavage per target site
s       = 1                  # Homozgous fitness cost of drive 
h       = 0.3                   # h*s = fitness cost of heterozygous W/D individual
Wf      = [1 1-h*s; 1-h*s 1-s]  # Female fitness matrix
Wm      = [1 1; 1 1]            # Male fitness matrix
R       = 6                     # Growth rate
f0      = [0.9991, 0.0009]           # Total starting frequency for W and D alleles
N0      = 100;                  # Total starting population
Cc      = 100;                  # Carrying capacity
alpha2  = Cc/(R-1)              # Discrete model carrying capacity

dt = 0.001 # Time step for contnous time model
T  = 30    # Number of generations simulated for

# Define a truncated normal distribution to ensure noise is positive
function create_truncated_normal(mean, std_dev)
    return Normal(mean, std_dev) #Truncated(Normal(mean, std_dev), 0, Inf)
end

function create_truncated_normal2(mean, std_dev)
    return Truncated(Normal(mean, std_dev), 0, Inf)
end

# Function to add positive Gaussian white noise to a signal
function add_positive_gaussian_noise(signal, mean, std_dev)
    dist = create_truncated_normal(mean, std_dev)
    noise = rand(dist, size(signal))
    return signal .+ noise
end

function add_positive_gaussian_noise2(signal, mean, std_dev)
    dist = create_truncated_normal2(mean, std_dev)
    noise = rand(dist, size(signal))
    return signal .+ noise
end

function continuous_time(N0, f0, Cc, dt, T, type)

    timed = 0:dt:T;

    N_all       = zeros(size(timed)[1], 3);
    N_all[1, 1] = N0*f0[1]^2;
    N_all[1, 2] = N0*2*f0[1]*f0[2]
    N_all[1, 3] = N0*f0[2]^2;

    ind = 1;

    for t in dt:dt:T

        ind = ind + 1;

        if type == "cubic"

            df = 1
            bb = R^2

            Es = (1- (sum(N_all[ind-1, :]))/Cc) #number of empty spaces

            A = 1
            B = 1
            C = 1
            D = 1-h*s
            E = 1-h*s
            F = 1-h*s
            alpha1 = (1-s)
            beta  = (1-s)
            gamma = (1-s)

            #Combined Birth and Drive:
            
            dNwwdt = bb*(0.5*A*N_all[ind-1, 1]*N_all[ind-1, 1]*Es + (1-epsilon)*0.25*(D+B)*N_all[ind-1, 1]*N_all[ind-1, 2]*Es + (1-epsilon)^2*E*N_all[ind-1, 2]*N_all[ind-1, 2]*Es/8)/Cc - df*N_all[ind-1, 1]
            dNwddt = bb*((1+epsilon)*0.25*(B+D)*N_all[ind-1, 1]*N_all[ind-1, 2]*Es + 0.5*(C+gamma)*N_all[ind-1, 1]*N_all[ind-1, 3]*Es + (1-epsilon)*(1+epsilon)*E*N_all[ind-1, 2]*N_all[ind-1, 2]*Es/4 + (1-epsilon)*0.25*(F+beta)*N_all[ind-1, 2]*N_all[ind-1, 3]*Es)/Cc - df*N_all[ind-1, 2]
            dNdddt = bb*((1+epsilon)^2*E*N_all[ind-1, 2]*N_all[ind-1, 2]*Es/8 + (1+epsilon)*0.25*(F+beta)*N_all[ind-1, 2]*N_all[ind-1, 3]*Es + 0.5*alpha1*N_all[ind-1, 3]*N_all[ind-1, 3]*Es)/Cc - df*N_all[ind-1, 3]
        
            # dNwwdt = add_positive_gaussian_noise(dNwwdt, 0, 10)
            # dNwddt = add_positive_gaussian_noise(dNwddt, 0, 10)
            # dNdddt = add_positive_gaussian_noise(dNdddt, 0, 10)

            # Euler's method
            N_all[ind, 1] = N_all[ind-1, 1] + dt * dNwwdt
            N_all[ind, 2] = N_all[ind-1, 2] + dt * dNwddt
            N_all[ind, 3] = N_all[ind-1, 3] + dt * dNdddt

            
            # if N_all[ind, 1] <= 0
            #     N_all[ind, 1] = 0
            # else
            #     N_all[ind, 1] = add_positive_gaussian_noise(N_all[ind, 1], 0, (1/Cc^2)/(sqrt(N_all[ind, 1])))
            #     if N_all[ind, 1] <= 0
            #         N_all[ind, 1] = 0
            #     end

            # end

            # if N_all[ind, 2] <= 0
            #     N_all[ind, 2] = 0
            # else
            #     N_all[ind, 2] = add_positive_gaussian_noise(N_all[ind, 2], 0, (1/Cc^2)/(sqrt(N_all[ind, 2])))
            #     if N_all[ind, 2] <= 0
            #         N_all[ind, 2] = 0
            #     end
            # end

            # if N_all[ind, 3] <= 0
            #     N_all[ind, 3] = 0
            # else
            #     N_all[ind, 3] = add_positive_gaussian_noise(N_all[ind, 3], 0, (1/Cc^2)/(sqrt(N_all[ind, 3])))
            #     if N_all[ind, 3] <= 0
            #         N_all[ind, 3] = 0
            #     end
            # end


            # N_all[ind, 1] = add_positive_gaussian_noise(N_all[ind, 1], 0, 1/(sqrt(N_all[ind, 1]))) #1/(N_all[ind, 1]*100))
            # #println(add_positive_gaussian_noise(N_all[ind, 1], 1, 10) - N_all[ind, 1])
            # N_all[ind, 2] = add_positive_gaussian_noise(N_all[ind, 2], 0, 1/(sqrt(N_all[ind, 2]))) #1/(N_all[ind, 2]*100))
            # N_all[ind, 3] = add_positive_gaussian_noise(N_all[ind, 3], 0, 1/(sqrt(N_all[ind, 3]))) #1/(N_all[ind, 3]*100))


            # end
            # if N_all[ind, 1] <= 0
            #     N_all[ind, 1] = 0
            # end

            # if N_all[ind, 2] <= 0
            #     N_all[ind, 2] = 0
            # end

            # if N_all[ind, 3] <= 0
            #     N_all[ind, 3] = 0
            # end

        # Seperated Birth and Drive
        else
            Es = (1- (sum(N_all[ind-1, :]))/Cc) #number of empty spaces
            A = R
            B = R
            df = 1

            dNwwdt = (0.5*A*N_all[ind-1, 1]*Cc*Es + (1-epsilon)*0.25*B*N_all[ind-1, 1]*N_all[ind-1, 2]*Es + (1-epsilon)^2*B*N_all[ind-1, 2]*N_all[ind-1, 2]*Es/8)/Cc - df*N_all[ind-1, 1]
            dNwddt = (A*(1-h*s)*N_all[ind-1, 2]*Cc*Es/2 + (1+epsilon)*0.25*B*N_all[ind-1, 1]*N_all[ind-1, 2]*Es + 0.5*B*N_all[ind-1, 1]*N_all[ind-1, 3]*Es + (1-epsilon)*(1+epsilon)*B*N_all[ind-1, 2]*N_all[ind-1, 2]*Es/4 + (1-epsilon)*0.25*B*N_all[ind-1, 2]*N_all[ind-1, 3]*Es)/Cc - df*N_all[ind-1, 2]
            dNdddt = ((1+epsilon)^2*B*N_all[ind-1, 2]*N_all[ind-1, 2]*Es/8 + (1+epsilon)*0.25*B*N_all[ind-1, 2]*N_all[ind-1, 3]*Es + (1-s)*0.5*A*Cc*N_all[ind-1, 3]*Es)/Cc - df*N_all[ind-1, 3]

            # Euler's method
            N_all[ind, 1] = N_all[ind-1, 1] + dt * dNwwdt
            N_all[ind, 2] = N_all[ind-1, 2] + dt * dNwddt
            N_all[ind, 3] = N_all[ind-1, 3] + dt * dNdddt

        end


        # #Predator Prey
        # A = R
        # B = 0.5

        # dNwwdt = (A*N_all[ind-1, 1] - B*epsilon*N_all[ind-1, 1]*N_all[ind-1, 2]/Cc)*Es - df*N_all[ind-1, 1]
        # dNwddt = (A*(1-h*s)*N_all[ind-1, 2] + B*epsilon*N_all[ind-1, 1]*N_all[ind-1, 2]/Cc - B*epsilon*N_all[ind-1, 2]*N_all[ind-1, 3]/Cc)*Es - df*N_all[ind-1, 2]
        # dNdddt = (A*(1-s)*N_all[ind-1, 3] + B*epsilon*N_all[ind-1, 2]*N_all[ind-1, 3]/Cc)*Es - df*N_all[ind-1, 3]
      

        # # Euler's method
        # N_all[ind, 1] = N_all[ind-1, 1] + dt * dNwwdt
        # N_all[ind, 2] = N_all[ind-1, 2] + dt * dNwddt
        # N_all[ind, 3] = N_all[ind-1, 3] + dt * dNdddt

        # N_all[ind, 1] = add_positive_gaussian_noise(N_all[ind, 1], 1, 10)
        # N_all[ind, 2] = add_positive_gaussian_noise(N_all[ind, 2], 1, 10)
        # N_all[ind, 3] = add_positive_gaussian_noise(N_all[ind, 3], 1, 10)


    end

    return N_all
end


####CONTINUOUS
N_all      = continuous_time(N0, f0, Cc, dt, T, "cubic")

####DISCRETE
fem  = zeros(T+1, 2);
male = zeros(T+1, 2);
fem[1, :]  = f0
male[1, :] = f0

pop_dis    = zeros(T+1)
pop_dis[1] = N0

geno_pop = zeros(T+1, 6)
geno_pop[1, 1] = f0[1]^2*N0/2
geno_pop[1, 2] = 2*f0[1]*f0[2]*N0/2
geno_pop[1, 3] = f0[2]^2*N0/2

geno_pop[1, 4] = f0[1]^2*N0/2
geno_pop[1, 5] = 2*f0[1]*f0[2]*N0/2
geno_pop[1, 6] = f0[2]^2*N0/2


for i in 2:T+1

    meanf = transpose(fem[i-1, :])*Wf*male[i-1, :];
    meanm = transpose(male[i-1, :])*Wm*fem[i-1, :];

    Ht = fem[i-1, 1]*male[i-1, 2] + male[i-1, 1]*fem[i-1, 2]
    
    fem[i, 1]  = ((Wf*male[i-1, :])[1]*fem[i-1, 1] + (Wf*fem[i-1, :])[1]*male[i-1, 1])/(2*meanf) - Ht*epsilon*(1-h*s)/(2*meanf)
    fem[i, 2]  = ((Wf*male[i-1, :])[2]*fem[i-1, 2] + (Wf*fem[i-1, :])[2]*male[i-1, 2])/(2*meanf) + Ht*epsilon*(1-h*s)/(2*meanf)
    male[i, 1] = ((Wm*fem[i-1, :])[1]*male[i-1, 1] + (Wm*male[i-1, :])[1]*fem[i-1, 1])/(2*meanm) - Ht*epsilon/(2*meanm)
    male[i, 2] = ((Wm*fem[i-1, :])[2]*male[i-1, 2] + (Wm*male[i-1, :])[2]*fem[i-1, 2])/(2*meanm) + Ht*epsilon/(2*meanm)

    Nmalenew = R*meanm*sum(geno_pop[i-1, 1:3])/(1 + sum(geno_pop[i-1, :])/alpha2); 
    Nfnew    = R*meanf*sum(geno_pop[i-1, 1:3])/(1 + sum(geno_pop[i-1, :])/alpha2);

    # fem[i, 1] = add_positive_gaussian_noise2(fem[i, 1], 0, 0.1)
    # fem[i, 2] = add_positive_gaussian_noise2(fem[i, 2], 0, 0.1)
    # male[i, 1] = add_positive_gaussian_noise2(male[i, 1], 0, 0.1)
    # male[i, 2] = add_positive_gaussian_noise2(male[i, 2], 0, 0.1)


    geno_pop[i, 1] = (fem[i, 1]^2)*Nfnew
    geno_pop[i, 2] = (2*fem[i, 1]*fem[i, 2])*Nfnew
    geno_pop[i, 3] = (fem[i, 2]^2)*Nfnew
    geno_pop[i, 4] = (male[i, 1]^2)*Nmalenew
    geno_pop[i, 5] = (2*male[i, 1]*male[i, 2])*Nmalenew
    geno_pop[i, 6] = (male[i, 2]^2)*Nmalenew

end

###COMPARISION

x_axis  = 0:T
x_axis2 = 0:dt:T

plot1 = plot(x_axis2, N_all[:, 1] , label ="Continuous", title="WW, epsilon=$epsilon" , linewidth=3, ylims=(0, 10)) #yscale=:log10, ylims =(1e-6, 1e6))
plot!(x_axis, geno_pop[:, 1] + geno_pop[:, 4] , label="Discrete", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot1)

plot2=plot(x_axis2, N_all[:, 2] , label ="Continuous", title="WD, epsilon=$epsilon", linewidth=3) #yscale=:log10, ylims =(1e-6, 1e6))
plot!(x_axis, geno_pop[:, 2] + geno_pop[:, 5] , label="Discrete", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot2)

plot3=plot(x_axis2, N_all[:, 3] , label ="Continuous", title="DD, epsilon=$epsilon", linewidth=3) #yscale=:log10, ylims =(1e-6, 1e6))
plot!(x_axis, geno_pop[:, 3] + geno_pop[:, 6] , label="Discrete", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot3)


# Combine the plots into a single plot with a layout
plot(plot1, plot2, plot3, layout=(2, 2))

