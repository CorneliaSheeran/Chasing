using LinearAlgebra
using Plots


# Discrete Model Constants
epsilon = 0.995               # Efficiency of drive cleavage per target site
s       = 0.95                  # Homozgous fitness cost of drive 
h       = 0.3                   # h*s = fitness cost of heterozygous W/D individual
Wf      = [1 1-h*s; 1-h*s 1-s]  # Female fitness matrix
Wm      = [1 1; 1 1]            # Male fitness matrix
R       = 6                     # Growth rate
f0      = [0.90, 0.1]           # Total starting frequency for W and D alleles
N0      = 1e5;                  # Total starting population
Cc      = 1e6;                  # Carrying capacity
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

dt = 0.001 # Time step for contnous time model
T  = 50    # Number of generations simulated for

function continuous_time(N0, f0, Cc, dt, T)
    # Initial conditions

    timed = 0:dt:T;

    N_all       = zeros(size(timed)[1], 6);
    N_all[1, 1] = N0*f0[1]^2/2;
    N_all[1, 2] = N0*2*f0[1]*f0[2]/2 
    N_all[1, 3] = N0*f0[2]^2/2;
    N_all[1, 4] = N0*f0[1]^2/2;
    N_all[1, 5] = N0*2*f0[1]*f0[2]/2; 
    N_all[1, 6] = N0*f0[2]^2/2;

    N_2 = zeros(size(timed)[1], 4);
    N_2[1, 1] = N0*f0[1]
    N_2[1, 2] = N0*f0[2]
    N_2[1, 3] = N0*f0[1]
    N_2[1, 4] = N0*f0[2]

    ind = 1;

    for t in dt:dt:T

        ind = ind + 1;

        bb = 2*R^2
        df = 1 #0.9
        dm = 1 #1.1
        

        Es = (1- (sum(N_all[ind-1, :]))/Cc)/Cc #number of empty spaces

        ## CORRECT
        dNwwfdt = bb*(0.5*A*N_all[ind-1, 1]*N_all[ind-1, 4]*Es + (1-epsilon)*0.25*B*N_all[ind-1, 1]*N_all[ind-1, 5]*Es + (1-epsilon)*0.25*D*N_all[ind-1, 4]*N_all[ind-1, 2]*Es + (1-epsilon)^2*E*N_all[ind-1, 2]*N_all[ind-1, 5]*Es/8) - df*N_all[ind-1, 1]
        dNwdfdt = bb*((1+epsilon)*0.25*B*N_all[ind-1, 1]*N_all[ind-1, 5]*Es + 0.5*C*N_all[ind-1, 1]*N_all[ind-1, 6]*Es + (1+epsilon)*0.25*D*N_all[ind-1, 2]*N_all[ind-1, 4]*Es + (1-epsilon)*(1+epsilon)*E*N_all[ind-1, 2]*N_all[ind-1, 5]*Es/4 + (1-epsilon)*0.25*F*N_all[ind-1, 2]*N_all[ind-1, 6]*Es + (1-epsilon)*0.25*beta*N_all[ind-1, 3]*N_all[ind-1, 5]*Es + 0.5*gamma*N_all[ind-1, 3]*N_all[ind-1, 4]*Es) - df*N_all[ind-1, 2] #- N_all[ind-1, 2]*delta + N_all[ind-1, 1]*delta
        dNddfdt = bb*((1+epsilon)^2*E*N_all[ind-1, 2]*N_all[ind-1, 5]*Es/8 + (1+epsilon)*0.25*F*N_all[ind-1, 2]*N_all[ind-1, 6]*Es + 0.5*alpha*N_all[ind-1, 3]*N_all[ind-1, 6]*Es + (1+epsilon)*0.25*beta*N_all[ind-1, 3]*N_all[ind-1, 5]*Es) - df*N_all[ind-1, 3]
        
        dNwwmdt = bb*(0.5*A*N_all[ind-1, 1]*N_all[ind-1, 4]*Es + (1-epsilon)*0.25*B*N_all[ind-1, 1]*N_all[ind-1, 5]*Es + (1-epsilon)*0.25*D*N_all[ind-1, 4]*N_all[ind-1, 2]*Es + (1-epsilon)^2*E*N_all[ind-1, 2]*N_all[ind-1, 5]*Es/8) - dm*N_all[ind-1, 4]
        dNwdmdt = bb*((1+epsilon)*0.25*B*N_all[ind-1, 1]*N_all[ind-1, 5]*Es + 0.5*C*N_all[ind-1, 1]*N_all[ind-1, 6]*Es + (1+epsilon)*0.25*D*N_all[ind-1, 2]*N_all[ind-1, 4]*Es + (1+epsilon)*(1-epsilon)*E*N_all[ind-1, 2]*N_all[ind-1, 5]*Es/4 + (1-epsilon)*0.25*F*N_all[ind-1, 2]*N_all[ind-1, 6]*Es + (1-epsilon)*0.25*beta*N_all[ind-1, 3]*N_all[ind-1, 5]*Es + 0.5*gamma*N_all[ind-1, 3]*N_all[ind-1, 4]*Es) - dm*N_all[ind-1, 5] #- N_all[ind-1, 5]*delta + N_all[ind-1, 4]*delta
        dNddmdt = bb*((1+epsilon)^2*E*N_all[ind-1, 2]*N_all[ind-1, 5]*Es/8 + (1+epsilon)*0.25*F*N_all[ind-1, 2]*N_all[ind-1, 6]*Es + 0.5*alpha*N_all[ind-1, 3]*N_all[ind-1, 6]*Es + (1+epsilon)*0.25*beta*N_all[ind-1, 3]*N_all[ind-1, 5]*Es) - dm*N_all[ind-1, 6]

        # Euler's method

        N_all[ind, 1] = N_all[ind-1, 1] + dt * dNwwfdt
        N_all[ind, 2] = N_all[ind-1, 2] + dt * dNwdfdt
        N_all[ind, 3] = N_all[ind-1, 3] + dt * dNddfdt
        N_all[ind, 4] = N_all[ind-1, 4] + dt * dNwwmdt
        N_all[ind, 5] = N_all[ind-1, 5] + dt * dNwdmdt
        N_all[ind, 6] = N_all[ind-1, 6] + dt * dNddmdt


    end

    return N_all
end



####CONTINUOUS
N_all       = continuous_time(N0, f0, Cc, dt, T)

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

#K = [-epsilon, epsilon]

for i in 2:T+1

    meanf = transpose(fem[i-1, :])*Wf*male[i-1, :];
    meanm = transpose(male[i-1, :])*Wm*fem[i-1, :];

    Ht = fem[i-1, 1]*male[i-1, 2] + male[i-1, 1]*fem[i-1, 2]
    
    fem[i, 1] = ((Wf*male[i-1, :])[1]*fem[i-1, 1] + (Wf*fem[i-1, :])[1]*male[i-1, 1])/(2*meanf) - Ht*epsilon*(1-h*s)/(2*meanf)
    fem[i, 2] = ((Wf*male[i-1, :])[2]*fem[i-1, 2] + (Wf*fem[i-1, :])[2]*male[i-1, 2])/(2*meanf) + Ht*epsilon*(1-h*s)/(2*meanf)
    male[i, 1] = ((Wm*fem[i-1, :])[1]*male[i-1, 1] + (Wm*male[i-1, :])[1]*fem[i-1, 1])/(2*meanm) - Ht*epsilon/(2*meanm)
    male[i, 2] = ((Wm*fem[i-1, :])[2]*male[i-1, 2] + (Wm*male[i-1, :])[2]*fem[i-1, 2])/(2*meanm) + Ht*epsilon/(2*meanm)

    pop_dis[i] = 2*R*meanf*sum(geno_pop[i-1, 1:3])/(1 + sum(geno_pop[i-1, :])/alpha2);

    Nmalenew = R*meanm*sum(geno_pop[i-1, 1:3])/(1 + sum(geno_pop[i-1, :])/alpha2); 
    Nfnew = R*meanf*sum(geno_pop[i-1, 1:3])/(1 + sum(geno_pop[i-1, :])/alpha2);

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

plot1 = plot(x_axis2, N_all[:, 1] , label ="Continuous", linewidth=3, title="WW Female")
plot!(x_axis, geno_pop[:, 1] , label="Discrete", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot1)

plot2=plot(x_axis2, N_all[:, 2] , label ="C", linewidth=3, title="WD Female")
plot!(x_axis, geno_pop[:, 2] , label="D", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot2)

plot3=plot(x_axis2, N_all[:, 3] , label ="C", linewidth=3, title="DD Female")
plot!(x_axis, geno_pop[:, 3] , label="D", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot3)

plot4=plot(x_axis2, N_all[:, 4] , label ="C", linewidth=3, title="WW Male")
plot!(x_axis, geno_pop[:, 4] , label="D", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot4)

plot5=plot(x_axis2, N_all[:, 5] , label ="C", linewidth=3, title="WD Male")
plot!(x_axis, geno_pop[:, 5] , label="D", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot5)

plot6=plot(x_axis2,  N_all[:, 6] , label ="C", linewidth=3, title="DD Male")
plot!(x_axis, geno_pop[:, 6] , label="D", linewidth=3)
xlabel!("Time")
ylabel!("Population")
#display(plot6)

# Combine the plots into a single plot with a layout
plot(plot1, plot2, plot3, plot4, plot5, plot6, layout=(3, 3))