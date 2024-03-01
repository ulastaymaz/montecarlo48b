import numpy as np
from scipy.stats import qmc

#calculating the integrals over [0,1]^s
#where s is the dimension parameter, can be changed by the user

#it produces 2^N points on every dimension, following sobol sequence

#f(x_1,...,x_n) = (x_1 ^2 + ... + x_n ^2)^(1/2) is the function in the comments
#I used f(x,y,r,t) = cos(t*x*(r^2)) * (sin(x*y))^r
def f(x):
    sum=0
    #for t in x:
    #    sum = sum + t**2
    return np.cos(x[3]*x[0]*(x[2]**2))*np.sin(x[0]*x[1])**x[2]

#regular MC method
def regularmc(dimension, sample_power_2):
    total_sum = 0
    for r in range(2**sample_power_2):
        vector_x = []
        for t in range(dimension):
            vector_x.append(np.random.uniform(0,1))
        
        total_sum = total_sum + f(vector_x)
    
    return total_sum/(2**sample_power_2)

#Quasi MC method
def quasimc(dimension, sample_power_2, sobolseq):
    i = 0
    total_sum=0
    for x in range(2**sample_power_2):
        total_sum = total_sum + f(sobolseq[x])

    return total_sum / (2**sample_power_2)


#dimension of the function, how many variable it has?
dimension = 4
#For Sobol sequence, choosing 2^N sample point is advised
#sample_power_2 is the N variable above
sample_power_2 = 14

#Sample Sobol sequence
sampler = qmc.Sobol(d = dimension)
sample = sampler.random_base2(m = sample_power_2)

#error approximately 3 * 10^(-4)
print("MC", regularmc(dimension, sample_power_2))
#error approximately 8 * 10^(-7)
print("QMC", quasimc(dimension,sample_power_2, sample))

