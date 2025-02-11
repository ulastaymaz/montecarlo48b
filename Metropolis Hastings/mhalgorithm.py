import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.stats as scistat
import matplotlib.pyplot as plt

#it is only for computation of the correctness of our estimation
#MCMC Metropolis-Hastings method is useful when normalisation constant is complicated
def normalisationConstant(x):
    return np.exp(-(x**2))*(2+np.sin(5*x)+np.sin(2*x))

#this function is our simplicifation of the distrubition that we want to sample
#distribution probability = simpleProb/normalisationConstant
def simpleProb(x):
    return np.exp(-(x**2))*(2+np.sin(5*x)+np.sin(2*x))

#define the proposed probability
def propProbability(mean, point, sigma):
    return scistat.norm(mean,sigma).pdf(point)

#define the proposed pdf
def proposePoint(x_prev, sigma):
    return np.random.normal(x_prev,sigma)

#computation of acceptance probability, results int [0,1]
#the return in comment is the correct one but the below return also true if our proposed function, or markov chain is reflective (detailed balance)
def acceptanceProb(x_prev, x_new, sigma):
    #return min(1, simpleProb(x_new)*propProbability(x_new,x_prev,sigma)/(simpleProb(x_prev)*propProbability(x_prev,x_new,sigma)))
    return min(1,simpleProb(x_new)/simpleProb(x_prev))

def sampler(x_init, sigma):
    prop_point = proposePoint(x_init,sigma)
    acceptence_rate = acceptanceProb(x_init,prop_point,sigma)
    acceptance_probability = np.random.uniform(0,1)
    if acceptance_probability <= acceptence_rate:
        return prop_point
    else:
        return x_init

sigma = 1
samples = [0]

#number of columns in graph
number_of_bins = 200

#estimated (or guessed) standart_deviation of proposed distrubition
standart_deviation = 1


for t in range(1000000):
    samples.append(sampler(samples[t], standart_deviation))
    
samples_without_burn_in = samples[1000:1000000]
n, bins, patches = plt.hist(samples_without_burn_in, number_of_bins, density = 1, color ='blue',alpha = 0.7)

normalisation_constant_computed = integrate.quad(lambda x: normalisationConstant(x),-np.inf,np.inf)

y=simpleProb(bins)/normalisation_constant_computed[0]


plt.plot(bins,y,'--',color='red')

plt.show()


