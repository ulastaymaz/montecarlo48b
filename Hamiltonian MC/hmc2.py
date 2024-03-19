import autograd.numpy as np
from scipy import interpolate
import autograd as ad
import matplotlib.pyplot as plt
from autograd import grad
from scipy.stats import norm
import math
import time
from scipy.stats import multivariate_normal

def covarianceMatrixM(diagonal_array):
    arr = np.identity(len(diagonal_array))
    for x in range(len(diagonal_array)):
        arr[x][x] = diagonal_array[x]
    
    return arr

def func(x):
    return np.matmul(x,np.transpose(x))*x[0]
#it will work for every q_i in q=( q_0, ... , q_(n-1) )
#mainly depends on https://colindcarroll.com/2019/04/11/hamiltonian-monte-carlo-from-scratch/
#p_i and q_i are in R, q in R^n
#gradU = grad(U,i-1), i^th variable's derivative
def leapfrog(p_i, q_i, q, gradU, path_length, step_size, i):
    q_i, p_i = np.copy(q_i), np.copy(p_i)

    p_i = p_i - step_size * gradU(q)[i] / 2  # half step
    for _ in range(int(path_length / step_size) - 1):
        q_i = q_i + step_size * p_i  # whole step
        p_i -= step_size * gradU(q)[i]  # whole step
    q_i = q_i+ step_size * p_i  # whole step
    p_i = p_i- step_size * gradU(q)[i] / 2  # half step

    # momentum flip at end
    return q_i, -p_i


#its only a placeholder
#bunu düzelt
def logpdf(x, inverse_M, det_inv_M, mean):
    return -np.log(normal_pdf(x, inverse_M, det_inv_M, mean))
    #return -((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2) / 20

def normal_pdf(x, inverse_M, det_inv_M, mean):
    temp1 = np.matmul(np.transpose(x-mean),(np.matmul(inverse_M,(x-mean))))
    return np.exp(-temp1/2)/(math.pow(2*np.pi,len(x)/2)*np.sqrt(det_inv_M))

def hmc_sample(U_1, current_q, path_length, step_size, n_samples, burn_in):
    U= lambda x: U_1(x)
    dimension = len(current_q)
    q=np.copy(current_q)
    total_samples = n_samples + burn_in
    samples = np.array([current_q])

    for t in range(total_samples):
        p=norm.rvs(size=dimension)
        current_p = p
        for i in range(dimension):
            gradU = grad(U,i)
            #deneme
            q[i],p[i] = leapfrog(p[i], q[i], samples[t], gradU, path_length, step_size)
        
        #kinetic energy function K can be changed
        current_U = U(samples[t])
        current_K = 0.5*sum(current_p**2)
        proposed_U = U(q)
        proposed_K = 0.5*sum(p**2)

        #should we choose the point q we've obtained?
        #acceptance = min(1, np.exp(current_U-proposed_U+current_K-proposed_K))
        acceptance = np.exp(current_U-proposed_U+current_K-proposed_K)
        #uniform random variable for acceptance
        accept_probability = np.random.uniform(0,1)

        if acceptance >= accept_probability:
            samples=np.append(samples,q)
        else:
            samples=np.append(samples,samples[t])
    
    return samples

def new_hmc_sample(U_1, inverse_M, det_inv_M, mean, current_q, path_length, step_size, n_samples, burn_in):
    U= lambda x: U_1(x, inverse_M, det_inv_M, mean)
    dimension = len(current_q)
    q=np.copy(current_q)
    total_samples = n_samples + burn_in
    samples = np.array([current_q])

    for t in range(total_samples):
        ld =0
        p=norm.rvs(size=dimension)
        current_p = p
        gradU = grad(U)
        for i in range(dimension):
            #last_sample = np.transpose(samples[(ld*dimension):(ld*dimension)+dimension])
            #deneme
            q[i],p[i] = leapfrog(p[i], q[i], samples[t], gradU, path_length, step_size, i)
        
        #kinetic energy function K can be changed
        current_U = U(samples[t])
        current_K = 0.5*sum(current_p**2)
        proposed_U = U(q)
        proposed_K = 0.5*sum(p**2)

        ld = ld+1
        #should we choose the point q we've obtained?
        #acceptance = min(1, np.exp(current_U-proposed_U+current_K-proposed_K))
        acceptance = np.exp(current_U-proposed_U+current_K-proposed_K)
        #uniform random variable for acceptance
        accept_probability = np.random.uniform(0,1)

        if acceptance >= accept_probability:
            #samples = samples + q
            samples=np.append(samples,[q],axis=0)
        else:
            #samples = samples + samples[t]
            samples=np.append(samples,[samples[t]],axis=0)
    
    return samples


def grad_f(x):
  # Create a gradient function using autograd.grad
  grad_func = grad(func)
  # Calculate the gradients directly using the NumPy array
  gradients = grad_func(x)
  return gradients


#hop = lambda x: -math.log(math.sqrt(math.pi*2)*math.exp(-0.5*(x**2)))
start_time = time.time()
q = np.array([0.0, 0.0])

diagonal_of_matrix = np.array([1.0,1.0])
M=covarianceMatrixM(diagonal_of_matrix)
mean_tuple=np.array([0.0, 0.0])
inverse_M=np.linalg.inv(M)
det_inv_M=np.linalg.det(inverse_M)
samples = new_hmc_sample(logpdf, inverse_M, det_inv_M, mean_tuple, q, 0.1, 0.01, 10000, 10)
x_1=[]
y_1=[]





###########################################






# Example usage


# Print the partial derivatives
#print("Partial derivatives of f(x) with respect to each element of x:")
#print(partial_derivatives)


#h=plt.contourf(x_1,y_1,Z)
#plt.axis('scaled')

#plt.plot(R,marker ='o', color = 'k', linestyle = 'none')
#plt.show()
#deneme1=normal_pdf(Z,inverse_M,det_inv_M,mean_tuple)



#print(type(logpdf))
#for N=100.000, it took 96.27 sec
# Create a function to compute the gradient
#grad_func = grad(func,1)
#print(type(grad_func))
# Evaluate the gradient at x = 1.0
#print(grad_func(1.0,2.0,3.0))

