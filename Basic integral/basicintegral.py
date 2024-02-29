import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
def randomPointInterval(lowerbound,upperbound):
    random_point=np.random.uniform(0,1)
    return lowerbound+random_point*(upperbound-lowerbound)

def f(x):
    return 4*np.sqrt(1-x**2)

def mcintegration(lowerbound,upperbound,N):
    total_sum=0
    for t in range(N):
        total_sum=total_sum+f(randomPointInterval(lowerbound,upperbound))

    total_sum=(upperbound-lowerbound)*total_sum/N
    return total_sum

lowerbound=0
upperbound=1
N=10000
number_of_experiments = 100
real_result = integrate.quad(lambda x: f(x), lowerbound, upperbound)
if number_of_experiments==1:
    approximation = mcintegration(lowerbound,upperbound,N)
    print("Approximated result is:", approximation)
    print("Real value, calculated by numpy.integration:", real_result)
    print("Error:" , abs(real_result[0]-approximation))
    print("Error rate: ", 100*(abs(real_result[0]-approximation)/real_result[0]))
else:
    result_array=[]
    error_array=[]
    error_rate_array=[]
    for x in range(number_of_experiments):
        result_array.append(mcintegration(lowerbound,upperbound,N))
        error_array.append(abs(real_result[0]-result_array[x]))
        error_rate_array.append(100*(abs(real_result[0]-result_array[x])/real_result[0]))
    
    average_result= sum(result_array)/number_of_experiments

#experiments' plot
plt.plot(range(number_of_experiments), result_array, color = 'b', linewidth=0.3)
#real result plot
plt.axhline(y=real_result[0],color = 'r', linestyle ='dashed')
#average result plot
plt.axhline(y=average_result,color = 'g', linestyle ='dashed')
print("Minimum error:", min(error_array))
print("Maximum error:", max(error_array))
print("Average result:", average_result)
print("Average error", abs(real_result[0]-average_result))
print("Average error rate:", 100*(abs(real_result[0]-average_result)/real_result[0]))
plt.show()



