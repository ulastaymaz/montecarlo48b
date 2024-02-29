import random
import numpy as np
from scipy.special import gamma, factorial
def getpoints(p):
    coordinates=[]
    for x in range(p):
        #in every dimension, our interval will be [-1,1]
        #positive or negative determines whether we will be on [-1,0] or [0,1]
        positive_or_negative = np.random.randint(1,3)
        #uniformly get points between [0,1]
        abs_distance = np.random.uniform(0,1)
        if positive_or_negative == 1:
            coordinates.append(abs_distance)
        else:
            coordinates.append((-1)*abs_distance)
    return coordinates

def isInSphere(coordinates):
    #distance of our points to the origin
    dist=0
    for x in range(len(coordinates)):
        dist = dist+coordinates[x]**2


    if dist > 1:
        return 0
    else:
        return 1
    
#set dimension p
p=10
#real values of hypersphere and hypercube
real_volume_sphere = (np.pi**(p/2))/(gamma((p/2)+1))
real_volume_cube = 2**p

#number of points in sphere
count_sphere = 0
#number of choosing a random point
test_number = 100000

#test the random points
for t in range(test_number):
    coordinates = getpoints(p)
    is_in_sphere = isInSphere(coordinates)

    if is_in_sphere == 1:
        count_sphere = count_sphere+1
    
#this approximation is directly from the paper included in README
appr_vol = real_volume_cube * count_sphere/test_number



print("Error is:", abs(appr_vol - real_volume_sphere))
print("Error rate is:", 100*abs((appr_vol-real_volume_sphere)/real_volume_sphere))


    