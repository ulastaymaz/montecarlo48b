import numpy as np
import matplotlib.pyplot as plt
import math

real_mean = 0.5
real_dev = 0.5

upper_bound=[]
lower_bound=[]

xpoints = list(range(1,1001))

for x in xpoints:
    upper_bound.append(real_mean+3*real_dev/(math.sqrt(x)))
    lower_bound.append(real_mean-3*real_dev/(math.sqrt(x)))

for k in range(1,25):
    mean_calculated=[]
    for N in range(1,1001):
        number_of_success = 0
        for x in range(0,N):
            number_of_games = 24
            dice_rolls1 = np.random.randint(1, 7, number_of_games)
            dice_rolls2 = np.random.randint(1, 7, number_of_games)
            dice_rolls = dice_rolls1+dice_rolls2
            count=0
            for x in dice_rolls:
                if (x==12):
                    count=1

            if count>0:
                number_of_success=number_of_success+1

        ratio_of_success = number_of_success/N

        mean_calculated.append(ratio_of_success)

    plt.plot(xpoints,mean_calculated,'b', linewidth=0.3)

plt.plot(xpoints,lower_bound,'r',linestyle='--')
plt.plot(xpoints,upper_bound,'r',linestyle='--')
plt.show()
