import numpy as np

def normal(x, mean, std_deviation):
    #calculates the normal distribution
    return float((1/(np.pow(2*np.pi,0.5)*std_deviation))*np.exp(-0.5*((x-mean)/std_deviation)**2))

#print(normal_distribution(1.5,0.5,3))