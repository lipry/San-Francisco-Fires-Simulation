import numpy as np

def KS_stats(obsv, F):
    f_values = np.array([F(y) for y in np.sort(obsv)])
    n = len(obsv)+1
    i = np.arange(1, n)
    return np.amax(np.array([np.amax((i/n)-f_values), np.amax(f_values-((i-1)/n))]))

if __name__ == "__main__":
    D = KS_stats(np.array([66,72,81,94,112,116,124,140,145,155]), lambda x: 1-(np.e**(-x/100)))
    print(D)



