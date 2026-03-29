import numpy as np
import scipy

# Should be move to a util folder
def calc_mean_sem_ci(scores):
    n = len(scores)
    
    mean = np.mean(scores)
    print("MEAN: ", mean)
    
    if n == 1:
        return mean, 0, 0

    sem = scipy.stats.sem(scores)
    print("SEM: ", sem)
    
    # 90% confidence interval
    #p = ppf(0.95)

    # 95% confidence interval
    p = 0.975

    # 99% confidence interval
    #p= ppf(0.995)

    if (n >= 30):
        ci = scipy.stats.norm.ppf(p) * sem
    else:
        ci = scipy.stats.t.ppf(p, df = n-1) * sem
    print("CI: ", ci)

    return mean, sem, ci