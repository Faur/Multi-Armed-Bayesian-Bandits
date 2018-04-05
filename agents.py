import numpy as np
from scipy.stats import beta as beta_dist

class RandomAgent():
    def __init__(self, k):
        self.k = k
    def reset(self):
        pass
    def action(self):
        return np.random.randint(self.k)
    def update(self, *args):
        pass

class OptimalAgent():
    def __init__(self, env):
        self.env = env
    def reset(self):
        pass
    def action(self):
        return self.env.optimal_action()
    def update(self, *args):
        pass


class FreqUCB():
    def __init__(self, k):
        self.k = k
        self.reset()
        
    def reset(self):
        self.steps = 0
        self.means = np.zeros(self.k)
        self.pulls = np.zeros(self.k)
    
    def action(self):
        if self.steps < self.k:
            # Pull each arm once initially
            return self.steps
        
        a = np.argmax(self.means + np.sqrt(2*np.log(self.steps)/self.pulls))
        return a
            
    def update(self, action, reward):
        self.steps += 1
        self.pulls[action] += 1
        self.means[action] = self.means[action]*(self.pulls[action]-1)/(self.pulls[action]) + reward/(self.pulls[action])


class BayesUCB():
    def __init__(self, k, n, c=0):
        self.k = k
        self.n = n
        self.c = c

        self.prior_param = [1,1]

    def reset(self):
        self.t=0
        self.post_param = np.zeros([self.k])
        self.draws = np.zeros([self.k])

    @property
    def values(self):
        quantiles = []
        for i in range(self.k):
            a = self.prior_param[0]+self.post_param[i]
            b = self.prior_param[0]+self.draws[i]-self.post_param[i]
            quantiles.append(beta_dist(a,b).ppf(1-1/(self.t*np.log(self.n)**self.c)))
        return quantiles

    @property
    def theta(self):
        thetas = []
        for i in range(self.k):
            a = self.prior_param[0]+self.post_param[i]
            b = self.prior_param[0]+self.draws[i]-self.post_param[i]
            thetas.append(a/(a+b))
        return thetas

    def action(self):
        self.t += 1
        return np.argmax(self.values)
    
    def update(self, action, reward):
        self.post_param[action] += reward
        self.draws[action] += 1


class ThompsonSampling(BayesUCB):
    @property
    def values(self):
        samples = []
        for i in range(self.k):
            a = self.prior_param[0]+self.post_param[i]
            b = self.prior_param[0]+self.draws[i]-self.post_param[i]
            samples.append(beta_dist(a,b).rvs())
        return samples



