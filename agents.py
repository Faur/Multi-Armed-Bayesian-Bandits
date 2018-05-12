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
        self.n = n # UCB param - Episode length
        self.c = c # UCB param ~ Exploration parameter 

    @property
    def prior_param(self):
        return [1,1]

    def reset(self):
        self.t=0
        self.post_param = np.zeros([self.k])
        self.draws = np.zeros([self.k])

    def compute_ab(self, i):
        a = self.prior_param[0]+self.post_param[i]
        b = self.prior_param[1]+self.draws[i]-self.post_param[i]
        assert a > 0, 'a = {}, prior {}, post {}'.format(a, self.prior_param[0], self.post_param[i])
        assert b > 0, 'b = {}, prior {}, post {}'.format(b, self.prior_param[1], self.draws[i]-self.post_param[i])
        # print(a,b)
        return a, b

    @property
    def values(self):
        quantiles = []
        for i in range(self.k):
            a, b = self.compute_ab(i)
            quantiles.append(beta_dist(a,b).ppf(1-1/(self.t*np.log(self.n)**self.c)))
        return quantiles

    @property
    def theta(self):
        thetas = []
        for i in range(self.k):
            a, b = self.compute_ab(i)
            thetas.append(a/(a+b))
        return thetas

    def action(self):
        self.t += 1
        return np.argmax(self.values)
    
    def update(self, action, reward):
        self.post_param[action] += reward
        self.draws[action] += 1

class HierarchicalBayesUCB(BayesUCB):    
    @property
    def prior_param(self):
        u = np.mean(self.post_param/self.draws)
        v = np.var(self.post_param/self.draws)
        u = u if u != 1 else u+1e-10
        v = v if v != 0 else 1e-10
        assert v > 0, 'Variance is ' + str(v)
        
        b = u*(1-u)**2/v - (1-u)
        a = u*b/(1-u)

        eps = 1e-10
        if a <= eps: a = eps
        if b <= eps: b = eps

        assert b > 0, 'b = {}, u {}, v {}'.format(b, u, v)
        assert a > 0, 'a = {}, u {}, v {}'.format(a, u, v)
        return [a, b]

    def action(self):
        self.t += 1
        if self.t <= self.k:
            # first pick each arm to avoid zero variance
            return (self.t-1)
        return np.argmax(self.values)

class ThompsonSampling(BayesUCB):
    @property
    def values(self):
        samples = []
        for i in range(self.k):
            a, b = self.compute_ab(i)
            samples.append(beta_dist(a,b).rvs())
        return samples

class HierarchicalThompsonSampling(HierarchicalBayesUCB, ThompsonSampling):
    """ Watch out for deadly diamond of death inheritance."""
    pass
