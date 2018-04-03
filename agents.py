import numpy as np

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
