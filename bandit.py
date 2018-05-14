## bandit.py 

import numpy as np
from scipy.stats import beta as beta_dist

class KBandit():
    """ Bernuli multiarmed bandit with Beta prior. 
        Example 3.1 of "Bayesian Reinforcement Learning."
        
        Implementation is slightly different than the Model 5 (Simple Family of Alternative Bandit Processes)
        (page 32)
    """
    def __init__(self, k, alpha=1, beta=1, max_steps=None):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.max_steps = max_steps
        
        self.dist = beta_dist(self.alpha, self.beta)
        
    def reset(self):
        self.cur_step = 0
        self._theta = self.dist.rvs(self.k)
        return None, 0, False, None

    def _draw(self):
        draws = np.random.uniform(size=self.k)
        # print(draws)
        self._reward_table = (draws < self._theta).astype(int)

    def optimal_action(self):
        return np.argmax(self._theta)
    
    def step(self, action, draw=True):
        """ If draw=False the step isn't counted, and the draws from last step are reused.
            This makes it easier to fairly compare algorithms.
        """
        if draw:
            self._draw()
            self.cur_step += 1
        
        s, i = None, None
        r = self._reward_table[action]
        d = True if self.max_steps and self.cur_step > self.max_steps else False
        return s, r, d, i