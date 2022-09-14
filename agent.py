import numpy as np
from scipy.optimize import toms748
import math
from scipy.stats import entropy


class Agent:
    def __init__(self, pop_n, id, init_x=None, state=True):

        if not init_x:
            self.x = np.random.uniform(0.01,0.99)
        else:
            # belief of possible world H1, probability that H1 is true
            self.x = init_x

        # each agent has its own id
        self.id = id

        # each agent has its belief of agents
        self.confidence = self.init_confidence(pop_n, id)
        self.mal_ls = []

        # belief dict
        self.belief_dict = {i: {0: 0.5} for i in range(pop_n)}
        # agent is functioning or not.
        self.state = state

    def mal_detection(self):
        mal_id = np.zeros(len(self.confidence))
        if self.state:
            self.mal_ls = list(np.where(self.confidence==0.0)[0])
            mal_id[self.mal_ls] = 1
        return mal_id

    def init_confidence(self, pop_n, id):
        # belief of other agents is 1
        confidence = np.ones(pop_n)

        # belief of itself is 1
        # confidence[id] = 1

        return confidence

    def update_belief_dict(self, id_ls, time_step, pool_belief, pool_belief_dict):
        for i ,id in enumerate(id_ls):
            self.belief_dict[id][time_step] = pool_belief[i]
        self.belief_dict.update(pool_belief_dict)

    def malfunction(self):
        self.x = np.random.uniform(0.01,0.99)
        return self.x




# class Malicious:
#     def __init__(self, pop_n, id, threshold, mal_c):
#
#         self.id = id
#         self.state = False
#         self.threshold = threshold
#         self.confidence = self.init_confidence(pop_n, id)
#         self.mal_ls = []
#         self.x = 0.5
#         self.tol = 10e-5
#         self.pool = None
#         self.w = None
#         self.c = mal_c
#
#     def deception(self, pool_prob, weights, guess = (0.001, 0.999), tol = 10e-5):
#
#
#         self.pool = np.array(pool_prob)
#         self.w = np.array(weights)
#
#         self.x = self.secant(guess)
#
#         while self.conf(self.x) < 0:
#             guess = (guess[0]*2, guess[1])
#             self.x = self.secant(guess)
#
#         if self.c:
#             x_m = self.pooled_root()
#             if not np.isnan(x_m) and x_m > self.x:
#                 if not np.isnan(guess[1]) and x_m <= guess[1]:
#                     self.x = x_m
#                 elif self.conf(x_m) > 0:
#                     self.x = x_m
#
#         self.x = math.ceil(self.x / self.tol) * self.tol
#         pooled_prob = self.log_linear(self.x, self.pool, self.w)
#
#         return self.x, pooled_prob
#
#     def pull_only(self, pool_prob, weights):
#
#         self.pool = np.array(pool_prob)
#         self.w = np.array(weights)
#
#         if self.c:
#             self.x = self.pooled_root()
#             if np.isnan(self.x):
#                 self.x = self.c
#             else:
#                 self.x = math.ceil(self.x / self.tol) * self.tol
#             pooled_prob = self.log_linear(self.x, self.pool, self.w)
#         else:
#             self.x = 0.001
#             pooled_prob = self.log_linear(self.x, self.pool, self.w)
#         return self.x, pooled_prob
#
#     def secant(self, guess):
#
#         if self.conf(guess[0]) > 0:
#             return guess[0]
#
#         root = newton(self.conf, guess)
#         x = min(root)
#         if not np.isnan(x):
#             x_m = math.ceil(x / self.tol) * self.tol
#             if self.conf(x_m) < 0:
#                 x_m = math.floor(x/self.tol) * self.tol
#         else:
#             x_m = guess[0]
#         return x_m
#
#     def log_linear(self, x, pool, w):
#
#         numerator = np.prod(pool ** w) * (x ** (1-w.sum()))
#         return numerator / (numerator + np.prod((1 - pool) ** w) * ((1 - x) ** (1-w.sum())))
#
#
#     def pooled_root(self):
#         w_m = 1-self.w.sum()
#         b = np.prod( ((1-self.pool)/self.pool) ** self.w )
#         x_m = 1/( ((1-self.c)/(b*self.c)) ** (1/w_m) + 1)
#
#         return x_m if x_m != 0.0 else self.tol
#
#     def conf(self, x):
#         return np.exp(-self.kl_divergence(np.array(self.log_linear(x, self.pool, self.w)), x)) - self.threshold
#
#     def mal_detection(self):
#         mal_id = np.zeros(len(self.confidence))
#         if self.state:
#             self.mal_ls = list(np.where(self.confidence==0.0)[0])
#             mal_id[self.mal_ls] = 1
#         return mal_id
#
#     def init_confidence(self, pop_n, id):
#
#         # belief of other agents is 0.5
#         confidence = np.ones(pop_n) / 2
#
#         # belief of itself is 1
#         confidence[id] = 1
#
#         return confidence
#
#     def kl_divergence(self, x1, x2):
#
#         return entropy([x1, 1-x1], [x2, 1-x2])
#
#
#


class Malicious:
    def __init__(self, pop_n, id, threshold, mal_c):

        self.id = id
        self.state = False
        self.threshold = threshold
        self.confidence = self.init_confidence(pop_n, id)
        self.mal_ls = []
        self.x = 0.5
        self.tol = 10e-6
        self.pool = None
        self.w = None
        self.c = mal_c

    def deception(self, pool_prob, weights, guess = (0.001, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 0.999), tol = 10e-5):

        self.pool = np.array(pool_prob)
        self.w = np.array(weights)

        region = self.region(guess)

        if self.c:
            x_m = self.pooled_root()

            if region[0] <= x_m <= region[1]:
                self.x = x_m
            elif x_m > region[1]:
                self.x = region[1]
            else:
                self.x = region[0]
        else:
            self.x = region[0]

        pooled_prob = self.log_linear(self.x, self.pool, self.w)
        return self.x, pooled_prob

    def pull_only(self, pool_prob, weights):

        self.pool = np.array(pool_prob)
        self.w = np.array(weights)

        if self.c:
            self.x = self.pooled_root()
            self.x = math.ceil(self.x / self.tol) * self.tol
            pooled_prob = self.log_linear(self.x, self.pool, self.w)
        else:
            self.x = 0.001
            pooled_prob = self.log_linear(self.x, self.pool, self.w)

        return self.x, pooled_prob

    def region(self, guess):

        region = []

        if self.conf(guess[0]) > 0:
            region.append(guess[0])
            for x in guess[1:]:
                if self.conf(x) < 0:
                    region.append(self.conf_root(guess[0], x))
                    break
        else:
            for x in guess[1:]:
                if self.conf(x) > 0:
                    region.append(self.conf_root(guess[0], x))
                    break
        if len(region) == 2:
            return region
        elif len(region) == 1:
            if self.conf(guess[-1]) > 0:
                region.append(guess[-1])
            else:
                for x in np.flip(guess[:-1]):
                    if self.conf(x) > 0:
                        region.append(self.conf_root(x, guess[-1]))
                        break
        else:

            print(f'pool:')
            print(self.pool)
            print(f'w:')
            print(self.w)
            print(region)
            raise Exception('no safe region')

        return region

    def conf_root(self, low, upper):
        x = toms748(self.conf, low, upper)
        return self.check_zeros(x)

    def log_linear(self, x, pool, w):
        if 1-w.sum() < self.tol:
            numerator = np.prod(pool ** w)
            return numerator / (numerator + np.prod((1 - pool) ** w))
        else:
            numerator = np.prod(pool ** w) * (x ** (1-w.sum()))
            return numerator / (numerator + np.prod((1 - pool) ** w) * ((1 - x) ** (1-w.sum())))

    def pooled_root(self):

        w_m = 1-self.w.sum()
        b = np.prod( ((1-self.pool)/self.pool) ** self.w )
        if w_m < self.tol or b == 0.0:
            return self.c
        x_m = 1/( ((1-self.c)/(b*self.c)) ** (1/w_m) + 1)

        if x_m == 1.0:
            x_m = 0.99

        return x_m if x_m != 0.0 else 0.01

    def conf(self, x):
        return np.exp(-self.kl_divergence(np.array(self.log_linear(x, self.pool, self.w)), x)) - self.threshold

    def mal_detection(self):
        mal_id = np.zeros(len(self.confidence))
        if self.state:
            self.mal_ls = list(np.where(self.confidence==0.0)[0])
            mal_id[self.mal_ls] = 1
        return mal_id

    def init_confidence(self, pop_n, id):

        # belief of other agents is 0.5
        confidence = np.ones(pop_n) / 2

        # belief of itself is 1
        confidence[id] = 1

        return confidence

    def check_zeros(self, x):
        x_m = math.ceil(x / self.tol) * self.tol
        if self.conf(x_m) < 0:
            x_m = math.floor(x/self.tol) * self.tol
        return x_m

    def kl_divergence(self, x1, x2):

        return entropy([x1, 1-x1], [x2, 1-x2])


