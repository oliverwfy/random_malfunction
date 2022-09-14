from model import simulate_malfunctioning
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
warnings.filterwarnings('ignore')

font_size = 16

# number of all agents
pop_n = 100

# initial belief of possible world H1
init_x = None

# (Malfunctioning agents) initial belief of possible world H1
mal_x = None

# number of pooled agents in each iteration
k = 10

# maximum iteration
max_iteration = 600

# simulation times
simulation_times = 1

# P(E|H1) = 1 - alpha   if E = H1
# P(E|H1) = alpha       if E = H2
alpha = 0.1

# probability of receiving evidence
prob_evidence = 0.02

# percentage of malicious agents
malicious = 0.1

# confidence threshold
threshold= 0.4

# constant belief c
mal_c = 0.5

dampening = 0

trend = False

weights_updating = None



malfunctioning = 0.1

file_name = 'image/'

trend = True

result = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n, trend = trend,
                                 max_iteration=max_iteration, k=k, init_x = init_x, weights_updating=weights_updating,dampening=dampening,
                                 mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence, detection_only=True,
                                 memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
                                 pooling=True)

belief_dict = result['belief_dict']


def padding_belief(belief_dict_i, max_iteration):
    output = np.empty(max_iteration)
    for k,v in belief_dict_i.items():
        output[k:] = v
    return output


plt.plot(range(max_iteration), padding_belief(belief_dict[99], max_iteration))
plt.plot(range(max_iteration), padding_belief(belief_dict[0], max_iteration), '--')

plt.xlabel('Iteration')
plt.ylabel('Belief')
plt.legend(['normal', 'malfunctioning'])
plt.savefig(file_name + f'learning_pattern_dense_over.png')

plt.show()




# trend = True
# result_with_trend = simulate_malfunctioning(simulation_times=simulation_times, pop_n=pop_n, trend = trend,
#                                  max_iteration=max_iteration, k=k, init_x = init_x, weights_updating=weights_updating,dampening=dampening,
#                                  mal_x = mal_x, alpha=alpha, prob_evidence=prob_evidence, detection_only=True,
#                                  memory=False, memory_weight=0.5, malfunctioning=malfunctioning, threshold=threshold,
#                                  pooling=True)

# acc_with_trend = result_with_trend['accuracy_avg'].mean(axis=1)
#
# plt.plot(range(max_iteration), acc)
# plt.plot(range(max_iteration), acc_with_trend)
#
# plt.legend(['acc', 'acc_with_trend'])
# plt.xlabel('Iteration', fontsize=font_size)
# plt.ylabel('accuracy', fontsize=font_size)
# plt.ylim([0,1])
# plt.title(f'Simulation times : {simulation_times}', fontsize=font_size)
# plt.savefig(file_name + f'accuracy_trend_over_{simulation_times}.png')
# plt.show()
#
