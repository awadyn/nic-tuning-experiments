from model import *
from env import *
import utils

import pandas as pd
import matplotlib.pylab as plt
plt.ion()

'''
Bug in model.PolicyGradient

env dynamics isn't correct

reward needs shaping - penalty for early termination

     - visualization
     - remove dvfs = 65535
     - reward?
     - remove joules_per_interrupt, time_per_interrupt, read_lat
     - interpretation/explainability
'''

if __name__=='__main__':

    #df = utils.combine_data('features') #unnormalized CSV file
    #df = pd.read_csv('features.csv')

    df2 = utils.normalize(df)

    env = WorkloadEnv(df2)

    pg = PolicyGradient(env, len(env.state), 6, 1, 10, nn.ReLU())

    _ = pg.create_trajectories(env, pg.policy, 10, debug=True)
    rcurve = pg.training_loop(1000, 8, env, pg.policy, causal=True, lr=1e-3)
    #1000 = number of updates to policy
    #8 = number of trajectories generated for each update
    #policy is part of pg class
    #causal should generally be True