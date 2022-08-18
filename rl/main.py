from model import *
from env import *
import utils
import config

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
#    df = pd.read_csv('./features/features.csv')
    df = pd.read_csv('./features/logs_0_ebbrt_percentiles.csv', sep = ' ')
    print(df)
    print()
    print()
#    df2 = utils.normalize(df)
#    print(df2)
#    print()
#    print()

#    env = WorkloadEnv(df2)
    env = WorkloadEnv(df)
    
    pg = PolicyGradient(env, len(env.state), 3*config.N_outputs_per_knob, 1, 128, nn.ReLU()) #128 -> number of hidden nodes, 1 -> number of hidden layers

#    _ = pg.create_trajectories(env, pg.policy, 10, debug=True) #test trajectories - don't need to run explicitly
    rcurve = pg.training_loop(1000, 5, env, pg.policy, causal=True, lr=1e-3)
#    rcurve = pg.training_loop(1000, 32, env, pg.policy, causal=True, lr=1e-3) #updates same model for another 1000 iterations

    #torch.save(pg.policy, open('mymodel.pt', 'wb))
    #torch.load(open('mymodel.pt', 'rb'))
    
    #1000 = number of updates to policy
    #8 = number of trajectories generated for each update
    #policy is part of pg class
    #causal should generally be True
