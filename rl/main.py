from model import *
from env import *
import utils
import config
from config import debug

import sys
from colorama import Fore, Back, Style

import pandas as pd
import matplotlib.pylab as plt
plt.ion()


if __name__=='__main__':

    if 0:
        df = utils.combine_data('features') #unnormalized CSV file
        df = pd.read_csv('./features/features.csv')

    df = pd.read_csv('./features/logs_0_ebbrt_percentiles.csv', sep = ' ')

    if debug:
        print('------------------------------------------------')
        print(Fore.BLACK + Back.GREEN + "df: " + Style.RESET_ALL)
        print(df)

    if 0:
        df2 = utils.normalize(df)
        env = WorkloadEnv(df2)

    # initialize RL env: states, actions, rewards
    env = WorkloadEnv(df)
    
    # initialize RL policy: sequential NN
    # 128 -> number of hidden nodes, 1 -> number of hidden layers
    pg = PolicyGradient(env, len(env.state), config.N_knobs*config.N_outputs_per_knob, 1, 128, nn.ReLU())

    # training NN
    rcurve = pg.training_loop(1, 1, env, pg.policy, causal=True, lr=1e-3)

    sys.exit()

    print()
    print()
    print("Done training...")
    print()
    print()
#    rcurve = pg.training_loop(1000, 32, env, pg.policy, causal=True, lr=1e-3) #updates same model for another 1000 iterations

    #torch.save(pg.policy, open('mymodel.pt', 'wb))
    #torch.load(open('mymodel.pt', 'rb'))
    
    #1000 = number of updates to policy
    #8 = number of trajectories generated for each update
    #policy is part of pg class
    #causal should generally be True
