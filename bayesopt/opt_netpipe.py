import read_agg_data
from bayes_opt import BayesianOptimization, UtilityFunction

import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec
import itertools

plt.ion()

#ADD CHECKS

#prepare data
df_comb, df, outlier_list = read_agg_data.start_analysis('netpipe')
df_comb = df_comb[(df_comb['msg']==65536) & (df_comb['sys']=='linux')].copy()
df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
df_comb['edp_mean'] *= -1

itr_unique = np.sort(df_comb['itr'].unique())
dvfs_unique = np.sort(df_comb['dvfs'].unique())
rapl_unique = np.sort(df_comb['rapl'].unique())

df_comb.set_index(['itr', 'dvfs'], inplace=True)

#plot
plt.plot(df_comb['edp_mean'].sort_values().tolist())

def objective_interp(itr, dvfs, debug=False):
    '''This can use a MUCH better implementation!
    '''
    itr_list = np.sort(itr_unique[np.argsort(np.abs(itr_unique - itr))][0:2])
    dvfs_list = np.sort(dvfs_unique[np.argsort(np.abs(dvfs_unique - dvfs))][0:2])
    #rapl_list = np.sort(rapl_unique[np.argsort(np.abs(rapl_unique - rapl))][0:2])

    def compute_alpha(limit_list, val):
        low = limit_list[0]
        high = limit_list[1]
        assert(high > low)

        return (val - low) / (high - low)

    #print(itr_list)
    #print(dvfs_list)
    ##print(rapl_list)

    alpha_itr = compute_alpha(itr_list, itr)
    alpha_dvfs = compute_alpha(dvfs_list, dvfs)
    #alpha_rapl = compute_alpha(rapl_list, rapl)

    space = list(itertools.product(itr_list, dvfs_list)) #assumption lists are sorted
    coefficients = [(1-alpha_itr)*(1-alpha_dvfs), 
                    (1-alpha_itr)*alpha_dvfs,
                    alpha_itr*(1-alpha_dvfs),
                    alpha_itr*alpha_dvfs
                    ]
    assert(len(space)==len(coefficients))

    if debug: print(alpha_itr, alpha_dvfs)

    edp_avg = 0
    for idx in range(len(space)):
        corner = space[idx]
        #print(corner, itr, dvfs)
        edp_val = df_comb.loc[corner].loc['edp_mean']

        if debug: print(corner, coefficients[idx], edp_val)
        edp_avg += coefficients[idx] * edp_val

    return edp_avg


#define domain of interest [-5,5] interval and plot
#x = np.linspace(-5, 5, 100).reshape(-1,1)
#y = objective(x)
#plt.plot(x,y)
#plt.title('Actual objective to maximize. Full function unknown in practice.')


#See: https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb
def compute_posterior(optimizer, x, y, grid):
    '''Function to fit a Gaussian Process on points observed so far
    and predict mean and std dev on full domain defined by grid
    '''
    optimizer._gp.fit(x, y) #sklearn's gp regressor

    mu, sig = optimizer._gp.predict(grid, return_std=True)

    return mu, sig

def plot_gp(optimizer, x, y, kappa):
    '''Makes full plots (see comments in line below)
    '''

    #Initialize plot
    plt.figure(figsize=(10,7)) #empty canvas
    plt.suptitle(f'Gaussian Process and Utility after {len(optimizer.space)} Steps') #title
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  #split plot into two subplots
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    #optimizer.res stores all data seen till now
    #get both x values and y values observed so far
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    #use current GP to make predictions across grid and plot
    mu, sig = compute_posterior(optimizer, x_obs, y_obs, x) #fit data on points observed so far and predict on full domain

    ax1.plot(x, y, linewidth=3, label='Target') #plot known function (generally this is unknown)
    ax1.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r') #plot points seen till now
    ax1.plot(x, mu, '--', color='k', label='Prediction') #plot mean prediction across domain
    ax1.fill(np.concatenate([x, x[::-1]]), 
             np.concatenate([mu - 1.9600 * sig, (mu + 1.9600 * sig)[::-1]]),
             alpha=.6, fc='c', ec='None', label='95% confidence interval') #plot 95% confidence intervals
    ax1.set_xlim((-5,5))
    ax1.set_ylim((None, None))
    ax1.set_ylabel('f(x)', fontdict={'size':20})
    ax1.set_xlabel('x', fontdict={'size':20})

    #plot acquisition function
    utility_function = UtilityFunction(kind='ucb', kappa=kappa, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    ax2.plot(x, utility, label='Utility Function', color='purple')
    ax2.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    ax2.set_xlim((-5, 5))
    ax2.set_ylim((0, np.max(utility) + 0.5))
    ax2.set_ylabel('Utility', fontdict={'size':20})
    ax2.set_xlabel('x', fontdict={'size':20})
    
    ax1.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    ax2.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)    



def plot_maximization(n_iter, kappa):
    optimizer = BayesianOptimization(objective_interp, {'itr': (itr_unique[0], itr_unique[-1]),
                                                        'dvfs': (dvfs_unique[0], dvfs_unique[-2])},
                                                        ) #randomly generate points
    optimizer.maximize(init_points=2, n_iter=n_iter, kappa=kappa)
    
    plt.figure()
    vals = df_comb['edp_mean'].sort_values().tolist()
    N = len(vals)
    plt.plot(vals)

    for i in range(len(optimizer.res)):    
        plt.plot(np.arange(N), [optimizer.res[i]['target']] * N, color='r')

    return optimizer
#utility = UtilityFunction(kind='ucb', kappa=3)