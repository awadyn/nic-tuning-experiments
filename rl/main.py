import model
import env
import viz

if __name__=='__main__':

    state_dict = read_all_csvs(loc)

    nic_env = env.Workload(state_dict)

    pg = PolicyGradient(data, N_inputs, N_nodes, N_layers, activation, output_activation, df)

    pg.training_loop()

    viz.plot(pg)