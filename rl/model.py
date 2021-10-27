 class PolicyGradient:
    def __init__(self, data, N_inputs, N_nodes, N_layers, activation, output_activation, df):
        self.data = data

        self.policy = PolicyNet(N_inputs, N_nodes, N_layers, activation, output_activation, df)

        '''
        Options:
        Summarizes data (features) -> Action (ITR)

        Raw log files (RNN) -> Action (ITR)

        Features/Log files look up and Architecture
        '''

    def training_loop(self,
                      N_iter, 
                      batch_size, 
                      env,
                      policy=None, 
                      lr=1e-2,
                      critic_lr=1e-1, 
                      causal=False,
                      baseline=False,
                      critic=None,
                      critic_update='MC',
                      debug=False):
        
        if policy is None:
            policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, nn.ReLU(), nn.Sigmoid())
        
        reward_curve = {}

        optimizer_policy = optim.Adam(policy.parameters(), lr=lr)
        if critic:
            optimizer_critic = optim.Adam(critic.parameters(), lr=critic_lr)

        exp_reward_list = []

        for i in range(N_iter):
            #step 1: generate batch_size trajectories
            J_policy, mean_reward, J_critic = create_trajectories(env, policy, batch_size, causal=causal, baseline=baseline, critic=critic, critic_update=critic_update)
            #print(f'Critic Loss = {J_critic}')
            
            #step 2: define J and update policy
            optimizer_policy.zero_grad()
            (-J_policy).backward()
            optimizer_policy.step()

            #step 3: 
            if critic:
                if J_critic.item() > 0.1:
                    optimizer_critic.zero_grad()
                    J_critic.backward()
                    optimizer_critic.step()        

            if i % 10 == 0:
                print(f"Iteration {i} : Mean Reward = {mean_reward}")
                reward_curve[i] = mean_reward

        return reward_curve
   
    def create_trajectories(self,
                            env, 
                            policy, 
                            N, 
                            causal=False, 
                            baseline=False, 
                            critic=False, 
                            critic_update='MC'):

        action_probs_all_list = []
        rewards_all_list = []
        states_all_list = []

        for _ in range(N): #run env N times
            state = env.reset()
            
            action_prob_list, reward_list, state_list = torch.tensor([]), torch.tensor([]), torch.tensor([])
            done = False 

            while not done:
                state_list = torch.cat((state_list, torch.tensor([state]).float())) #

                action_probs = policy(torch.from_numpy(state).unsqueeze(0).float()).squeeze(0)

                action_selected_index = torch.multinomial(action_probs, 1)
                action_selected_prob = action_probs[action_selected_index]
                action_selected = action_space[action_selected_index]

                state, reward, done, info = env.step(action_selected.item())

                action_prob_list = torch.cat((action_prob_list, action_selected_prob))
                reward_list = torch.cat((reward_list, torch.tensor([reward])))

            action_probs_all_list.append(action_prob_list)
            rewards_all_list.append(reward_list)
            states_all_list.append(state_list)

        #non-optimized code (negative strides not supported by torch yet)
        rewards_to_go_list = [torch.tensor(np.cumsum(traj_rewards.numpy()[::-1])[::-1].copy())
                              for traj_rewards in rewards_all_list]


        #compute objective
        J = 0
        critic_inputs, critic_targets = [], []
        #for clarity, refactoring code below into specific modes. some of these can be combined
        if not baseline and not causal:
            print("No Baseline - Not Causal")
            for idx in range(N):
                J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum())

        if not baseline and causal:
            print("No Baseline - Causal")
            for idx in range(N):
                row = rewards_to_go_list[idx]
                J += (action_probs_all_list[idx].log() * row).sum()

        #critic only shows up in baseline cases
        if baseline and not causal and not critic: #critic only affects baseline cases
            print("Baseline - Not Causal, No Critic")
            baseline_term = np.mean([torch.sum(r).item() for r in rewards_all_list])

            for idx in range(N):
                J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum() - baseline_term)

        if baseline and not causal and critic:
            raise NotImplementedError("Probably not useful")
            for idx in range(N):
                row = rewards_to_go_list[idx]
                actions = action_probs_all_list[idx]
                states = states_all_list[idx]

                T = len(row)
                for t in range(T):
                    J += actions[t].log() * (row[t] - critic(torch.cat([torch.tensor([t]).float(), states[t]])))

        if baseline and causal and not critic:
            '''Need time-dependent baseline terms
            '''
            print("Baseline - Causal, No Critic")
            #compute time-dependent baseline terms (mean reward to go)
            T = np.max([len(row) for row in rewards_to_go_list])
            baseline_term = torch.zeros(N, T)
            for idx in range(N):
                row = rewards_to_go_list[idx]
                baseline_term[idx] = torch.cat((row, torch.zeros(T-len(row)))) #pad with 0s if episode took time < T (end)
            baseline_term = torch.mean(baseline_term, dim=0) #time-dependent means

            #compute J
            for idx in range(N):
                row = rewards_to_go_list[idx]
                J += (action_probs_all_list[idx].log() * (row - baseline_term[:len(row)])).sum() #subtract time-dependent means

        if baseline and causal and critic:
            #train/update critic
            #(state, t) -> (rewards to go)

            #print("Baseline - Causal, Critic")
            #return states_all_list, action_probs_all_list, rewards_to_go_list
            for idx in range(N):
                row = rewards_to_go_list[idx]
                rewards = rewards_all_list[idx] #needed to update critic
                actions = action_probs_all_list[idx]
                states = states_all_list[idx]

                T = len(row)
                for t in range(T):
                    critic_in = torch.cat([torch.tensor([t]).float(), states[t]])
                    
                    J += actions[t].log() * (row[t] - critic(critic_in))

                    if critic_update=='MC':
                        critic_inputs.append(critic_in)
                        critic_targets.append(row[t]) #pure MC - target = reward to go
                    
                    if critic_update=='TD':
                        if t < T-1:
                            critic_inputs.append(critic_in)
                            critic_targets.append(rewards[t].unsqueeze(0) + critic(torch.cat([torch.tensor([t+1]).float(), states[t+1]]))) #TD - target = current reward + critic estimate
        J = J / N


        R = np.mean([torch.sum(r).item() for r in rewards_all_list]) #track overall progress

        #critic loss computation
        J_critic = None
        critic_loss = nn.MSELoss()
        
        if critic:
            critic_inputs = torch.stack(critic_inputs)
            critic_targets = torch.stack(critic_targets)

            preds = critic(critic_inputs)

            J_critic = critic_loss(preds, critic_targets)

        return J, R, J_critic    


    def train(self, N_batch_size, T=10):
        for exp in range(N_batch_size):

            for t in range(T):
                pass
                #step 1: construct inputs
                #step 2: predict action probs
                #step 3: sample from multinomial distribution (everything discrete here)
                #step 4: measure instantaneous reward

            #step 5:

    def get_learning_curves(self,
                            N_exp, 
                            N_iter, 
                            batch_size, 
                            env, 
                            causal=False, 
                            baseline=False):

        reward_curve_list = []
        for _ in range(N_exp):
            policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation)

            reward_curve = training_loop(N_iter, batch_size, env, policy, causal=causal, baseline=baseline)

            reward_curve_list.append(reward_curve)

        #combine results
        rcurve = {}
        for k in reward_curve_list[0]:
            rcurve[k] = [r[k] for r in reward_curve_list]
            rcurve[k] = (np.mean(rcurve[k]), np.std(rcurve[k]))

        return rcurve

    def plot_learning_curve(self,
                            rcurve, 
                            label):

        k = list(rcurve.keys())

        plt.errorbar(k, [rcurve[key][0] for key in k], [rcurve[key][1] for key in k], label=label)
        plt.legend()
        plt.xlabel('Episode Number')
        plt.ylabel('Average Score')
        plt.title('10 runs, 200 iterations each, batch-size 20 episodes')
