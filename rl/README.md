Experiment 1:
Stability of features over K features vs full log files

Fixed time

while not trained: #10000 iterations

      #one update to the policy model

      #step 1: create trajectories: (itr,...) -> energy, time -> (itr',...) ->
      #N trajectories

      #update the model

create_trajectories:
	start with some initial (itr,...) - uniform,

	compute vector/state:
		#log file experiments:
		read pre-computed

		#real life
		fixed time-budget > stat significance (> 10 ms)
		state/vector, (energy, time)

		#policy: state -> (itr',...)

option 1: establish stability of windows vs full log file -> if stable: use full log file
option 2: find stat sig level -> for each log file, compute several vectors and then pick randomly

l1, l2, ...lN:
    3 lines at a time: (l1,l2,l3), (l4,l5,l6) -> v1, v4, v7,....

full log files: tx_bytes_mean: 15, (15 +/- 5)
     -> inject noise into state

	
	    