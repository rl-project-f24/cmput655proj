# Sections breakdown

SECITON 1 (environment / inference)
- We start with a policy gradient method initialized with random weights.
- Then, we get our uniform random n trajectories. 
SECTION 2 (preference calculation / fitting)
- Then we get the scalar total cumulative reward of all these, 1 value for each trajectory.
- Then, we calculate preference data for these trajectories - exponential in the trajectory count for n trajectories - representing comparative values (a > b), 1 and -1 (better for the MLE). This algorithm used for calculating preferences is in the openai paper. (we have to write this code)
- Then, we CORRUPT some amount of preferences here.
- We use the preference data to fit reward function, which will give us reward for all state action pairs in all trajectories that we have seen. This fit reward algo is going to be the one found in the paper - 2.2.3. (We have to write/find this code)
- We apply the reward function to each trajectory - getting per state-action pair in each trajectory, an associated reward for that pair. (is a list, each element is appended a reward)
SECTION 3 (policy training / optimization)
- THEN, we do training our policy gradient, AS IF it is running online, through the trajectory, for each algo.
- Then we generate new trajectories.


