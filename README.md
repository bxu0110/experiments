Experiment 1: We test the initial approach on a few selected functions - Results: it doesn't work

Experiment 2: We change the number of actions in the initial approach to 10 and 50 actions - Results: it still doesn't work

Now we test the environments on selected policies (observations and agents not relevant)
 
Experiment 3: Initial approach - Results: indepently of the policies the rewards in this environment are very similar

Experiment 4: Different ranges for sampling initial solutions around best solution in restarts- Results: different policies lead to different rewards

Experiment 5: Different converges rates for UES - Results: some difference in policies but not as clear as in Experiment 4.

Experiment 6: Different combination of parameters ofr UES-CMAES (including restart ranges and convergence rates) - Results: very clear difference in rewards for different policies

Experiment 7: In this experiment we determine which combinations/combos of parameters work best

Now we test the agent inside the environment for the environemnts with restart ranges and combos. We do this in some selected functions - same as experiments 1 and 2

Experiment 8: Assesing the DQN agent with environment that controls restart range - Results: Agent improves 

Experiment 9: Assesing the DQN agent with environment that controls combo of parameters - Results: Agent improves better than previous environment.

Now that we have a working Agent/Environemnt we are ready to generalize. For generalizing we need to determine what reward works best between standard reward and normalized

Experiment 10: DQN Agent + combo environment with standard reward - Results: very hard to interpret because rewards vary to much from function to function

Experiment 11: DQN Agent + combo environment with normalized reward - Results: easier to interpret, but still unclear of how much improves

Because it is hard to determine how well the agent/environments perform we test both trained models with the optimization hybrid.

Experiment 12: UES-CMAES restart hybrid with trained agent on standard rewards - Results: very good

Experiment 13: UES-CMAES restart hybrid with trained agent on standard rewards - Results: very good
 
