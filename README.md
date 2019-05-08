# Reinforcement Algorithms

Here We implement three RL/IL algorithms: TRPO, PPO and GAIL.  

+ Trust Region Policy Optimization (TRPO) 
	+ Running: python train.py --als trpo --vv 1 --env_name HalfCheetah-v2
+ Proximal Policy Optimization (PPO)
	+ Running: python train.py --als ppo --vv 1 --env_name HalfCheetah-v2
+ Generative Adversial Imitation Learning (GAIL)
	+ Normal GAIL
		+ Running: python train.py --als gail --vv 1 --mm 1 --env_name HalfCheetah-v2
	+ GAIL without Expert Action
		+ Running: python train.py --als gail --vv 1 --mm 1 --env_name HalfCheetah-v2 --action 1
	+ GAIL with Agent's Action as Expert's 
		+ Running: python train.py --als gail --vv 1 --mm 1 --env_name HalfCheetah-v2 --action 2
	+ Collect Expert Trajectories
		+ Running: python train.py --als save --vv 1 -env_name HalfCheetah-v2 

##References:

+ Schulman, John, Levine, Sergey, Moritz, Philipp, Jordan, Michael I, and Abbeel, Pieter. Trust region policy optimization. In International Conference on Machine Learning (ICML), 2015a.

+ John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 

+ Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. In NIPS, pp. 4565–4573, 2016.

