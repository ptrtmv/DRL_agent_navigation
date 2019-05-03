[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

### DRL_agent_navigation
A DRL agent navigating in a large, square world (continuous state) and collecting bananas (discrete actions)! 


---

## Intorduction

This is a Deep Reinforcement Learning project training a single Unity ML-agent to navigate the [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment which has continuous 37 dimensional state space and 4 possible actions.

![Trained Agent][image1]


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The task is episodic, and in order to solve the environment, our agent must get an average score of +13 over 100 consecutive episodes.



## Getting Started

In order to run the environment you need to download it for your  operation system from one of the following links:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


Additionally you will need following packages: 
* **matplotlib**: used to visualise the training results 
* **numpy**: shouldn't really surprise you...
* **torch**: used for the deep Q-Network
* **unityagents**: used to run the downloaded [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment

which can be directly installed while running the **Jupiter** notebook `Navigation.ipynb`



## Instructions

You can run the project via the **Jupiter** notebook `Navigation.ipynb`. The classes defining the agent and its "brain" are stored in `agent.py` and the Q-Network is defined in `networks.py`.








