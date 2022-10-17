# Discrete-metaworld

Meta-world is an open-source simulated benchmark for meta-reinforcement learning and multi-task learning consisting of 50 distincy robotic manipulation tasks: See more information on the github [https://github.com/rlworkgroup/metaworld](https://github.com/rlworkgroup/metaworld)

The purpose of this repository is to reduce the dimensions of the action space by *discretisation.* There are other reason that discretisation are desired; in another project based on building hierarchical domain specific languages it is beneficial to have the primitive action space as discrete (because the hierarchical actions are also discrete).

Finally the discrete version will be wrapped as an open-ai gym environment and therefore any baseline can be tested using stable-baselines / spinning-up etc …
