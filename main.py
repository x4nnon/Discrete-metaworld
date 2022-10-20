#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 09:50:47 2022

@author: x4nno
"""

""" So the sys.argv should be in the order 0=main.py, 1= the v2 environments,
    2= the run number"""

from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                                ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

import stable_baselines3
import os
import pickle
import sys


import discrete_metaworld
from discrete_metaworld import discreteMetaWorld, record_decisions

# DELETE THESE WHEN RUNNING FROM CMD LINE: THIS IS JUST FOR INVESTIGATION 
sys.argv = [1,2,3]
sys.argv[0] = None
sys.argv[1] = "drawer-close-v2-goal-observable"
sys.argv[2] = 0

# delete end


env_name = sys.argv[1]

mw_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]

# it is the seed which changed the starting and reward locations.
cont_env = mw_env(seed=42)

log_dir = "./PPO_ABC_prim"
os.makedirs(log_dir, exist_ok = True)

tag = "{}".format(env_name) # this will be long but it's probably better to be accurate 
recording_folder = "./test_records/{}".format(tag) # again massive folder but clear.

try:
    os.makedirs(recording_folder)
except:
    pass # hacky

run_number = sys.argv[2]
print("starting run ", run_number) # sys.argv[1] is because we running on hex temp thing.
disc_env = discreteMetaWorld(cont_env)
model = stable_baselines3.PPO('MlpPolicy', disc_env, tensorboard_log=log_dir,
                              device="cuda")

model.learn(total_timesteps=1000000, callback=record_decisions)

pickle.dump(disc_env.decisions_list, open("{}/{}_{}_decisions".format(recording_folder, 
                                                             tag, run_number), "wb"))

pickle.dump(disc_env.rewards_list, open("{}/{}_{}_reward".format(recording_folder,
                                                            tag, run_number), "wb"))

print(disc_env.rewards_list)

pickle.dump(disc_env.length_list, open("{}/{}_{}_length".format(recording_folder,
                                                       tag, run_number), "wb"))

pickle.dump(disc_env.success_trajectory_list, open("{}/{}_{}_success_ts".format(recording_folder,
                                                       tag, run_number), "wb"))

pickle.dump(disc_env.failure_trajectory_list, open("{}/{}_{}_failure_ts".format(recording_folder,
                                                       tag, run_number), "wb"))

pickle.dump(disc_env.all_trajectory_list, open("{}/{}_{}_all_ts".format(recording_folder,
                                                       tag, run_number), "wb"))

pickle.dump(disc_env.success_list, open("{}/{}_{}_success_list".format(recording_folder,
                                                       tag, run_number), "wb"))

model.save("PPO_{}_{}".format(sys.argv[1], sys.argv[2])) # RL method, env, run number

print("number of failure : ", len(disc_env.failure_trajectory_list))
print("number of success : ", len(disc_env.success_trajectory_list))