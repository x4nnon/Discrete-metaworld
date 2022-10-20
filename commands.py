#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:22:19 2022

@author: x4nno
"""

# These are just the commands to run the code on hex. 
# This just speeds it up instead of having to type every time


# change the gpu and the run argument 

hare run --rm --workdir /app --gpus device=0 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "drawer-close-v2-goal-observable" 0