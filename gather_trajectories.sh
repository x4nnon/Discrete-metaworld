#!/bin/bash

echo "starting script"

hare run --rm --workdir /app --gpus device=0 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "drawer-close-v2-goal-observable" 0
hare run --rm --workdir /app --gpus device=0 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "basketball-v2-goal-observable" 0

echo "one and two have started"

hare run --rm --workdir /app --gpus device=3 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "button-press-v2-goal-observable" 0
hare run --rm --workdir /app --gpus device=3 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "dial-turn-v2-goal-observable" 0

echo "three and four have started"

hare run --rm --workdir /app --gpus device=2 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "reach-wall-v2-goal-observable" 0
hare run --rm --workdir /app --gpus device=2 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "peg-insert-side-v2-goal-observable" 0 

echo "five and six have started: there may be a wait now"

hare run --rm --workdir /app --gpus device=0 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "push-wall-v2-goal-observable" 0 &
hare run --rm --workdir /app --gpus device=0 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "pick-place-wall-v2-goal-observable" 0 &

echo "seven and eight have started"

hare run --rm --workdir /app --gpus device=1 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "sweep-into-v2-goal-observable" 0 &
hare run --rm --workdir /app --gpus device=1 -it -d -v "$(pwd)":/app tc2034/metaworld python3 /app/main.py "window-open-v2-goal-observable" 0 &

echo "nine and ten have started"