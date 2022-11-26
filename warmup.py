from collections import deque
import os
import random
import sys
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 50000
RENDER = False
SAVE_PREFIX = "./models"
# BASIC_MODEL = "./saved_models/model_000"
BASIC_MODEL = None
STACK_SIZE = 4

EPS_START = .9
EPS_END = 0
EPS_DECAY = 200000

BATCH_SIZE = 32
POLICY_UPDATE = 10
TARGET_UPDATE = 10000
WARM_STEPS = 40000
MAX_STEPS = 600000

EPS_DECAY = MAX_STEPS - WARM_STEPS

EVALUATE_FREQ = 50000
BASE_COUNT = 0

BEST_AWARD = 0

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
# new_seed = lambda: 0

last = []

for dirpath, dirnames, filenames in os.walk(SAVE_PREFIX):
    last = [i for i in filenames if not i.endswith("mem")]
    break

if len(last) == 0:
    print("No model found, creating new one.")
    os.makedirs(SAVE_PREFIX, exist_ok=True)
    torch.manual_seed(new_seed())
    device = torch.device("mps")
    env = MyEnv(device)
    agent = Agent(
        env.get_action_dim(),
        device,
        GAMMA,
        new_seed(),
        EPS_START,
        EPS_END,
        EPS_DECAY,
        BASIC_MODEL
    )
    memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)
else:
    last = sorted(last)[-1]
    print(f"Loading model from {last}")
    BASE_COUNT = int(last.split("_")[1])
    torch.manual_seed(new_seed())
    device = torch.device("mps")
    env = MyEnv(device)
    agent = Agent(
        env.get_action_dim(),
        device,
        GAMMA,
        new_seed(),
        EPS_START,
        EPS_END,
        EPS_DECAY,
        os.path.join(SAVE_PREFIX, last),
    )
    memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)
    memory.load(os.path.join(SAVE_PREFIX, last + ".mem"))

#### Training ####
obs_queue = deque(maxlen=5)
BEST_AWARD, _ = env.evaluate(obs_queue, agent, render=False)
done = True

print(f"Best award: {BEST_AWARD}")

progressive = tqdm(range(MEM_SIZE), unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, False)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

memory.save("warmup.mem")

print("Warmup finished.")
