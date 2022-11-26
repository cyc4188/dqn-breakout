from collections import deque
import os
import random
import sys
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory

BASIC_MODEL = None
GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 2000
WARM_STEPS = 1000
MAX_STEPS = 5000000
EVALUATE_FREQ = 10000

EPS_DECAY = MAX_STEPS - WARM_STEPS

BASE_COUNT = 0

BEST_AWARD = 0


rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)

torch.set_num_threads(32)

last = []

for dirpath, dirnames, filenames in os.walk(SAVE_PREFIX):
    last = [i for i in filenames if not i.endswith("mem")]
    break

if len(last) == 0:
    print("No model found, creating new one.")
    os.makedirs(SAVE_PREFIX, exist_ok=True)
    torch.manual_seed(new_seed())
    device = torch.device("cuda")
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
    device = torch.device("cuda")
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

last_saved_mem = ''

progressive = tqdm(range(MAX_STEPS), unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"Cur: {BASE_COUNT + step//EVALUATE_FREQ:03d} Reward: {avg_reward:.4f}\n")

        progressive.set_description(f"reward: {avg_reward:.4f}")

        if avg_reward < BEST_AWARD:
            continue

        BEST_AWARD = avg_reward
        progressive.write(f"\rSaving model with reward {avg_reward:.4f}")

        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{BASE_COUNT + step//EVALUATE_FREQ:03d}"))
        memory.save(os.path.join(
            SAVE_PREFIX, f"model_{BASE_COUNT + step//EVALUATE_FREQ:03d}.mem"))

        if last_saved_mem != '':
            os.remove(os.path.join(SAVE_PREFIX, last_saved_mem))

        last_saved_mem = f"model_{BASE_COUNT + step//EVALUATE_FREQ:03d}.mem"

        done = True
