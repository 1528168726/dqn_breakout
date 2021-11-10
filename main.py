from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4
REMOVED_PARAM = 0.35  # 不移动的奖励参数

EPS_START = 0.1      # 前期eps大，倾向于随机选
EPS_END = 0.1
EPS_DECAY = 1000000 # 需要多少回合达到EPS_END

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 12_000_000
EVALUATE_FREQ = 100_000

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
os.mkdir(SAVE_PREFIX)

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    REMOVED_PARAM,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    "./model_040"        # 预训练模型的读入路径
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done: # 若一次生命结束，将该新一局的frame加入obs_queue中
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    # 每一轮由模型得到一个动作，动作输入环境得到下一帧及对应得分和是否结束
    # 最后将这些信息加入memory中
    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)

    # 每POLICY_UPDATE次进行一次取样训练
    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    # 每TARGET_UPDATE次进行一次target和policy的同步
    if step % TARGET_UPDATE == 0:
        agent.sync()

    # 每EVALUATE_FREQ次进行一次评价并写回文件中（游戏场景，训练得分，模型参数）
    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
