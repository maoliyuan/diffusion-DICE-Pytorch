import argparse
import yaml
import gym
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="walker2d-medium-replay-v2")
    parser.add_argument("--seed", default=0, type=int)             
    parser.add_argument("--device", default="cuda", type=str)      
    parser.add_argument("--device_num", default=0, type=int)
    parser.add_argument('--actor_load_path', type=str, default=None)
    parser.add_argument('--inference_sample', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_lr_schedule', type=int, default=0)
    parser.add_argument('--min_value_lr', type=float, default=1e-4)
    print("**************************")
    args = parser.parse_args()
    env_type = "antmaze" if "antmaze" in args.env else "mujoco"
    with open(f"configs/{env_type}.yaml", "r") as file:
        config = yaml.safe_load(file)
    for key, value in config.items():
        setattr(args, key, value)
    print(args)
    return args

def pallaral_eval_policy(policy_fn, env_name, evaluation_num, seed, diffusion_steps=15):
    ori_eval_envs = []
    for i in range(evaluation_num):
        eval_env = gym.make(env_name)
        eval_env.seed(seed + 1001 + i)
        ori_eval_envs.append(eval_env)
    eval_envs = [env for env in ori_eval_envs]
    for env in eval_envs:
        env.buffer_state = env.reset()
        env.buffer_return = 0.0
    while len(eval_envs) > 0:
        new_eval_envs = []
        states = np.stack([env.buffer_state for env in eval_envs])
        actions = policy_fn(states, diffusion_steps=diffusion_steps)
        for i, env in enumerate(eval_envs):
            state, reward, done, info = env.step(actions[i])
            env.buffer_return += reward
            env.buffer_state = state
            if not done:
                new_eval_envs.append(env)
        eval_envs = new_eval_envs
    return ori_eval_envs
