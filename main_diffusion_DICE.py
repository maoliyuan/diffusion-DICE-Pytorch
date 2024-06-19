import os
import gym
import d4rl
import copy
import time
import tqdm
import functools
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from diffusion_DICE.schedule import marginal_prob_std
from diffusion_DICE.model import ScoreNet
from utils import get_args, pallaral_eval_policy
from dataset import D4RL_dataset

def evaluation_subprocess(v_model_key, marginal_prob_std_fn, args, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_num}"
    eval_results = {}
    score_model= ScoreNet(input_dim=args.sdim+args.adim, output_dim=args.adim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    print("loading actor for evaluation...")
    actor_ckpt = torch.load(args.actor_load_path, map_location=args.device)
    score_model.load_state_dict(actor_ckpt)
    print("loading guidance network for evaluation...")
    score_model.v[0].to(args.device)
    v_path = os.path.join("./models_rl", str(args.env), f"critic_ckpt_temp_{v_model_key}.pth")
    v_ckpt = torch.load(v_path, map_location=args.device)
    score_model.v[0].load_state_dict(v_ckpt)

    for env_type in list(args.s.keys()):
        if env_type in args.env:
            guidance_scale = args.s[env_type]
    score_model.v[0].guidance_scale = guidance_scale
    envs = args.eval_func(score_model.select_actions)
    normalized_returns = d4rl.get_normalized_score(args.env, np.array([envs[i].buffer_return for i in range(args.seed_per_evaluation)])) * 100.0
    mean, std = np.mean(normalized_returns), np.std(normalized_returns)
    print("normalized return mean: ", mean, "normalized return std: ", std)
    eval_results.update({"normalized return mean": mean, "normalized return std": std})
    os.remove(v_path) # remove temporary guidance network
    result_queue.put(eval_results)

def run_diffusion_DICE(args, score_model, data_loader):
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    tqdm_epoch = tqdm.trange(0, args.train_epoch)
    evaluation_interval = args.evaluation_interval
    save_interval = 20
    result_queue = torch.multiprocessing.Queue()

    for epoch in tqdm_epoch:
        tot_guidance_loss = 0.
        num_items = 0
        for mini_batch in range(10000):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}
            score_model.v[0].update_v0(data)
            guidance_loss = score_model.v[0].update_wt(data)
            tot_guidance_loss += guidance_loss
            num_items += 1
            if mini_batch % 1000 == 0:
                tqdm_epoch.set_description('Average Guidance Loss: {:5f}'.format(tot_guidance_loss / num_items))
        if args.use_lr_schedule:
            for scheduler in score_model.v[0].schedulers:
                scheduler.step()
        if (epoch % evaluation_interval == (evaluation_interval -1)) or epoch==0:
            # generate key to identify model under present main process
            v_model_key = str(time.time())
            process = torch.multiprocessing.Process(target=evaluation_subprocess, args=(copy.deepcopy(v_model_key), args.marginal_prob_std_fn, args, result_queue))
            if len(torch.multiprocessing.active_children()) > 1:
                torch.multiprocessing.active_children()[0].join()
            print('saving temporary guidance model for evaluation...')
            torch.save(score_model.v[0].state_dict(), os.path.join("./models_rl", str(args.env), f"critic_ckpt_temp_{v_model_key}.pth"))
            process.start()
            while not result_queue.empty():
                eval_result = result_queue.get()
                wandb.log(eval_result, step=epoch * 10000 + num_items)
        if args.save_model and ((epoch % save_interval == (save_interval - 1)) or epoch==0):
            torch.save(score_model.v[0].state_dict(), os.path.join("./models_rl", str(args.env), f"critic_ckpt{epoch+1}_seed{args.seed}.pth"))
        wandb.log({"w_t loss": tot_guidance_loss / num_items}, step=epoch * 10000 + num_items)

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_num}"
    if not os.path.exists("./models_rl"):
        os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.env))):
        os.makedirs(os.path.join("./models_rl", str(args.env)))
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.eval_func = functools.partial(pallaral_eval_policy, env_name=args.env, evaluation_num=args.seed_per_evaluation, seed=args.seed, diffusion_steps=args.diffusion_steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    args.adim, args.sdim = action_dim, state_dim
    
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    score_model.v[0].to(args.device)

    print("loading actor")
    ckpt = torch.load(args.actor_load_path, map_location=args.device)
    score_model.load_state_dict(ckpt)
    print("finish loading actor")
    
    print("preparing dataset")
    dataset = D4RL_dataset(args)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print("dataset ready")

    wandb.init(project=f"diffusion_DICE_{args.env}",
                name=f"{args.env}",
                config={
                    "env_name": args.env,
                    "seed": args.seed,
                    "twin_v": args.twin_v,
                    "inference_sample": args.inference_sample,
                    "support_action_num": args.M,
                    "batch_size": args.batch_size,
                    "value_lr": args.value_lr,
                    "wt_lr": args.wt_lr,
                    "weight_decay": args.weight_decay,
                    "alpha": args.alpha,
                    "q_ensemble_num": args.q_ensemble_num,
                    "use_lr_schedule": args.use_lr_schedule,
                    "min_value_lr": args.min_value_lr,
                })

    print("running diffusion-DICE")
    run_diffusion_DICE(args, score_model, data_loader)
    print("finished")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = get_args()
    main(args)