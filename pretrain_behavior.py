import os
import gym
import tqdm
import functools
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_DICE.loss import loss_fn
from diffusion_DICE.schedule import marginal_prob_std
from diffusion_DICE.model import ScoreNet
from utils import get_args
from dataset import D4RL_dataset

def train_behavior(args, score_model, data_loader):
    def datas_():
        while True:
            yield from data_loader
    datas = datas_()
    n_epochs = 600
    tqdm_epoch = tqdm.trange(0, n_epochs)
    save_interval = 100
    
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)

    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for _ in range(1000):
            data = next(datas)
            data = {k: d.to(args.device) for k, d in data.items()}

            s = data['s']
            a = data['a']
            score_model.condition = s
            loss = loss_fn(score_model, a, args.marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            score_model.condition = None

            avg_loss += loss
            num_items += 1
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            
        if (epoch % save_interval == (save_interval - 1)) or epoch == 599:
            torch.save(score_model.state_dict(), os.path.join("./models_rl", str(args.env), f"behavior_ckpt{epoch+1}_seed{args.seed}.pth"))
        wandb.log({"actor/loss": avg_loss / num_items})

def main(args):
    # The diffusion behavior training pipeline is copied directly from https://github.com/ChenDRAG/CEP-energy-guided-diffusion/blob/main/Offline_RL_2D/train_behavior.py
    if not os.path.exists("./models_rl"):
        os.makedirs(dir)
    if not os.path.exists(os.path.join("./models_rl", str(args.env))):
        os.makedirs(os.path.join("./models_rl", str(args.env)))
    
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    wandb.init(project=f"diffusion_DICE_behavior",
                name=f"{args.env}",
                config={
                    "env_name": args.env,
                    "seed": args.seed,
                    "n_behavior_epochs": args.n_behavior_epochs,
                    "diffusion_steps": args.diffusion_steps,
                })
    marginal_prob_std_fn = functools.partial(marginal_prob_std, device=args.device)
    args.marginal_prob_std_fn = marginal_prob_std_fn
    score_model= ScoreNet(input_dim=state_dim+action_dim, output_dim=action_dim, marginal_prob_std=marginal_prob_std_fn, args=args).to(args.device)
    
    dataset = D4RL_dataset(args)
    data_loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    print("training behavior")
    train_behavior(args, score_model, data_loader)
    print("finished")

if __name__ == "__main__":
    args = get_args()
    main(args)