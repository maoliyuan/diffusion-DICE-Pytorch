# Diffusion-DICE
Official implementation for [Diffusion-DICE: In-Sample Diffusion Guidance for Offline Reinforcement Learning](https://arxiv.org/pdf/2407.20109) [NeurlIPS 2024]. Code are based on PyTorch.

Liyuan Mao\*, Haoran Xu\*, Weinan Zhang†, Xianyuan Zhan, Amy Zhang†

\*equal contribution, †equal advising

## Usage

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://www.roboti.us/download.html), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. [Wandb](https://docs.wandb.ai/quickstart) are used for logging.

### Running
Before running main_diffusion_DICE.py, please pre-train the diffusion behavior policy by running:
``` Bash
bash pretrain_behavior.sh
```
To reproduce the experiments on D4RL MuJoCo locomotion datasets and AntMaze navigation datasets, please run:
``` Bash
python main_diffusion_DICE.py --env_name {your_env_name} --seed {your_seed} --actor_load_path /{your_behavior_ckpt_folder}/behavior_ckpt{your_ckpt_epoch}_seed{your_ckpt_seed} --inference_sample {your_inference_sample_num} --alpha {your_alpha} 
```

To ensure training stability, you can adjust `batch_size`. We also support `CosineAnnealingLR` schedule, which is configured with `use_lr_schedule` and `min_value_lr`.


## Citation

Please cite our paper as:

```

```

## License

MIT
