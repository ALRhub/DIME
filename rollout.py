import os
import jax
import time
import hydra
import wandb
import omegaconf
import traceback
import pandas

from common.buffers import DMCCompatibleDictReplayBuffer
from common.envs import make_dummy_env
from diffusion.dime import DIME
from omegaconf import DictConfig
from models.utils import is_slurm_job
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from models.actor_critic_evaluation_callback import EvalCallback
from diffusion.mt_dime import MTDIME
from pandas import DataFrame, read_csv

def _create_alg(cfg: DictConfig):
    import gymnasium as gym
    try:
        import myosuite
    except ImportError:
        print("myosuite not installed")
        pass
    cfg.env_name = [cfg.env_name] if isinstance(cfg.env_name, str) else cfg.env_name
    env_name_split = cfg.env_name[0].split('/')

    training_env = make_dummy_env(cfg.env_name)
    rb_class = None
    # if env_name_split[0] == 'dm_control':
    #     rb_class = DMCCompatibleDictReplayBuffer if env_name_split[1].split('-')[0] in ['humanoid', 'fish', 'walker', 'quadruped','finger'] else None

    tensorboard_log_dir = f"./logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/seed= + {str(cfg.seed)}/"
    eval_log_dir = f"./eval_logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/seed= + {str(cfg.seed)}/eval/"


    policy = "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy"

    model = MTDIME(
        policy,
        env=training_env,
        model_save_path=None,
        save_every_n_steps=int(cfg.tot_time_steps / 10),
        cfg=cfg,
        tensorboard_log=tensorboard_log_dir,
    )
    save_path = './checkpoints'
    if os.environ.get('SLURM_SUBMIT_DIR'):
        save_path = '/pfs/work9/workspace/scratch/ka_et4232-restored/ka_et4232-tcx-1763778846/checkpoints/dime'
    save_path = save_path + f'/{cfg.env_name}/{cfg.seed}'
    model.load_model(save_path,'100000', '100000')
    callback_list = None

    return model, callback_list


def initialize_and_run(cfg):
    cfg = hydra.utils.instantiate(cfg)
    seed = cfg.seed
    if cfg.wandb["activate"]:
        name = f"{str(cfg.env_name)}_{seed}"
        wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb["project"],
            group=cfg.wandb["group"],
            job_type=cfg.wandb["job_type"],
            name=name,
            config=wandb_config,
            entity=cfg.wandb["entity"],
            sync_tensorboard=True,
        )
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')
    model, callback_list, df = _create_alg(cfg)
    model.learn(total_timesteps=cfg.tot_time_steps, progress_bar=True, callback=callback_list)

    if os.environ.get('SLURM_SUBMIT_DIR') is not None:
        submit_dir = os.environ.get('SLURM_SUBMIT_DIR')
    else:
        submit_dir = '.'
    out_path = submit_dir + '/summary.csv'
    if not os.path.exists(out_path):
        summary = DataFrame(columns=['task', 'num_seeds', 'goal', 'return'])
    else:
        summary = read_csv(out_path)

    entry_list = []
    to_remove_idx = []
    for index, row in summary.iterrows():
        if row['task'] in cfg.env_name and row['seed'] == cfg.seed:
            to_remove_idx.append(index)

    df = summary.drop(to_remove_idx)
    mean_rewards = model.replay_buffer.rewards[:model.replay_buffer.pos].mean(1)
    entry_list = {'tasks': cfg.env_name, f'seed{cfg.seed} mean_rewards':mean_rewards, **summary.iloc[to_remove_idx[0]]}
    df = pandas.concat((df,entry_list), ignore_index=True)
    df.to_csv(out_path, index=False)



@hydra.main(version_base=None, config_path="configs", config_name="slurm_base")
def main(cfg: DictConfig) -> None:
    try:
        starting_time = time.time()
        if cfg.use_jit:
            initialize_and_run(cfg)
        else:
            with jax.disable_jit():
                initialize_and_run(cfg)
        end_time = time.time()
        print(f"Training took: {(end_time - starting_time)/3600} hours")
        if cfg.wandb["activate"]:
            wandb.finish()
    except Exception as ex:
        print("-- exception occured. traceback :")
        traceback.print_tb(ex.__traceback__)
        print(ex, flush=True)
        print("--------------------------------\n")
        traceback.print_exception(ex)
        if cfg.wandb["activate"]:
            wandb.finish()


if __name__ == "__main__":
    main()
