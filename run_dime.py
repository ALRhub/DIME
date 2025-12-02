import os
import jax
import time
import hydra
import wandb
import omegaconf
import traceback

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

def _create_alg(cfg: DictConfig):
    import gymnasium as gym
    try:
        import myosuite
    except ImportError:
        print("myosuite not installed")
        pass

    # training_env = gym.make(cfg.env_name)
    # eval_env = make_vec_env(cfg.env_name, n_envs=1, seed=cfg.seed)
    cfg.env_name = [cfg.env_name] if isinstance(cfg.env_name, str) else cfg.env_name
    env_name_split = cfg.env_name[0].split('/')

    training_env = make_dummy_env(cfg.env_name)
    eval_env = make_dummy_env(cfg.env_name, seed=42)
    rb_class = None
    # if env_name_split[0] == 'dm_control':
    #     rb_class = DMCCompatibleDictReplayBuffer if env_name_split[1].split('-')[0] in ['humanoid', 'fish', 'walker', 'quadruped','finger'] else None

    tensorboard_log_dir = f"./logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/seed= + {str(cfg.seed)}/"
    eval_log_dir = f"./eval_logs/{cfg.wandb['group']}/{cfg.wandb['job_type']}/seed= + {str(cfg.seed)}/eval/"

    save_path = './checkpoints'
    if os.environ.get('SLURM_SUBMIT_DIR'):
        save_path = '/pfs/work9/workspace/scratch/ka_et4232-restored/ka_et4232-tcx-1763778846/checkpoints/dime'
    env_path = ''
    for env_name in cfg.env_name:
        env_path += env_name.split('-')[-2] + '_'
    env_path = env_path[:-1]
    save_path = save_path + f'/{env_path}/{cfg.seed}'

    os.makedirs(save_path, exist_ok=True)

    policy = "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy"
    # model = DIME(
    #     policy,
    #     env=training_env,
    #     model_save_path=save_path,
    #     save_every_n_steps=int(cfg.tot_time_steps / 10),
    #     cfg=cfg,
    #     tensorboard_log=tensorboard_log_dir,
    #     replay_buffer_class=rb_class
    # )

    model = MTDIME(
        policy,
        env=training_env,
        model_save_path=save_path,
        save_every_n_steps=int(cfg.tot_time_steps / 10),
        cfg=cfg,
        tensorboard_log=tensorboard_log_dir,
    )
    # Create log dir where evaluation results will be saved
    os.makedirs(eval_log_dir, exist_ok=True)
    # Create callback that evaluates agent

    eval_callback = EvalCallback(
        eval_env,
        jax_random_key_for_seeds=cfg.seed,
        best_model_save_path=None,
        log_path=eval_log_dir,
        eval_freq=max(300000 // cfg.log_freq, 1),
        n_eval_episodes=5, deterministic=True, render=False
    )
    if cfg.wandb["activate"]:
        callback_list = CallbackList([eval_callback, WandbCallback(verbose=0, )])
    else:
        callback_list = CallbackList([eval_callback])
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
    model, callback_list = _create_alg(cfg)
    model.learn(total_timesteps=cfg.tot_time_steps, progress_bar=True, callback=callback_list)


@hydra.main(version_base=None, config_path="configs", config_name="slurm_mtbase")
def main(cfg: DictConfig) -> None:
    starting_time = time.time()
    if cfg.use_jit:
        initialize_and_run(cfg)
    else:
        with jax.disable_jit():
            initialize_and_run(cfg)
    end_time = time.time()
    print(f"Training took: {(end_time - starting_time) / 3600} hours")
    if cfg.wandb["activate"]:
        wandb.finish()
    # try:
    #     starting_time = time.time()
    #     if cfg.use_jit:
    #         initialize_and_run(cfg)
    #     else:
    #         with jax.disable_jit():
    #             initialize_and_run(cfg)
    #     end_time = time.time()
    #     print(f"Training took: {(end_time - starting_time)/3600} hours")
    #     if cfg.wandb["activate"]:
    #         wandb.finish()
    # except Exception as ex:
    #     print("-- exception occured. traceback :")
    #     traceback.print_tb(ex.__traceback__)
    #     print(ex, flush=True)
    #     print("--------------------------------\n")
    #     traceback.print_exception(ex)
    #     if cfg.wandb["activate"]:
    #         wandb.finish()


if __name__ == "__main__":
    main()
