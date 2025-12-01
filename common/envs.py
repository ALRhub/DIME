from typing import List, Any, Optional, Sequence, Union, Type

import numpy as np
import gymnasium as gym
import random

from dm_control.suite.ball_in_cup import catch
from gymnasium import spaces
# from envpool.mujoco.dmc.registration import task_name
from gymnasium.wrappers import FlattenObservation, RescaleAction, TimeLimit
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnvIndices
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.utils import set_random_seed

class MultitaskDummy(DummyVecEnv):
    def __init__(self, env_fns, num_tasks):
        super().__init__(env_fns)
        self.num_tasks = num_tasks
        assert not self.num_envs % self.num_tasks

    # def step(self, actions: np.ndarray) -> VecEnvStepReturn:
    #     obs, rewards, dones, infos = super().step(actions)
    #     # obs = np.concatenate((obs, self.task_ids), axis=0)
    #     return obs, rewards, dones, infos


class FlattenObservationShadowhandWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        subspaces = env.observation_space
        obs_box  = subspaces['observation']
        goal_box = subspaces['desired_goal']
        low  = np.concatenate((obs_box.low,  goal_box.low),  axis=-1)
        high = np.concatenate((obs_box.high, goal_box.high), axis=-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_box.dtype)

    def observation(self, obs: dict) -> np.ndarray:
        return np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)

def make_dummy_env(names, seed=0, repeat_envs=1):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.utils import set_random_seed
    set_random_seed(seed)
    names = [names] if isinstance(names, str) else list(names)
    if repeat_envs > 1:
        names = sorted(names * repeat_envs) #sort to make sure the same envs are adjacent
    envs = DummyVecEnv([lambda: make_env(name, seed+i) for i,name in enumerate(names)])
    envs.reset()
    return envs

def make_video_dummy_env(names,path, interval,record_trigger=None, seed=0, repeat_envs=1,):
    env = make_dummy_env(names,seed,repeat_envs)
    if not record_trigger:
        record_trigger = lambda x: x % 2000
    return VecVideoRecorder(env,path,record_trigger, interval)

def _make_env_dmc(env_name: str, seed: int = 0, num_env=1) -> gym.Env:
    if num_env != 1:
        # import  envpool
        # robot, task = env_name.split('-')
        # robot = robot.capitalize()
        # task = task.capitalize()
        # n = robot + task + '-v1'
        # env = envpool.make_dm(n, num_envs=num_env, seed=seed)
        # env.action_space
        raise NotImplementedError('vectorized env for DMC not implemented')
    from dm_control import suite
    suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
    suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
    from shimmy import DmControlCompatibilityV0
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={'random': seed},
    )
    env = DmControlCompatibilityV0(env, render_mode='rgb_array', render_kwargs={'camera_id': 1})
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    env = RescaleAction(env, -1.0, 1.0)
    return env

# def _make_env_metaworld(env_name: str, seed: int = 0, num_env=1) -> gym.Env:
#     if num_env != 1:
#         raise NotImplementedError('vectorized env for metaworld not implemented')
#     try:
#         constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
#     except:
#         from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
#         constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
#     env = constructor(seed=int(seed))
#     env = TimeLimit(env, 200)
#     return env
    
def _make_env_humanoidbench(env_name: str, seed: int = 0, num_env=1) -> gym.Env:
    import humanoid_bench
    from humanoid_bench.env import ROBOTS, TASKS
    if num_env > 1:
        env = gym.make_vec(env_name, num_envs=num_env,vectorization_mode="sync")
    else:
        env = gym.make(env_name, autoreset=False)
    return env

# def _make_env_shadowhand(env_name: str, seed: int = 0, num_env=1) -> gym.Env:
#     if num_env != 1:
#         raise NotImplementedError('vectorized env for shadowhand not implemented')
#     import dex_envs
#     env_version = 'v1'
#     reward_type = 'sparse'
#     name = f'{env_name}-rotate-{env_version}'
#     env = gym.make(name, reward_type=reward_type)
#     env = FlattenObservationShadowhandWrapper(env)
#     return env

def make_env(env_name: str, seed: int = 0, num_env=1) -> gym.Env:
    env = None
    suite, task = env_name.split('/')
    if suite == 'metaworld':
        env = _make_env_metaworld(task, seed, num_env)
    elif suite == 'hb':
        env = _make_env_humanoidbench(task, seed, num_env)
    elif suite == 'dm_control':
        env = _make_env_dmc(task, seed, num_env)
    else:
        env = _make_env_shadowhand(task, seed, num_env)
    return env

class ParallelEnv():
    def __init__(self, env_names: list, seed: int = 0):
        
        np.random.seed(seed)
        random.seed(seed)
        
        envs = []
        obs_dims = np.zeros(len(env_names), dtype=np.int32)
        act_dims = np.zeros(len(env_names), dtype=np.int32)
        for i, env_name in enumerate(env_names):
            envs.append(make_env(env_name, seed))
            obs_dims[i] = envs[-1].observation_space.shape[0]
            act_dims[i] = envs[-1].action_space.shape[0]

        max_state_dim = int(np.max(obs_dims))
        max_action_dim = int(np.max(act_dims))
        state_dim_differences = max_state_dim - obs_dims

        
        observation_space = gym.spaces.Box(low=(np.ones(max_state_dim, dtype=np.float64)[None, :] - np.inf).repeat(len(envs), axis=0),
                                           high=(np.ones(max_state_dim, dtype=np.float64)[None, :] + np.inf).repeat(len(envs), axis=0),
                                           shape=(len(envs), max_state_dim),
                                           dtype=envs[-1].observation_space.dtype)        
                
        action_space = gym.spaces.Box(low=(np.ones(max_action_dim, dtype=np.float64)[None, :] * -1).repeat(len(envs), axis=0),
                                      high=(np.ones(max_action_dim, dtype=np.float64)[None, :]).repeat(len(envs), axis=0),
                                      shape=(len(envs), max_action_dim),
                                      dtype=envs[-1].action_space.dtype)
    
        
        self.envs = envs
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.state_dim_differences = state_dim_differences
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_tasks = len(envs)
        
    def _reset_idx(self, idx: int):
        seed = np.random.randint(0, 1e8)
        state, _ = self.envs[idx].reset(seed=seed)
        state = np.concatenate((state, np.zeros(self.state_dim_differences[idx], dtype=np.float32)), axis=0)
        return state
    
    def generate_masks(self, terminals: np.ndarray, truncates: np.ndarray):
        masks = 1 - (terminals * (1 - truncates))
        return masks
    
    def reset_where_done(self, states: np.ndarray, terminals: np.ndarray, truncates: np.ndarray):
        for j, (terminal, truncate) in enumerate(zip(terminals, truncates)):
            if (terminal == True) or (truncate == True):
                states[j], terminals[j], truncates[j] = self._reset_idx(j), False, False
        return states, terminals, truncates
    
    def reset(self):
        states = []
        for i, env in enumerate(self.envs):
            states.append(self._reset_idx(i))
        return np.stack(states)

    def _get_goal(self, info: dict):
        if 'success' in info:
            goal = info['success']
        elif 'is_success' in info:
            goal = info['is_success']
        elif 'solved' in info:
            goal = info['solved']
        else:
            goal = 0
        return goal
        
    def step(self, actions: np.ndarray):
        states, rewards, terminals, truncates, goals = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminal, truncate, info = env.step(action[:self.act_dims[i]])
            state = np.concatenate((state, np.zeros(self.state_dim_differences[i], dtype=np.float64)), axis=0)
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            truncates.append(truncate)
            goals.append(self._get_goal(info))
        return np.stack(states), np.stack(rewards), np.stack(terminals), np.stack(truncates), np.stack(goals)   

    def evaluate(self, agent, num_episodes, temperature=0.0, render=False, max_render_steps=5000, render_frameskip=4):
        n_rollouts = np.zeros(self.num_tasks)
        returns = np.zeros(self.num_tasks)
        goals = np.zeros(self.num_tasks)
        mask = np.ones(self.num_tasks)
        mask_goals = np.ones(self.num_tasks)
        observations = self.reset()
        if render:
            renders = []
        i = 0
        while True:
            if render:
                if i % render_frameskip == 0:
                    if i < max_render_steps:
                        env_renders = self.render()
                        renders.append(env_renders)
            actions = agent.sample_actions(observations, temperature=temperature)
            #actions = envs.action_space.sample()
            next_observations, rewards, terms, truns, success = self.step(actions)
            returns += rewards * mask
            goals += success * mask_goals
            mask_goals = np.where(success, 0, mask_goals)
            mask_goals = np.where(np.logical_or(terms, truns), 1, mask_goals)
            observations = next_observations
            n_rollouts += np.logical_or(terms, truns)
            observations, terms, truns = self.reset_where_done(observations, terms, truns)
            mask = np.where(n_rollouts >= num_episodes, 0, 1)
            i += 1
            if n_rollouts.min() == num_episodes:
                break
        if render:
            renders = np.stack(renders)
            renders = np.transpose(renders, (1, 0, 4, 2, 3))
            return {'goal': goals/num_episodes, 'return': returns/num_episodes, 'renders': renders}
        else:
            return {'goal': goals/num_episodes, 'return': returns/num_episodes}

    def render(self):
        renders = []
        for i, env in enumerate(self.envs):
            render = env.render()
            renders.append(render)
        renders = np.stack(renders)
        return renders




class ParallelVecEnv(VecEnv, ParallelEnv):

    def __init__(self, env_names, seed: int = 0, render_mode='rgb_array'):
        if isinstance(env_names, str):
            env_names = [env_names]
        ParallelEnv.__init__(self, env_names, seed)
        self.reward = None
        self.dones = None
        self.infos = None
        self.state = None
        self.num_envs = len(self.envs)
        self.attrs = {'render_mode': [render_mode] * self.num_envs}
        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low[0],
            high=self.observation_space.high[0],
            shape=[self.observation_space.shape[-1]],
            dtype=self.observation_space.dtype)

        self.action_space = gym.spaces.Box(
            low=self.action_space.low[0],
            high=self.action_space.high[0],
            shape=[self.action_space.shape[-1]],
            dtype=self.action_space.dtype)
        VecEnv.__init__(self, self.num_envs,self.observation_space, self.action_space)

        self.outs = None

    def reset(self) -> VecEnvObs:
        return ParallelEnv.reset(self)

    def step_async(self, actions: np.ndarray) -> None:
        states, rewards, dones, infos = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminal, truncate, info = env.step(action[:self.act_dims[i]])
            state = np.concatenate((state, np.zeros(self.state_dim_differences[i], dtype=np.float64)), axis=0)
            states.append(state)
            rewards.append(reward)
            dones.append(np.logical_or(terminal, truncate))
            infos.append(info)
        self.state, self.reward, self.dones, self.infos = np.stack(states), np.stack(rewards), np.stack(dones), infos


    def step_wait(self) -> VecEnvStepReturn:
        return self.state, self.reward, self.dones, self.infos

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        try:
            if indices is None:
                return self.attrs[attr_name]
            elif isinstance(indices, int):
                return [self.attrs[attr_name][indices]]
            else:
                return self.attrs[attr_name][indices]
        except AttributeError :
            print(f'Env has no {attr_name}, or indices wrong')

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        if indices is None:
            self.attrs[attr_name] = value
        else:
            self.attrs[attr_name][indices] = value

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        # if indices is None:
        #     indices = range(self.num_envs)
        # for i in indices:
        #     self.envs[i]
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        VecEnv.seed(self, seed)
        for i, env in enumerate(self.envs):
            env.seed(self._seeds[i])

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util
        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def close(self):
        for env in self.envs:
            env.close()