# MLP with adaptive LR scheduler
task_name = "pogo_MLP"

from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymForPogo.env.bin.pogo_controller2 import RaisimGymForPogo
from raisimGymForPogo.env.bin.pogo_controller2 import NormalSampler
from raisimGymForPogo.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymForPogo.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
import os
import math
import time
import raisimGymForPogo.algo.ppo.module as ppo_module
import raisimGymForPogo.algo.ppo.ppo_KL as PPO
import torch.nn as nn
import numpy as np
import torch
import argparse
import datetime

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode
weight_path = args.weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

reward_list = cfg['environment']['reward']
reward_coeff = []
for key, value in reward_list.items():
    reward_coeff.append(value)

# create environment from the configuration file
env = VecEnv(RaisimGymForPogo(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
value_ob_dim = env.num_value_obs
act_dim = env.num_acts

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['mlp3_shape'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)

critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['mlp3_shape'], nn.LeakyReLU, value_ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymForPogo/data/" + task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp", task_path + "/PogoController.hpp",
                                       task_path + "/RandomHeightMapGenerator.hpp", task_path + "/runner_MLP.py",
                                       task_path + "/../../RaisimGymVecEnv.py", task_path + "/../../VectorizedEnvironment.hpp", task_path + "/../../debug_app.cpp"])

tensorboard_launcher(saver.data_dir + "/..")  # press refresh (F5) after the first ppo update

ppo = PPO.PPO(actor=actor,
               critic=critic,
               num_envs=cfg['environment']['num_envs'],
               num_transitions_per_env=n_steps,
               num_learning_epochs=4,
               gamma=0.996,
               lam=0.95,
               num_mini_batches=4,
               device=device,
               log_dir=saver.data_dir,
               shuffle_batch=False,
               )

iteration_number = 0


def reset():
    env.reset()


if mode == 'retrain':
    iteration_number = load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)
    # load observation scaling from files of pre-trained model
    weight_dir = weight_path.rsplit('/', 1)[0] + '/'
    env.full_load_scaling(weight_dir, iteration_number, env.num_envs * iteration_number * n_steps)
    for curriculum_update in range(int(iteration_number / cfg['environment']['curriculum']['iteration_per_update'])):
        env.curriculum_callback()

for update in range(iteration_number, 50001):
    start = time.time()
    reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.

    if update % cfg['environment']['iteration_per_save'] == 0:
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(),
            'actor_distribution_state_dict': actor.distribution.state_dict(),
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt')
        env.save_scaling(saver.data_dir, str(update))

    # if update % cfg['environment']['eval_every_n'] == 0:
        data_tags = env.get_step_data_tag()
        data_size = 0
        data_mean = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_square_sum = np.zeros(shape=(len(data_tags), 1), dtype=np.double)
        data_min = np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)
        data_max = -np.inf * np.ones(shape=(len(data_tags), 1), dtype=np.double)

        # we create another graph just to demonstrate the save/load method
        loaded_graph = ppo_module.MLP(cfg['architecture']['mlp3_shape'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        for step in range(n_steps):
            with torch.no_grad():
                obs = env.observe(False)
                # actions, _ = actor.sample(torch.from_numpy(np.expand_dims(obs, axis=0)).to(device))
                actions = loaded_graph.architecture(torch.from_numpy(obs).cpu()).detach().cpu().numpy()
                reward, dones = env.step_visualize(actions)
                data_size = env.get_step_data(data_size, data_mean, data_square_sum, data_min, data_max)

        data_std = np.sqrt((data_square_sum - data_size * data_mean * data_mean) / (data_size - 1 + 1e-16))
        data_std = np.sqrt(np.clip(data_square_sum - data_size * data_mean**2, 1e-16, None) / (data_size - 1 + 1e-16))


        for data_id in range(len(data_tags)):
            #ppo.writer.add_scalar('generalized_reward/' + data_tags[data_id], data_mean[data_id] / reward_coeff[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id] + '/mean', data_mean[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id] + '/std', data_std[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id] + '/min', data_min[data_id], global_step=update)
            ppo.writer.add_scalar(data_tags[data_id] + '/max', data_max[data_id], global_step=update)

        reset()


    for step in range(n_steps):
        with torch.no_grad():
            obs = env.observe(update < 10000)
            value_obs = env.get_value_obs(update < 10000)
            action = ppo.act(obs)
            reward, dones = env.step(action)
            ppo.step(value_obs=value_obs, rews=reward, dones=dones)
            done_sum = done_sum + np.sum(dones)
            reward_ll_sum = reward_ll_sum + np.sum(reward)
            torch.cuda.empty_cache()


    # take st step to get value obs
    value_obs = env.get_value_obs(update < 10000)
    ppo.update(actor_obs=obs, value_obs=value_obs, log_this_iteration=update % cfg['environment']['iteration_per_log'] == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    actor.update()

    if update % cfg['environment']['curriculum']['iteration_per_update'] == 0:
        env.curriculum_callback()

    if update % cfg['environment']['iteration_per_log'] == 0:
        ppo.writer.add_scalar('Training/average_reward', average_ll_performance, global_step=update)
        ppo.writer.add_scalar('Training/dones', average_dones, global_step=update)
        ppo.writer.add_scalar('Training/command tracking curriculum', data_mean[14], global_step=update)
        ppo.writer.add_scalar('Training/command tracking reward', data_mean[0], global_step=update)

    end = time.time()

    if update % 10 == 0:
        print('----------------------------------------------------')
        print('{:>6}th iteration'.format(update))
        print('{:<40} {:>6}'.format("average ll reward: ", '{:0.6f}'.format(average_ll_performance)))
        print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
        print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
        print('----------------------------------------------------\n')
