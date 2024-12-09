import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymForPogo.env.bin.pogo_controller import RaisimGymForPogo
from raisimGymForPogo.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import raisimGymForPogo.algo.ppo2.module as ppo_module
import os
import math
import torch
import argparse
import pygame
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(f"[torch] cuda:{torch.cuda.device_count()} detected.")

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
cfg['environment']['render'] = True
cfg['environment']['curriculum']['initial_factor'] = 1

env = VecEnv(RaisimGymForPogo(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
nMaps = env.num_ground
ground = 0
maxCommand = 2

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
command = np.zeros(3, dtype=np.float32)

gc = np.zeros(13, dtype=np.float32)
gv = np.zeros(12, dtype=np.float32)
info = np.zeros(23, dtype=np.float32)

# plotting
infoBag = []
gcBag = []
gvBag = []

if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))

    pygame.init()
    pygame.joystick.init()
    joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    for joystick in joysticks:
        print("detected" + joystick.get_name())

    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)
    loaded_graph = ppo_module.GRU_MLP_Actor(ob_dim - 3,
                                            cfg['architecture']['hidden_dim'],
                                            cfg['architecture']['mlp_shape'],
                                            act_dim,
                                            env.num_envs,
                                            device)
    loaded_graph.load_state_dict(torch.load(weight_path, map_location=device)['actor_architecture_state_dict'])
    loaded_graph.init_hidden()

    env.load_scaling(weight_dir, int(iteration_number))
    env.turn_on_visualization()

    try:
        running = True
        while running:
            frame_start = time.time()
            for event in pygame.event.get():  # User did something.
                if event.type == pygame.JOYBUTTONDOWN:  # If user clicked close.
                    if event.button == 1:
                        env.reset()
                        infoBag = []
                        print("env reset")
                    elif event.button == 4:
                        print("Exiting loop")
                        running = False
                        break
            if not running:
                break

            if len(joysticks) > 0:
                command[0] = -maxCommand * joysticks[0].get_axis(1)
                command[1] = -maxCommand * joysticks[0].get_axis(0)
                command[2] = -2 * joysticks[0].get_axis(3)
            print(command)
            env.set_command(command)
            env.visualizeArrow()
            obs = env.observe(False)
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
                action_ll = loaded_graph.forward(obs_tensor).squeeze(dim=0)
                env.step(action_ll.cpu().detach().numpy(), True)

            # plotting
            env.get_logging_info(info)
            infoBag.append(info.copy())


            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    except KeyboardInterrupt:
        print("Loop exited on button press")

    finally:
        env.turn_off_visualization()

        import mplcursors
        infoBag_np = np.array(infoBag)

        # fig 0 : GC, GV, GF
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(131)
        ax1 = fig0.add_subplot(132)
        ax2 = fig0.add_subplot(133)
        ax0.plot(infoBag_np[:, 0], label='R1')
        ax0.plot(infoBag_np[:, 1], label='R2')
        ax0.plot(infoBag_np[:, 2], label='P')
        ax0.set_title('GC')

        ax1.plot(infoBag_np[:, 3], label='R1')
        ax1.plot(infoBag_np[:, 4], label='R2')
        ax1.plot(infoBag_np[:, 5], label='P')
        ax1.set_title('GV')

        ax2.plot(infoBag_np[:, 6], label='R1')
        ax2.plot(infoBag_np[:, 7], label='R2')
        ax2.plot(infoBag_np[:, 8], label='P')
        ax2.set_title('GF')

        # fig 1 : Command Tracking
        fig1 = plt.figure(1)
        ax3 = fig1.add_subplot(131)
        ax4 = fig1.add_subplot(132)
        ax5 = fig1.add_subplot(133)
        ax3.plot(infoBag_np[:, 9], label='v_x', color='tab:red')
        ax3.plot(infoBag_np[:, 15], label='v_x command', linestyle='dashed', color='tab:red')
        ax4.plot(infoBag_np[:, 10], label='v_y', color='tab:red')
        ax4.plot(infoBag_np[:, 16], label='v_y command', linestyle='dashed', color='tab:red')
        ax5.plot(infoBag_np[:, 14], label='w_z', color='tab:red')
        ax5.plot(infoBag_np[:, 17], label='w_z command', linestyle='dashed', color='tab:red')
        ax3.legend()
        ax4.legend()
        ax5.legend()
        ax3.set_title('V_x command tracking')
        ax4.set_title('V_y command tracking')
        ax5.set_title('w_z command tracking')

        mplcursors.cursor()

        plt.show()