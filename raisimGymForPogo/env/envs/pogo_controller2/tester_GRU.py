import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymForPogo.env.bin.pogo_controller2 import RaisimGymForPogo
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
cfg['environment']['curriculum']['cmd_initial_factor'] = 1.0

env = VecEnv(RaisimGymForPogo(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
nMaps = env.num_ground
ground = 0
maxCommand = 1.5

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
command = np.zeros(3, dtype=np.float32)
deadzone = 0.5


info = np.zeros(30, dtype=np.float32)

# plotting
infoBag = []


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
                    if event.button == 0:
                        env.terrain_callback()
                        env.reset()
                        infoBag = []
                        print("reset and terrain change")
                    elif event.button == 1:
                        env.reset()
                        infoBag = []
                        time.sleep(1.0)
                        print("reset")
                    elif event.button == 4:
                        print("Exiting loop")
                        running = False
                        break
            if not running:
                break


            if len(joysticks) > 0:
                axis_y = joysticks[0].get_axis(1)
                axis_x = joysticks[0].get_axis(0)
                axis_z = joysticks[0].get_axis(3)

                command[0] = -maxCommand * axis_y if abs(axis_y) > deadzone else 0
                command[1] = -maxCommand * axis_x if abs(axis_x) > deadzone else 0
                command[2] = -1 * axis_z if abs(axis_z) > deadzone else 0
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
        time_array = np.arange(0, len(infoBag_np) * 0.01, 0.01)


        # fig 0 : GC, GV, GF
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(412)
        ax1 = fig0.add_subplot(413)
        ax2 = fig0.add_subplot(414)
        ax3 = fig0.add_subplot(411)
        ax0.plot(time_array, infoBag_np[:, 0], label='R1', color='tab:red')
        ax0.plot(time_array, infoBag_np[:, 1], label='R2', color='tab:green')
        ax0.plot(time_array, infoBag_np[:, 2], label='P', color='tab:blue')
        ax0.set_ylabel('Joint Position', rotation=90, labelpad=50, va='center')
        ax0.legend(loc='upper right')

        ax1.plot(time_array, infoBag_np[:, 3], label='R1', color='tab:red')
        ax1.plot(time_array, infoBag_np[:, 4], label='R2', color='tab:green')
        ax1.plot(time_array, infoBag_np[:, 5], label='P', color='tab:blue')
        ax1.set_ylabel('Joint Velocity', rotation=90, labelpad=50, va='center')
        ax1.legend(loc='upper right')


        ax2.plot(time_array, infoBag_np[:, 6], label='R1', color='tab:red')
        ax2.plot(time_array, infoBag_np[:, 7], label='R2', color='tab:green')
        ax2.plot(time_array, infoBag_np[:, 8], label='P', color='tab:blue')
        ax2.set_ylabel('Joint Torque', rotation=90, labelpad=50, va='center')
        ax2.set_xlabel('Time [s]')
        ax2.legend(loc='upper right')


        ax3.plot(time_array, infoBag_np[:, 29], color='tab:red')
        ax3.set_ylabel('Base Height', rotation=90, labelpad=50, va='center')



        # fig 1 : Command Tracking
        fig1 = plt.figure(1)
        fig1.suptitle('Command Tracking Performance', fontsize=16)
        ax4 = fig1.add_subplot(311)
        ax5 = fig1.add_subplot(312)
        ax6 = fig1.add_subplot(313)
        ax4.plot(time_array, infoBag_np[:, 12], label='v_x', color='tab:red')
        ax4.plot(time_array, infoBag_np[:, 21], label='v_x command', linestyle='dashed', color='tab:red')
        ax4.legend(loc='upper right')

        ax5.plot(time_array, infoBag_np[:, 13], label='v_y', color='tab:red')
        ax5.plot(time_array, infoBag_np[:, 22], label='v_y command', linestyle='dashed', color='tab:red')
        ax5.legend(loc='upper right')

        ax6.plot(time_array, infoBag_np[:, 20], label='w_z', color='tab:red')
        ax6.plot(time_array, infoBag_np[:, 23], label='w_z command', linestyle='dashed', color='tab:red')
        ax6.legend(loc='upper right')
        ax6.set_xlabel('Time [s]')


        mplcursors.cursor()

        plt.show()