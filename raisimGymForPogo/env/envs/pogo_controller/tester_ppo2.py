import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymForPogo.env.bin.pogo_controller import RaisimGymForSegway
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

env = VecEnv(RaisimGymForSegway(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
nMaps = env.num_ground
ground = 0
maxCommand = 3

weight_path = args.weight
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
command = np.zeros(3, dtype=np.float32)

gc = np.zeros(13, dtype=np.float32)
gv = np.zeros(12, dtype=np.float32)
info = np.zeros(76, dtype=np.float32)

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
                    if event.button == 0:
                        env.terrain_callback()
                        ground = (ground+1)%nMaps
                        maxCommand = env.calculateMaxSpeed(ground)
                        env.reset()
                        print("change env to {}. Max speed: {}".format(ground, maxCommand))

                    elif event.button == 1:
                        env.reset()
                        print("env reset")
                    elif event.button == 4:
                        print("Exiting loop")
                        running = False
                        break
            if not running:
                break

            if len(joysticks) > 0:
                command[0] = -maxCommand * joysticks[0].get_axis(1)
                command[2] = -2 * joysticks[0].get_axis(3)
                if(command[0] < -1.0):
                    command[0] = -1.0
#             print(command)
            env.set_command(command)
            obs = env.observe(False)
            with torch.no_grad():
                obs_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).to(device)
                action_ll = loaded_graph.forward(obs_tensor).squeeze(dim=0)
                env.step(action_ll.cpu().detach().numpy(), True)

            # plotting
            #env.get_state(gc, gv)
            env.get_logging_info(info)
            infoBag.append(info.copy())
            #gcBag.append(gc.copy())
            #gvBag.append(gv.copy())

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

        # fig 0 : Left leg torque, Right leg torque
        fig0 = plt.figure(0)
        ax0 = fig0.add_subplot(121)
        ax1 = fig0.add_subplot(122)
        ax0.plot(infoBag_np[:, 6], label='L_HFE')
        ax0.plot(infoBag_np[:, 7], label='L_KFE')
        ax0.plot(infoBag_np[:, 8], label='L_WHEEL')
        ax1.plot(infoBag_np[:, 9], label='R_HFE')
        ax1.plot(infoBag_np[:, 10], label='R_KFE')
        ax1.plot(infoBag_np[:, 11], label='R_WHEEL')
        ax0.axhline(y=0, color='k', linestyle='--')
        ax1.axhline(y=0, color='k', linestyle='--')
        ax0.legend()
        ax1.legend()
        ax0.set_title('Left Leg Torque')
        ax1.set_title('Right Leg Torque')
        ax0.set_ylim(-60, 60)
        ax1.set_ylim(-60, 60)

        # fig 1 : Left leg Pos, Vel
        fig1 = plt.figure(1)
        ax2 = fig1.add_subplot(121)
        ax3 = fig1.add_subplot(122)
        ax2.plot(infoBag_np[:, 23], label='L_HFE', color='tab:red')
        ax2.plot(infoBag_np[:, 24], label='L_KFE', color='tab:green')
        ax2.plot(infoBag_np[:, 35], label='L_HFE target', linestyle='dashed', color='tab:red')
        ax2.plot(infoBag_np[:, 36], label='L_KFE target', linestyle='dashed', color='tab:green')
        ax3.plot(infoBag_np[:, 29], label='L_HFE', color='tab:red')
        ax3.plot(infoBag_np[:, 30], label='L_KFE', color='tab:green')
        ax3.plot(infoBag_np[:, 31], label='L_WHEEL', color='tab:blue')
        ax3.plot(infoBag_np[:, 43], label='L_WHEEL target', linestyle='dashed', color='tab:blue')
        ax2.legend()
        ax3.legend()
        ax2.set_title('Left Leg Position')
        ax3.set_title('Left Leg Velocity')

        # fig 2 : Right leg Pos, Vel
        fig2 = plt.figure(2)
        ax4 = fig2.add_subplot(121)
        ax5 = fig2.add_subplot(122)
        ax4.plot(infoBag_np[:, 26], label='R_HFE', color='tab:red')
        ax4.plot(infoBag_np[:, 27], label='R_KFE', color='tab:green')
        ax4.plot(infoBag_np[:, 38], label='R_HFE target', linestyle='dashed', color='tab:red')
        ax4.plot(infoBag_np[:, 39], label='R_KFE target', linestyle='dashed', color='tab:green')
        ax5.plot(infoBag_np[:, 32], label='R_HFE', color='tab:red')
        ax5.plot(infoBag_np[:, 33], label='R_KFE', color='tab:green')
        ax5.plot(infoBag_np[:, 34], label='R_WHEEL', color='tab:blue')
        ax5.plot(infoBag_np[:, 46], label='R_WHEEL target', linestyle='dashed', color='tab:blue')
        ax4.legend()
        ax5.legend()
        ax4.set_title('Right Leg Position')
        ax5.set_title('Right Leg Velocity')

        # fig 3 : BASE angvel, linvel
        fig3 = plt.figure(3)
        ax6 = fig3.add_subplot(131)
        ax7 = fig3.add_subplot(132)
        ax8 = fig3.add_subplot(133)
        ax6.plot(infoBag_np[:, 20], label='v_x', color='tab:red')
        ax6.plot(infoBag_np[:, 21], label='v_y', color='tab:green')
        ax6.plot(infoBag_np[:, 22], label='v_z', color='tab:blue')
        ax7.plot(infoBag_np[:, 47], label='w_x', color='tab:red')
        ax7.plot(infoBag_np[:, 48], label='w_y', color='tab:green')
        ax7.plot(infoBag_np[:, 49], label='w_z', color='tab:blue')
        ax8.plot(infoBag_np[:, 50], label='r11', color='tab:red')
        ax8.plot(infoBag_np[:, 54], label='r22', color='tab:green')
        ax8.plot(infoBag_np[:, 58], label='r33', color='tab:blue')
        ax6.axhline(y=0, color='k', linestyle='--')
        ax7.axhline(y=0, color='k', linestyle='--')
        ax8.axhline(y=1, color='k', linestyle='--')
        ax6.legend()
        ax7.legend()
        ax6.set_title('Base Linear Velocity')
        ax7.set_title('Base Angular Velocity')
        ax8.set_title('Base Orientation')

        # fig 4 : joint_vel - torque trajectories
        fig4 = plt.figure(4)
        ax9 = fig4.add_subplot(121)
        ax10 = fig4.add_subplot(122)
        ax9.plot(infoBag_np[:, 29], infoBag_np[:, 6], label='L_HFE')
        ax9.plot(infoBag_np[:, 30], infoBag_np[:, 7], label='L_KFE')
        ax9.plot(infoBag_np[:, 31], infoBag_np[:, 8], label='L_WHEEL')
        ax10.plot(infoBag_np[:, 32], infoBag_np[:, 9], label='R_HFE')
        ax10.plot(infoBag_np[:, 33], infoBag_np[:, 10], label='R_KFE')
        ax10.plot(infoBag_np[:, 34], infoBag_np[:, 11], label='R_WHEEL')
        ax9.legend()
        ax10.legend()
        ax9.set_title('Left w - Torque')
        ax10.set_title('Right w - Torque')

        # fig 5 : Command Tracking
        fig5 = plt.figure(5)
        ax11 = fig5.add_subplot(121)
        ax12 = fig5.add_subplot(122)
        ax11.plot(infoBag_np[:, 20], label='v_x', color='tab:red')
        ax11.plot(infoBag_np[:, 59], label='v_x command', linestyle='dashed', color='tab:red')
        ax12.plot(infoBag_np[:, 49], label='w_z', color='tab:red')
        ax12.plot(infoBag_np[:, 61], label='w_z command', linestyle='dashed', color='tab:red')
        ax11.legend()
        ax12.legend()
        ax11.set_title('V_x command tracking')
        ax12.set_title('w_z command tracking')

        # fig 6 : Energy & Current calculate
        fig6 = plt.figure(6)
        ax13 = fig6.add_subplot(131)
        ax14 = fig6.add_subplot(132)
        ax15 = fig6.add_subplot(133)
        ax13.plot(infoBag_np[:, 62], label='L_HFE', color='tab:red')
        ax13.plot(infoBag_np[:, 63], label='L_KFE', color='tab:green')
        ax13.plot(infoBag_np[:, 64], label='L_WHEEL', color='tab:blue')
        ax13.plot(infoBag_np[:, 65], label='R_HFE', color='tab:red')
        ax13.plot(infoBag_np[:, 66], label='R_KFE', color='tab:green')
        ax13.plot(infoBag_np[:, 67], label='R_WHEEL', color='tab:blue')
        ax13.plot(infoBag_np[:, 68], label='Total', color='tab:orange')
        ax13.legend()
        ax13.set_title('T*w')

        ax14.plot(infoBag_np[:, 69], label='L_HFE', color='tab:red')
        ax14.plot(infoBag_np[:, 70], label='L_KFE', color='tab:green')
        ax14.plot(infoBag_np[:, 71], label='L_WHEEL', color='tab:blue')
        ax14.plot(infoBag_np[:, 72], label='R_HFE', color='tab:red')
        ax14.plot(infoBag_np[:, 73], label='R_KFE', color='tab:green')
        ax14.plot(infoBag_np[:, 74], label='R_WHEEL', color='tab:blue')
        ax14.plot(infoBag_np[:, 75], label='Total', color='tab:orange')
        ax14.legend()
        ax14.set_title('tau/Km ^2')

        ax15.plot((infoBag_np[:, 68] + 2*infoBag_np[:, 75])/57.6, label='Total current', color='tab:red')
        ax15.axhline(y=35, color='k', linestyle='--')

        # fig 7 : leg vel
        fig7 = plt.figure(7)
        ax16 = fig7.add_subplot(121)
        ax17 = fig7.add_subplot(122)
        ax16.plot(infoBag_np[:, 29], label='L_HFE')
        ax16.plot(infoBag_np[:, 30], label='L_KFE')
        ax16.plot(infoBag_np[:, 31], label='L_WHEEL')
        ax17.plot(infoBag_np[:, 32], label='R_HFE')
        ax17.plot(infoBag_np[:, 33], label='R_KFE')
        ax17.plot(infoBag_np[:, 34], label='R_WHEEL')
        ax16.axhline(y=0, color='k', linestyle='--')
        ax17.axhline(y=0, color='k', linestyle='--')
        ax16.legend()
        ax17.legend()
        ax16.set_title('Left Leg Velocity')
        ax17.set_title('Right Leg Velocity')
        ax16.set_ylim(-10, 20)
        ax17.set_ylim(-10, 20)


        mplcursors.cursor()

        plt.show()
