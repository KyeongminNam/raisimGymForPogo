## raisim_env_anymal

### How to use this repo
Read the instruction of raisimGym at raisim.com 

### Prerequisites
- Ubuntu 18~22
- Raisim: clone somewhere on your system
- Python + anaconda. Pip will automatically install other dependencies
- Latest GPU driver, Cuda and pytorch
- Copy ``rsc/activation.raisim`` file to ``<home>/<user_id>/.raisim`` directory.

### Train
1. Compile raisimgym: ```python setup develop --CMAKE_PREFIX_PATH <RAISIM_DIRECTORY>/raisim/linux```
2. Run runner.py of the task: ```cd raisimGymForSegway/env/envs/segwayBig && python ./runner.py```

### Test policy
1. Compile raisimgym: ```python setup develop --CMAKE_PREFIX_PATH <RAISIM_DIRECTORY>/raisim/linux```
2. Run tester.py of the task with policy: 
   * ```cd raisimGymForSegway/env/envs/segwayBig```
   * ```python tester.py --weight <POLICY_DIRECTORY>/full_XXX.pt```

### Debugging
1. Compile raisimgym with debug symbols: ```python setup develop --Debug```. This compiles <YOUR_APP_NAME>_debug_app
2. Run it with Valgrind. I strongly recommend using Clion for 

## TODO:

- [x] Fill in other side of the URDF
- [x] Currently, links are joined in a pin-point by making pulses towards each other every step. Make sure this is stable
- [x] Ease the environment to make the learning better. Step sizes should be 1/3 of the wheel size
- [ ] Train the agent, and see if they learn to follow the points.
- [ ] Check what to do with self-contact at controller logic
- [x] Fill in previous observations
- [x] Recover normalization