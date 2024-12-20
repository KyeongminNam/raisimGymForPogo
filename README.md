## raisimGymForPogo
Official implementation of *POGO-LOCO*(Pogostick Locomotion)  
A final project submission of KAIST’s CS672 Reinforcement Learning Class.

Check out our presentation here: [[Youtube_Link](https://www.youtube.com/watch?v=m0u2zGS2Mmg&t=123s)]


### Contributors (Team 3):  
- Kyeongmin Nam (Masters Student, KAIST)  
- Hyungho Choi (Masters Student, KAIST)  
- Donghyuk Choi (Undergraduate Student, KAIST)  

---

### Prerequisites
- Ubuntu 18~22
- Raisim : https://raisim.com/sections/Installation.html
- Python + anaconda. Pip will automatically install other dependencies
- Latest GPU driver, Cuda and pytorch
- Copy ``rsc/activation.raisim`` file to ``<home>/<user_id>/.raisim`` directory.

### Train
1. Compile raisimgym: ```python setup develop --CMAKE_PREFIX_PATH <RAISIM_DIRECTORY>/raisim/linux```
2. Run runner.py of the task: ```cd raisimGymForPogoy/env/envs/pogo_controller2 && python ./runner.py```

### Test policy
1. Compile raisimgym: ```python setup develop --CMAKE_PREFIX_PATH <RAISIM_DIRECTORY>/raisim/linux```
2. Run tester.py of the task with policy: 
   * ```cd raisimGymForPogo/env/envs/pogo_controller2```
   * ```python tester.py --weight <POLICY_DIRECTORY>/full_XXX.pt```
3. Download our policy at: 
   - [[our_policy](https://github.com/KyeongminNam/raisimGymForPogo/blob/main/raisimGymForPogo/policies/original/full_22000.pt)]
   - [[ablation 1](https://github.com/KyeongminNam/raisimGymForPogo/blob/main/raisimGymForPogo/policies/no_priv/full_22000.pt)]
   - [[ablation 2](https://github.com/KyeongminNam/raisimGymForPogo/blob/main/raisimGymForPogo/policies/MLP/full_22000.pt)]

### Debugging
1. Compile raisimgym with debug symbols: ```python setup develop --Debug```. This compiles <YOUR_APP_NAME>_debug_app
2. Run it with Valgrind. I strongly recommend using Clion.
