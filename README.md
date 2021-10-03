# UCL MSc Project :Stein Variational Policy Gradient (A Variational Reinforcement Learning Algorithm)

## 1. Outline

This work investigates whether the Variational Inference (VI) framework in Reinforcement Learn-ing (RL) problem can solve the long-existing problem about exploration-exploitation balance.  Inparticular,  this work explores the  adaptability of one  novel  VI method called Stein  VariationalGradient Descent in the context of policy-based RL algorithms and shows that the adaptions toexploit the VI yields a better balance between exploration and exploitation.  In recent years, manyresearchers have analysed the influence of hyperparameters (kernel, temperature annealing strate-gies) in the stein variational gradient descent, but few of them pay attention to its RL version.This thesis extends some works to the RL settings and examines their applicability.

## 2. Repo Structure 

- Agents: contain two RL algorithms Policy Gradient and Stein Variational Policy Gradient

- SVPG_utils.py: use to compute the kernel and derivative of kernel among different policy parameters. There are three available kernels Gaussian, Matern and Inverse Multi-Quadratic kernel.

- annealing.py: store three different temperature annealing strategies 

- gym_utils.py: parse gym environments

- hparams.py: save the hyper-parameters 

- model.py: Model Structure for Actor and Critic

- roll_out.py: Simulator 

- running_state.py: Standardise action and states of gym environments 

- simple_demo.py: start a simple runs with SVPG

  run python simple_demo.py

## 3. Dependency 

Dependencies are saved at requirements.txt. Apart from that, this repo depends on mujoco and mujoco-py. The installation guide line could be found at https://github.com/openai/mujoco-py and http://www.mujoco.org/



## 4. Experiments Environments 

- Hopper-v2

  <img src="/home/yiming/Documents/Msc_project/image_file/hopper.png" alt="zom" style="zoom:33%;" />

- Swimmer-v2

  ![](/home/yiming/Documents/Msc_project/image_file/swimmer.png)

- InvertedPendulum-v0

  <img src="/home/yiming/Documents/Msc_project/image_file/invertedpendulum.png" alt="z" style="zoom:67%;" />

- MountainCarContinuous-v0

  <img src="/home/yiming/Documents/Msc_project/image_file/mountaincar.png" alt="z" style="zoom:75%;" />

