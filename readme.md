## Algorithm Implemented

- [x] Rainbow   (except c51)
- [x] IQN

## Overall Architecture

<p align="center">
<img src="/results/Architecture.png" alt="average score in tensorboard" height="650">
</p>

Algorithms are implemented in [algo](https://github.com/xlnwel/atari_rl/tree/master/algo)

## Requirements

It is recommended to install Tensorflow from source following [this instruction](https://www.tensorflow.org/install/source) to gain some CPU boost and other potential benefits.

```shell
# Minimal requirements to run the algorithms. Tested on Ubuntu 18.04.2, using Tensorflow 1.13.1.
# Forget the deprecated warnings... This project is not designed according to Tensorflow 2.X
conda create -n gym python
conda activate gym
pip install -r requirements.txt
# install gym atari
pip install 'gym[atari]'
# Install tensorflow-gpu or install it from scratch as the above instruction suggests
pip install tensorflow-gpu
```

## Running

```shell
# Silence tensorflow debug message
export TF_CPP_MIN_LOG_LEVEL=3

# When running distributed algorithms, restrict numpy to one core
# Use numpy.__config__.show() to ensure your numpy is using OpenBlas
# For MKL and detailed reasoning, refer to [this instruction](https://ray.readthedocs.io/en/latest/example-rl-pong.html?highlight=openblas#the-distributed-version)
export OPENBLAS_NUM_THREADS=1

# For full argument specification, please refer to run/train.py
python run/train.py
```

## Details

All tests are done in [PongNoFrameskip-v4](https://gym.openai.com/envs/Pong-v0/), 
1. A hard-won one-week lesson: 
    1. convolutional layers use same padding
    2. bias is neither used in convolutional layers nor fully connected layers

2. Zero states are used as terminal states.

3. Arguments from homework3 of UCB [cs294-112](http://rail.eecs.berkeley.edu/deeprlcourse/).

## Paper References

Dan Horgan et al. Distributed Prioritized Experience Replay 

Hado van Hasselt et al. Deep Reinforcement Learning with Double Q-Learning

Tom Schaul et al. Prioritized Experience Replay

Meire Fortunato et al. Noisy Networks For Exploration

Ziyu Wang et la. Dueling Network Architectures for Deep Reinforcement Learning

Will Dabney et al. Implicit Quantile Networks for Distributional Reinforcement Learning

## Code References

[Homework of Berkeley CS291-112](http://rail.eecs.berkeley.edu/deeprlcourse/)

[Google Dopamine](https://github.com/google/dopamine)