In this project, we implement Rainbow and replace c51 in Rainbow with IQN.

## Algorithm Implemented

- [x] Rainbow
- [x] PER
- [x] Noisy Nets
- [x] Double
- [x] c51
- [x] Dueling nets
- [x] IQN

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

# By default, this line runs rainbow-iqn, which replaces c51 in rainbow with iqn
# For full argument specification, please refer to run/train.py
python run/train.py
```

## Details

All tests are done in [PongNoFrameskip-v4](https://gym.openai.com/envs/Pong-v0/), 
1. A hard-won one-week lesson: 
    1. convolutional layers use same padding
    2. bias is neither used in convolutional layers nor fully connected layers

2. Double Q nets, noisy layers, PER, multi-steps are used by default. 

3. Zero states are used as terminal states.

4. Best arguments are kept in `args.yaml`. Most arguments are from the reference paper, learning rate schedule is from homework3 of UCB [cs294-112](http://rail.eecs.berkeley.edu/deeprlcourse/).

5. I modify the network a little bit, by adding a dense layer before dueling heads. This saves more than 2/3 parameters(10 million vs 36 million, which is mainly induced by the combination of dueling heads and noisy layers). Forthermore, it mitigates overfitting on some environments such as breakout.

6. Background learning is initially designed for Ape-X, which I happened to find out works extremely well on (continuous environments](https://github.com/xlnwel/model-free-algorithms) that do not require a deep net. However, it does no work that well with Atari games. There are two potential reason I can conjecture:
    1. Background learning increase the learning frequency, which makes it more likely overfit and get stuck at a local optimum. Methods such as regularization and disentanglement might help alleviate the issue.
    2. I use a new thread to do background learning, which actually does not achieve parallelism because of Python's GIL. This actually slows down the interaction with environment, and exacerbate model overfitting. Multi-processing achieved in Ape-X alleviates this issue.

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