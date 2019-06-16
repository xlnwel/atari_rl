## Algorithm Implemented

- [x] Rainbow   (except c51)
- [x] IQN

## Overall Architecture

<p align="center">
<img src="/results/Architecture.png" alt="average score in tensorboard" height="650">
</p>

Algorithms are implemented in [algo](https://github.com/xlnwel/atari_rl/tree/master/algo)

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
1. A hard-won two-week lesson: 
    1. convolutional layers use same padding
    2. bias is neither used in convolutional layers nor fully connected layers

2. Zero states are used as terminal states.

3. Arguments from homework3 of UCB [cs294-112](http://rail.eecs.berkeley.edu/deeprlcourse/).