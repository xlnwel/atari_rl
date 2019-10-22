In this project, we implement Rainbow and replace c51 in Rainbow with IQN.

## Algorithm Implemented

- [x] Rainbow
- [x] PER
- [x] Noisy Nets
- [x] Double
- [x] c51
- [x] Dueling nets
- [x] IQN

## Results

Basically all arguments are from reference papers, I did not take much time to fine-tune these arguments, since it takes too long to run a trial on atari.

### Rainbow-IQN on BreakoutNoFrameskip-v4

**Video**

![](results/rainbow-iqn-BreakoutNoFrameskip-v4.gif)

**Learning Curve**

Episodic rewards averaged over 100 episodes **at training time**.
<figure>
  <img src="results/rainbow-iqn-BreakoutNoFrameskip-learning-curve.png" alt="" width="1000">
  <figcaption></figcaption>
</figure>

Compare to [Google's Dopamine](https://github.com/google/dopamine) shown below, our implementation manages to achieving better performance on Breakout

<figure>
  <img src="results/dopamine-BreakoutNoFrameskip.png" alt="" width="1000">
  <figcaption></figcaption>
</figure>

Source: https://google.github.io/dopamine/baselines/plots.html, each iteration stands for 250000 steps.

## Running

```shell
# Silence tensorflow debug message
export TF_CPP_MIN_LOG_LEVEL=3

# By default, this line runs rainbow-iqn, which replaces c51 in rainbow with iqn
# For full argument specification, please refer to run/train.py
python run/train.py
```

## Details

All tests are done in PongNoFrameskip-v4 and BreakoutNoFrameskip-v4, 

1. Double Q nets, noisy layers, PER, multi-steps are used by default. 

3. Unlike the official implementation, we apply layer normalization to dense layers, instance normalization to conv layers, which could be designated by `conv_norm` and `dense_norm` in `algo/rainbow_iqn/args.yaml`

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
