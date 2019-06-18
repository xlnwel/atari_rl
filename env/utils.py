import numpy as np


def encode_obs(idx, obs, done, frame_history_len, is_full, capacity):
    end_idx   = idx + 1 # make noninclusive
    start_idx = end_idx - frame_history_len
    # if there weren't enough frames ever in the buffer for context
    if start_idx < 0 and not is_full:
        start_idx = 0
    # we do not consider episodes whose length is less than n_step
    if not done[idx]:
        for i in range(start_idx, end_idx - 1):
            if done[i % capacity]:
                start_idx = i + 1
    missing_context = frame_history_len - (end_idx - start_idx)
    # if zero padding is needed for missing context
    # or we are on the boundry of the buffer
    if start_idx < 0 or missing_context > 0:
        frames = ([np.zeros_like(obs[0]) for _ in range(missing_context)]
                    + [obs[i] for i in range(start_idx, end_idx)])
        # for idx in range(start_idx, end_idx):
        #     frames.append(obs[idx])
        return np.concatenate(frames, 2)
    else:
        # this optimization has potential to saves about 30% compute time \o/
        h, w = obs.shape[1], obs.shape[2]
        return obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(h, w, -1)
