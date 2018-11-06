import math

"""
Function that returns the learning rate adjusted according to the specified schedule.
Based on "luong234" function in TensorFlow NMT implementation
(https://github.com/tensorflow/nmt/blob/master/nmt/model.py)
Applies lr warmup during the first num_warmup_iterations, starts halving the lr at
decay_start_iteration. At most 4 decays will be applied.

Parameters:
  iteration:                  current iteration
  learning_rate_warmup_steps: number of iterations over which lr is warmed up to the base_lr
  decay_start:                iteration at which the first lr decay is applied
  decay_interval:             number of iterations between lr decays
"""
def get_lr(iteration, learning_rate_warmup_steps, decay_start, decay_interval, base_lr):
  if iteration < learning_rate_warmup_steps:
    warmup_factor = math.exp(math.log(0.01) / num_warmup_iterations)
    inv_decay = warmup_factor ** (learning_rate_warmup_steps - iteration)
    return base_lr * inv_decay
  elif iteration < decay_start:
    return base_lr
  else: # iteration >= decay_start_iteration:
    max_num_decays = 4
    decay_factor = 0.5
    num_decay_steps = min(int((iteration - decay_start) / decay_interval) + 1, max_num_decays)
    return base_lr * (decay_factor ** num_decay_steps)

"""
num_warmup_iterations = 5
decay_start_iteration = 20
decay_interval = 5
base_lr = 0.1

learning_rates = []

n = 50
learning_rates = [get_lr(i, num_warmup_iterations, decay_start_iteration, decay_interval, base_lr)
                  for i in range(1, n)]

import matplotlib.pyplot as plt

plt.plot(learning_rates, 'b')
plt.show()
"""
