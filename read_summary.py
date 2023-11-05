from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter('saved/')
r = 5
for i in range(100):

    lol = {
        'xcosx': i * np.cos(i / r),
        'tanx': np.tan(i / r)
    } if i % 5 == 0 else {}
    writer.add_scalars('run_14h', {
      'xsinx':i*np.sin(i/r),
                                 **lol
                               }, i)
writer.close()
# This call adds three values to the same scalar plot with the tag
# 'run_14h' in TensorBoard's scalar section.