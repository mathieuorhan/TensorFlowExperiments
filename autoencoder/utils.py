#!/usr/bin/env python
import numpy as np
import warnings

# Workaround from https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur
warnings.simplefilter(action='ignore', category=FutureWarning)

def add_noise(batch, noise_type):
    """Add noise to a batch
    
    Args:
        x (np.array): The uncorrumpted batch
        noise_type (string): either gaussian or mask, use : "gaussian-0.1" and "mask-0.4" for 
        a gaussian noise of std 0.1 or a mask a 40% of the batch
    
    Returns:
        np.array: The corrumpted batch
    """
    if 'gaussian' in noise_type:
        gaussian_std = float(noise_type.split('-')[1])
        n = np.random.normal(0, gaussian_std, np.shape(batch))
        return batch + n
    if 'mask' in noise_type:
        frac = float(noise_type.split('-')[1])
        temp = np.copy(batch)
        for i in temp:
            n = np.random.choice(len(i), round(
                frac * len(i)), replace=False)
            i[n] = 0
        return temp
    return batch
