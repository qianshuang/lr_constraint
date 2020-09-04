# -*- coding: utf-8 -*-

import numpy as np

theta = np.mat(np.ones(5))
theta[0, 0] = 2
print(np.sort(theta, axis=-1))
print(theta)
