#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:54:43 2020

@author: joshowen121
"""

import timeit


def test_normal(a, b):
    import numpy as np
    return a + b


def test_numpy(a, b):
    import numpy as np
    return np.add(a, b)


setup = """
import numpy as np
from __main__ import test_normal, test_numpy
a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
"""

code_normal = "test_normal(a, b)"
code_numpy = "test_numpy(a, b)"

normal_times = timeit.repeat(setup=setup, stmt=code_normal, repeat=3, number=10_000)
numpy_times = timeit.repeat(setup=setup, stmt=code_numpy, repeat=3, number=10_000)

print(f"Normal: {min(normal_times):.3f}")
print(f"Numpy: {min(numpy_times):.3f}")
