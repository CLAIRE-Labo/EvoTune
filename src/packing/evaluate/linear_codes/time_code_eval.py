import numpy as np
from packing.utils.seeding import seed_everything
from packing.logging.function_class import FunctionClass
from argparse import Namespace
import time as time
from packing.evaluate.error_correcting_codes.custom_init import custom_init
from typing import Type
import logging
import copy
import random
from sage.all import *
from omegaconf import OmegaConf
import time

k = 32
n = 256
dist = 256
# Check time of final matrix evaluation
function_class = FunctionClass()
G_new_numpy = np.array(random_matrix(GF(2), k, n))

G_new = matrix(GF(2), G_new_numpy)

code = LinearCode(G_new)
# start_time = time.time()
# minimum_distance = code.minimum_distance()
# print(minimum_distance)
# end_time = time.time()
# print("Minimum distance ", int(end_time - start_time), "seconds")

start_time = time.time()
weight_distribution = code.weight_distribution()
end_time = time.time()
print("Weight distr ", int(end_time - start_time), "seconds")


