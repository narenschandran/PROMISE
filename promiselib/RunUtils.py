import os
from copy import deepcopy
import re
import numpy as np
import pandas as pd

import tensorflow as tf
import keras

from promiselib.model import PROMISE as Mod
from promiselib.Utils import *

libdir     = os.path.dirname(__file__)
projroot   = os.path.join(libdir, '..')
prereq_dir = os.path.join(projroot, 'prereq')
mdl_dir    = os.path.join(projroot, 'results', '01-ablation-study')

mdl_f     = os.path.join(mdl_dir, 'p3000', 'model-tse-adcca349',
                         'seed-00', 'model.keras')

#------------------------------------------------------------------------------#
#                       Input processing and validation                        #
#------------------------------------------------------------------------------#

#------------------------------------------------#
#                Input processing                #
#------------------------------------------------#

def model_load(modfn, model_f):
    mod = tf.keras.models.load_model(model_f, custom_objects={modfn.__name__: modfn})
    return mod


def ModelLoad():
    mod = model_load(Mod, mdl_f)
    return mod

