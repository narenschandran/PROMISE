import os
import re
import time

import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import BinaryCrossentropy

from promiselib.RunUtils import model_load
from promiselib.Utils import (
        writeDict, readDict,
        MSE, BCE, AUC, PRAUC,
        savePickle, readPickle
)

script_dir    = os.path.dirname(__file__)
projroot      = os.path.join(script_dir, "..")

#------------------------------------------------------------------------------#
#                               Custom Callbacks                               #
#------------------------------------------------------------------------------#
def DefaultEarlyStopping():
    """Early stopping on validation loss to prevent overfitting"""
    return EarlyStopping(
        min_delta=0.001,
        patience=5,
        monitor="val_loss",
        restore_best_weights=True,
    )


#------------------------------------------------------------------------------#
#                                Model training                                #
#------------------------------------------------------------------------------#
# We have a reasonably sophisticated system to ensure that model training is   #
# not repeated if we can detected instances of prior training. Our system also #
# provides the options to track checkpoints and use Tensorboard using the      #
# callbacks provided by Tensorflow. We also configure a default early stopping #
# callback (based on the callback provided by Tensorflow) to prevent           #
# overfitting.                                                                 #
#                                                                              #
# Out intention with this system is to use EarlyStopping to figure out the     #
# best hyperparameter set in a time-efficient manner, followed by running      #
# the best hyperparameter set without early stopping for a large number of     #
# epochs, where we track the performance across all epochs.                    #
#------------------------------------------------------------------------------#

def perf_report_generate(mod, dat, seed, batch_size = 32):
    '''Gets performance for gene-label models'''

    split_lst = []
    perf_dct = {k: [] for k, v in dat.items()}
    for k, v in dat.items():
        inp, out = v
        pred = mod.predict(inp, batch_size = batch_size)
        split_lst.append(k)
        perf_dct[k].append(BCE(out[0], pred))
        perf_dct[k].append(AUC(out[0], pred))
        perf_dct[k].append(PRAUC(out[0], pred))
        perf_dct[k] = np.round(np.asarray(perf_dct[k]), 3)


    meas = ['BCE', 'AUC', 'PRAUC']
    meas_datf  = pd.DataFrame({'measure': meas}, index = meas)
    perf_datf0 = pd.DataFrame(perf_dct, index = meas)
    perf_datf1 = pd.DataFrame({
        'worst-case' : [
            np.round(perf_datf0.loc['BCE',:].max(), 3),
            np.round(perf_datf0.loc['AUC',:].min(), 3),
            np.round(perf_datf0.loc['PRAUC',:].min(), 3)
        ]
    }, index = meas)

    perf = pd.concat([meas_datf, perf_datf0, perf_datf1], axis = 1)
    perf.loc[:,"seed"] = seed
    return perf

def best_model_from_chkpts(chkpt_dir):
    '''Gets the best model from a checkpoints directory based on the loss value.'''
    def parse_fname(f):
        bname = os.path.basename(f)
        fbase = re.sub("[.]keras$", "", bname)
        epoch, loss = fbase.split("-")
        return epoch, float(loss)

    fs = [os.path.join(chkpt_dir, fname) for fname in os.listdir(chkpt_dir)]
    fs.sort()

    epochs = []
    losses = []
    fpaths = []

    for f in fs:
        epoch, loss = parse_fname(f)
        epochs.append(epoch)
        losses.append(loss)

    datf = pd.DataFrame({"epochs": epochs, "loss": losses, "f": fs})
    return datf.sort_values("loss").iloc[0].loc["f"]


def model_train(modfn, params, data, seed, base_outdir=None,
                epochs=500, verbose=2, early_stopping=False,
                checkpoints=False, tensorboard=False, force=False,
                batch_size = 256):
    '''
    Wrapper function to set up model training

    Args:
        modfn          : The model class (used to instantiate the model).
        params         : Arguments passed to the model class.
        data           : Dictionary with train-val-test splits of the dataset.
        base_outdir    : Base output directory.
        epochs         : Maximum number of epochs used for the training process.
        verbose        : Control verbosity during model fitting.
        early_stopping : Enable/disable early stopping to prevent overfitting.
        checkpoints    : Enable/disable saving model checkpoints. If enabled,
                         the model that is reported in the output directory
                         will be the one with the best performance.
        tensorboard    : Enable/disable production of Tensorboard reports
        force          : Perform model training from scratch even if prior
                         model exists.
    '''
    def data_check(data):
        splits = ["train", "val", "test"]
        for split in splits:
            if split not in data.keys():
                raise ValueError(f"The [{split}] split is not present")

        return data["train"], data["val"], data["test"]

    # Check [1]: Ensure that the train, val and test splits
    # are present in the input data.
    data_check(data)
    traininp, trainout = data["train"]


    # print("@@@@@@@@@@")
    # traininp = [inp[:1000]for inp in traininp]
    # trainout = [out[:1000]for out in trainout]
    # data["val"][0] = [inp[:1000]for inp in data["val"][0]]
    # data["val"][1] = [inp[:1000]for inp in data["val"][1]]
    # print("@@@@@@@@@@")
    val = data["val"]

    if base_outdir is not None:
        mdl_name = modfn(**params).model_name
        outdir = os.path.join(base_outdir, mdl_name, f"seed-{seed:02d}")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        perf_file  = os.path.join(outdir, "perf.tsv")
        conf_file  = os.path.join(outdir, "conf.dct")
        model_file = os.path.join(outdir, "model.keras")
        hist_file  = os.path.join(outdir, "hist.pkl")
        time_file  = os.path.join(outdir, "time.txt")

        if (not force) and (os.path.exists(perf_file) and os.path.exists(model_file) and os.path.exists(conf_file) and os.path.exists(hist_file)):
            mod  = model_load(modfn, model_file)
            hist = readPickle(hist_file)
            perf = pd.read_csv(perf_file, sep = '\t')

            print("")
            print("-----------------------------------------------------------")
            print("The existence of the following files indicates the presence")
            print("of a pretrained model. If you want to do a fresh training")
            print("delete these files and continue:")
            print(perf_file)
            print(model_file)
            print(conf_file)
            print("-----------------------------------------------------------")
            print("")

            pcols = ['measure', 'train', 'val', 'test', 'worst-case', 'seed']
            print(perf.loc[:,pcols])
            return hist, perf, mod
    else:
        outdir = None

    if checkpoints and (base_outdir is None):
        raise ValueError("Specify base_outdir if you want to store checkpoints")

    if tensorboard and (base_outdir is None):
        raise ValueError("Specify base_outdir if you want to use Tensorboard")

    callbacks = []
    if early_stopping:
        callbacks.append(DefaultEarlyStopping())

    if checkpoints and (base_outdir is not None):
        chkptdir = os.path.join(outdir, "chkpt")
        if not os.path.exists(chkptdir):
            os.makedirs(chkptdir)
        chkptfile = os.path.join(chkptdir, "{epoch:04d}-{val_loss:0.4f}.keras")
        callbacks.append(ModelCheckpoint(chkptfile))

    if tensorboard and (base_outdir is not None):
        logdir = os.path.join(outdir, "logs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        callbacks.append(TensorBoard(logdir, histogram_freq=1))

    # Set seed once before model initialization
    tf.keras.utils.set_random_seed(seed)
    lossfn = [BinaryCrossentropy()]
    lossfn[0].name = 'bce'

    mod = modfn(**params)
    mod.compile(optimizer="adam", loss=lossfn)

    # Setting seed once again before training starts for
    # the sake of safety.
    start = time.time()
    tf.keras.utils.set_random_seed(seed)
    hist = mod.fit(
        traininp,
        trainout,
        validation_data=val,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
        batch_size = batch_size
    )
    end = time.time()
    elaps = end - start
    time_str = f"Model training time: {elaps:0.2f} seconds"
    print(time_str)
    if base_outdir is not None:
        with open(time_file, 'w') as f:
            f.writelines(time_str + '\n')

    if checkpoints:
        best_model_f = best_model_from_chkpts(chkptdir)
        print(f"Selecting best model from checkpoints:\n[{best_model_f}]")
        mod = model_load(modfn, best_model_f)

    perf = perf_report_generate(mod, data, seed)
    conf = mod.get_config()

    if outdir is not None:
        mod.save(model_file)
        writeDict(perf, perf_file)
        perf.to_csv(perf_file, sep = '\t', index = False)
        writeDict(conf, conf_file)
        savePickle(hist, hist_file)


        # Just testing that it can load the model
        tmp = model_load(modfn, model_file)
        del tmp
    pcols = ['measure', 'train', 'val', 'test', 'worst-case', 'seed']
    print(perf.loc[:,pcols])
    return hist, perf, mod
