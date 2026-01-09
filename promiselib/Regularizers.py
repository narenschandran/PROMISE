import tensorflow as tf
import keras

@keras.saving.register_keras_serializable("promiselib")
class UnitNormRegularizer(tf.keras.regularizers.Regularizer):
    def __call__(self, x):
        sumsq = tf.reduce_sum(tf.math.square(x), axis=-1, keepdims=True)
        loss = tf.reduce_sum(tf.math.square(1 - sumsq))
        return loss

    def get_config(self):
        config = {}
        return config

@keras.saving.register_keras_serializable("promiselib")
class UnitNormMaxRegularizer(tf.keras.regularizers.Regularizer):
    def __call__(self, x):
        sumsq = tf.reduce_sum(tf.math.square(x), axis=-1, keepdims=True)
        loss = tf.reduce_sum(tf.keras.activations.relu(sumsq - 1))
        return loss

    def get_config(self):
        config = {}
        return config

@keras.saving.register_keras_serializable("promiselib")
class MinMaxValRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, min_val, max_val, name):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.name = name

    def __call__(self, x):
        smaller = tf.keras.activations.relu((tf.ones_like(x) * self.min_val) - x)
        larger = tf.keras.activations.relu(x - (tf.ones_like(x) * self.max_val))
        return tf.reduce_sum(smaller + larger)

    def get_config(self):
        config = {"min_val": self.min_val, "max_val": self.max_val, "name": self.name}
        return config
