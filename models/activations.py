from tensorflow import math
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


# ======================= activation functions ===================

def gelu_(X):
    """GELU activation function (approximation)."""
    return 0.5 * X * (1.0 + math.tanh(0.7978845608028654 * (X + 0.044715 * math.pow(X, 3))))


def snake_(X, beta):
    """Snake activation function: X + (1/beta) * sin^2(beta * X)."""
    return X + (1 / beta) * math.square(math.sin(beta * X))


# ======================= activation layers ======================

class GELU(Layer):
    """
    Gaussian Error Linear Unit (GELU), an alternative to ReLU.

    Reference:
        Hendrycks & Gimpel, 2016. Gaussian error linear units (GELUs).
        arXiv:1606.08415.

    Usage: Y = GELU()(X)
    """

    def __init__(self, trainable=False, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self.supports_masking = True
        self.trainable = trainable

    def build(self, input_shape):
        super(GELU, self).build(input_shape)

    def call(self, inputs, mask=None):
        return gelu_(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(GELU, self).get_config()
        return {**base_config, 'trainable': self.trainable}


class Snake(Layer):
    """
    Snake activation function: X + (1/beta) * sin^2(beta * X).
    Proposed for learning periodic targets.

    Reference:
        Ziyin et al., 2020. Neural networks fail to learn periodic functions and how to fix it.
        arXiv:2006.08195.

    Usage: Y = Snake(beta=0.5, trainable=False)(X)
    """

    def __init__(self, beta=0.5, trainable=False, **kwargs):
        super(Snake, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = beta
        self.trainable = trainable

    def build(self, input_shape):
        self.beta_factor = K.variable(self.beta, dtype=K.floatx(), name='beta_factor')
        if self.trainable:
            self._trainable_weights.append(self.beta_factor)
        super(Snake, self).build(input_shape)

    def call(self, inputs, mask=None):
        return snake_(inputs, self.beta_factor)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(Snake, self).get_config()
        return {**base_config,
                'beta': self.get_weights()[0] if self.trainable else self.beta,
                'trainable': self.trainable}