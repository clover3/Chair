import warnings
msg = '`layer.updates` will be removed in a future version. '
warnings.filterwarnings("ignore", msg)

warnings.warn('`layer.updates` will be removed in a future version. '
              'This property should not be used in TensorFlow 2.0, '
              'as `updates` are applied automatically.')

warnings.warn("other warning")
