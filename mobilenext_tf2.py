import tensorflow as tf
from tensorflow.keras import layers

BN_MOMENTUM = 0.999
BN_EPSILON = 1e-3

def _make_divisible(channels, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_channels = max(min_value, int(channels+divisor/2)//divisor*divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels

class SandGlassBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, stride, reduction, name='block'):
        super(SandGlassBlock, self).__init__(name=name)

        self.conv = tf.keras.Sequential([
            # depthwise
            layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON),
            layers.ReLU(6.),
            # pointwise reduction
            layers.Conv2D(filters=in_channels//reduction, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON),
            # pointwise expansion
            layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON),
            layers.ReLU(6.),
            # depthwise
            layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same' if stride == 1 else 'valid', use_bias=False),
            layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON),
        ])

        self.residual = (in_channels == out_channels and stride == 1)

    def call(self, x, training=True):
        if self.residual:
            return self.conv(x, training=training) + x
        else:
            return self.conv(x, training=training)

class MobileNeXt(tf.keras.Model):
    config = [
        # channels, stride, reduction, blocks
        [96,   2, 2, 1],
        [144,  1, 6, 1],
        [192,  2, 6, 3],
        [288,  2, 6, 3],
        [384,  1, 6, 4],
        [576,  2, 6, 4],
        [960,  1, 6, 2],
        [1280, 1, 6, 1]
    ]

    def __init__(self, input_size=224, num_classes=1000, width_mult=1., name='mobilenext'):
        super(MobileNeXt, self).__init__(name=name)

        stem_channels = 32
        stem_channels = _make_divisible(int(stem_channels*width_mult), 8)
        self.conv_stem = tf.keras.Sequential([
            layers.Conv2D(filters=stem_channels, kernel_size=3, strides=2, padding='valid', use_bias=False),
            layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON),
            layers.ReLU(6.)
        ], name='conv_stem')

        self.blocks = tf.keras.Sequential()
        in_channels = stem_channels
        for i, (c, s, r, b) in enumerate(self.config):
            out_channels = _make_divisible(int(c*width_mult), 8)
            for j in range(b):
                stride = s if j == 0 else 1
                self.blocks.add(SandGlassBlock(in_channels, out_channels, stride, r, 'block{}_{}'.format(i, j)))
                in_channels = out_channels

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, x, training=True):
        x = self.conv_stem(x, training=training)
        x = self.blocks(x, training=training)
        x = self.avg_pool(x)
        y = self.classifier(x)

        return y

if __name__ == '__main__':
    model = MobileNeXt()
    x = tf.random.normal((1, 224, 224, 3))
    y = model(x)
