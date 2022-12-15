import flax.linen as nn
from .denseblock import DenseBlock

densenet_kernel_init = nn.initializers.kaiming_normal()


class DenseBlock(nn.Module):
    num_layers: int  # Number of dense layers to apply in the block
    bn_size: int  # Bottleneck size to use in the dense layers
    growth_rate: int  # Growth rate to use in the dense layers
    act_fn: callable  # Activation function to use in the dense layers

    @nn.compact
    def __call__(self, x, train=True):
        for _ in range(self.num_layers):
            x = DenseLayer(
                bn_size=self.bn_size, growth_rate=self.growth_rate, act_fn=self.act_fn
            )(x, train=train)
        return x
