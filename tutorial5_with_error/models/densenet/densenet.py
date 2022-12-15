import flax.linen as nn
from .denseblock import DenseBlock
from .transitionlayer import TransitionLayer


densenet_kernel_init = nn.initializers.kaiming_normal()


class DenseNet(nn.Module):
    num_classes: int
    act_fn: callable = nn.relu
    num_layers: tuple = (6, 6, 6, 6)
    bn_size: int = 2
    growth_rate: int = 16

    @nn.compact
    def __call__(self, x, train=True):
        c_hidden = (
            self.growth_rate * self.bn_size
        )  # The start number of hidden channels

        x = nn.Conv(c_hidden, kernel_size=(3, 3), kernel_init=densenet_kernel_init)(x)

        for block_idx, num_layers in enumerate(self.num_layers):
            x = DenseBlock(
                num_layers=num_layers,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                act_fn=self.act_fn,
            )(x, train=train)
            c_hidden += num_layers * self.growth_rate
            if (
                block_idx < len(self.num_layers) - 1
            ):  # Don't apply transition layer on last block
                x = TransitionLayer(c_out=c_hidden // 2, act_fn=self.act_fn)(
                    x, train=train
                )
                c_hidden //= 2

        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x
