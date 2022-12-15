import flax.linen as nn

densenet_kernel_init = nn.initializers.kaiming_normal()


class DenseLayer(nn.Module):
    bn_size: int  # Bottleneck size (factor of growth rate) for the output of the 1x1 convolution
    growth_rate: int  # Number of output channels of the 3x3 convolution
    act_fn: callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.bn_size * self.growth_rate,
            kernel_size=(1, 1),
            kernel_init=densenet_kernel_init,
            use_bias=False,
        )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.growth_rate,
            kernel_size=(3, 3),
            kernel_init=densenet_kernel_init,
            use_bias=False,
        )(z)
        x_out = jnp.concatenate([x, z], axis=-1)
        return x_out
