import flax.linen as nn

densenet_kernel_init = nn.initializers.kaiming_normal()


class TransitionLayer(nn.Module):
    c_out: int  # Output feature size
    act_fn: callable  # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)
        x = nn.Conv(
            self.c_out,
            kernel_size=(1, 1),
            kernel_init=densenet_kernel_init,
            use_bias=False,
        )(x)
        x = nn.avg_pool(x, (2, 2), strides=(2, 2))
        return x
