from flax import linen as nn
from .resnetblock import ResNetBlock


resnet_kernel_init = nn.initializers.variance_scaling(
    2.0, mode="fan_out", distribution="normal"
)


class PreActResNetBlock(ResNetBlock):
    @nn.compact
    def __call__(self, x, train=True):
        # Network representing F
        z = nn.BatchNorm()(x, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            strides=(1, 1) if not self.subsample else (2, 2),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)
        z = nn.BatchNorm()(z, use_running_average=not train)
        z = self.act_fn(z)
        z = nn.Conv(
            self.c_out,
            kernel_size=(3, 3),
            kernel_init=resnet_kernel_init,
            use_bias=False,
        )(z)

        if self.subsample:
            x = nn.BatchNorm()(x, use_running_average=not train)
            x = self.act_fn(x)
            x = nn.Conv(
                self.c_out,
                kernel_size=(1, 1),
                strides=(2, 2),
                kernel_init=resnet_kernel_init,
                use_bias=False,
            )(x)

        x_out = z + x
        return x_out
