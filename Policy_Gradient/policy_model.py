from torch import nn
from torch.functional import F

FIRST_LAYER_KERNEL_SIZE = 3
FIRST_LAYER_STRIDE = 2
FIRST_LAYER_OUT = 32

SECOND_LAYER_KERNEL_SIZE = 3
SECOND_LAYER_STRIDE = 1
SECOND_LAYER_OUT = 32

THIRD_LAYER_KERNEL_SIZE = 3
THIRD_LAYER_STRIDE = 1
THIRD_LAYER_OUT = 32


class Policy(nn.Module):
    def __init__(self, num_actions, frame_dim):
        super(Policy, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=frame_dim[2], out_channels=FIRST_LAYER_OUT,
                               kernel_size=FIRST_LAYER_KERNEL_SIZE, stride=FIRST_LAYER_STRIDE)
        self.conv1_bn = nn.BatchNorm2d(FIRST_LAYER_OUT)

        self.conv2 = nn.Conv2d(in_channels=FIRST_LAYER_OUT, out_channels=SECOND_LAYER_OUT,
                               kernel_size=SECOND_LAYER_KERNEL_SIZE, stride=SECOND_LAYER_STRIDE)
        self.conv2_bn = nn.BatchNorm2d(SECOND_LAYER_OUT)

        self.conv3 = nn.Conv2d(in_channels=SECOND_LAYER_OUT, out_channels=THIRD_LAYER_OUT,
                               kernel_size=THIRD_LAYER_KERNEL_SIZE, stride=THIRD_LAYER_STRIDE)
        self.conv3_bn = nn.BatchNorm2d(THIRD_LAYER_OUT)

    def conv2d_size_out(size, kernel_size, stride):
        return (size - (kernel_size - 1) - 1) // stride + 1


