import torch
import torch.nn as nn

class Residual(nn.Module):
  def __init__(self,
               channels: int):
    super(Residual, self).__init__()

    #
    self.conv1 = nn.Conv2d(in_channels = channels, out_channels = channels,
                          kernel_size = 3, stride = 1, padding = 1)
    self.norm1 = nn.BatchNorm2d(channels) #nn.InstanceNorm2d(out_channels)
    self.act = nn.PReLU(num_parameters = channels)
    self.conv2 = nn.Conv2d(in_channels = channels, out_channels = channels,
                          kernel_size = 3, stride = 1, padding = 1)
    self.norm2 = nn.BatchNorm2d(channels) #nn.InstanceNorm2d(out_channels)

  def forward(self, input):

    output = self.conv1(input)
    output = self.norm1(output)
    output = self.act(output)
    output = self.conv2(output)
    output = self.norm2(output)
    output = output + input
    return output
  
class Upscale2(nn.Module):
  def __init__(self,
               in_channels: int, out_channels: int):
    super(Upscale2, self).__init__()

    #
    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                          kernel_size = 3, stride = 1, padding = 1)
    self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')
    self.act = nn.PReLU(num_parameters = out_channels)

  def forward(self, input):

    output = self.conv(input)
    output = self.upscale(output)
    output = self.act(output)
    return output

class Generator(nn.Module):
  def __init__(self, in_channels = 3, hid_channels = 64, up_channels = 256, res_count = 16):
    super(Generator, self).__init__()

    #
    self.conv_in = nn.Conv2d(in_channels = in_channels, out_channels = hid_channels,
                             kernel_size = 9, stride = 1, padding = 4)
    self.act = nn.PReLU(num_parameters = hid_channels)
    #
    self.residual = nn.Sequential()
    for layer in range(res_count):
      self.residual.append(Residual(hid_channels))
    #
    self.conv_hid = nn.Conv2d(in_channels = hid_channels, out_channels = hid_channels,
                              kernel_size = 3, stride = 1, padding = 1)
    self.norm_hid = nn.BatchNorm2d(hid_channels) #nn.InstanceNorm2d(out_channels)
    #
    self.up1 = Upscale2(in_channels = hid_channels, out_channels = up_channels)
    self.up2 = Upscale2(in_channels = up_channels, out_channels = up_channels)
    #
    self.conv_out = nn.Conv2d(in_channels = up_channels, out_channels = in_channels,
                              kernel_size = 9, stride = 1, padding = 4)

  def forward(self, input):
    input = self.act(self.conv_in(input))
    output = self.residual(input)
    output = self.norm_hid(self.conv_hid(output))
    output = output + input
    output = self.up1(output)
    output = self.up2(output)
    output = self.conv_out(output)

    return output


class DiscLayer(nn.Module):
  def __init__(self,
               in_channels: int, out_channels: int, stride: int):
    super(DiscLayer, self).__init__()

    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                              kernel_size = 3, stride = stride, padding = 1)
    self.norm = nn.BatchNorm2d(out_channels) #nn.InstanceNorm2d(out_channels)
    self.act = nn.LeakyReLU(0.2)

  def forward(self, input):
    input = self.conv(input)
    input = self.norm (input)
    input = self.act (input)
    return input


class Discriminator(nn.Module):
  def  __init__(self,
                in_channels = 3, hid_channels = 64):
    super(Discriminator, self).__init__()
    self.conv = self.conv = nn.Conv2d(in_channels = in_channels, out_channels = hid_channels,
                              kernel_size = 3, stride = 1, padding = 1)
    self.act =  nn.LeakyReLU(0.1)
    self.basic = nn.Sequential(DiscLayer(in_channels = hid_channels, out_channels = hid_channels, stride = 2),
                               DiscLayer(in_channels = hid_channels, out_channels = hid_channels * 2, stride = 1),
                               DiscLayer(in_channels = hid_channels * 2, out_channels = hid_channels * 2, stride = 2),
                               DiscLayer(in_channels = hid_channels * 2, out_channels = hid_channels * 4, stride = 1),
                               DiscLayer(in_channels = hid_channels * 4, out_channels = hid_channels * 4, stride = 2),
                               DiscLayer(in_channels = hid_channels * 4, out_channels = hid_channels * 8, stride = 1),
                               DiscLayer(in_channels = hid_channels * 8, out_channels = hid_channels * 8, stride = 2))

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features = 131072, out_features = 1024)

    self.act1 =  nn.LeakyReLU(0.2)
    self.fc2 = nn.Linear(in_features = 1024, out_features = 1)
    self.act2 = nn.Sigmoid()

  def forward(self, input):

    input = self.conv(input)
    input = self.act(input)
    input = self.basic(input)
    input = self.flatten(input)
    input = self.fc1(input)
    input = self.act1(input)
    input = self.fc2(input)
    input = self.act2(input)

    return input