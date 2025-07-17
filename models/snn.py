import torch
import torch.nn as nn
import torch.nn.functional as fn
from . import functional as sf
from torch.nn.parameter import Parameter
from .utils import to_pair

device = torch.device('cpu')


# Lop convolution tuy bien cho mang SNN
class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.05):
        super(Convolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Chuyen doi kernel_size thanh tuple neu can
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = to_pair(kernel_size)

        # Khoi tao trong so kernel
        self.weight = Parameter(torch.Tensor(
            self.out_channels, self.in_channels, *self.kernel_size))
        self.weight.requires_grad_(False)  # Khong su dung backpropagation
        self.reset_weight(weight_mean, weight_std)

    # Ham khoi tao trong so theo phan phoi Gaussian
    def reset_weight(self, weight_mean=0.8, weight_std=0.02):
        self.weight.normal_(weight_mean, weight_std)

    # Ham lan truyen tien cua convolution
    def forward(self, input):
        return fn.conv2d(input, self.weight)


# Lop hoc STDP (Spike-Timing Dependent Plasticity)
class STDP(nn.Module):
    def __init__(self, conv_layer, learning_rate=(0.004, -0.003), use_stabilizer=True, lower_bound=0, upper_bound=1):
        super(STDP, self).__init__()
        self.conv_layer = conv_layer
        # toc do hoc ap dung cho cap truoc-sau va sau-truoc
        self.learning_rate = (torch.tensor([learning_rate[0]]),
                              torch.tensor([learning_rate[1]]))
        self.use_stabilizer = use_stabilizer  # Kiem soat on dinh trong so
        self.lower_bound = lower_bound  # Gioi han duoi cua trong so
        self.upper_bound = upper_bound  # Gioi han tren cua trong so

    # Ham lay thu tu xuat hien cua tin hieu input va output (truoc hay sau)
    def get_pre_post_ordering(self, input_spikes, output_spikes, winners):
        # Tinh tong tin hieu de lay thoi gian xuat hien (latency)
        input_latencies = torch.sum(input_spikes, dim=0)
        output_latencies = torch.sum(output_spikes, dim=0)
        result = []

        for winner in winners:
            # Tao tensor voi gia tri thoi gian phat tin hieu cua noron winner
            out_tensor = torch.ones(
                *self.conv_layer.kernel_size, device=output_latencies.device) * output_latencies[winner]

            # Lay vung cam ung cua noron winner tu input
            in_tensor = input_latencies[:, winner[-2]: winner[-2] + self.conv_layer.kernel_size[-2],
                                        winner[-1]: winner[-1] + self.conv_layer.kernel_size[-1]]
            # So sanh: True neu input xuat hien som hon hoac bang output
            result.append(torch.ge(in_tensor, out_tensor))

        return result

    # Ham lan truyen tien cua STDP
    def forward(self, input_spikes, output_spikes, winners):
        # Xac dinh cac cap pre-post tuong ung
        pairings = self.get_pre_post_ordering(
            input_spikes, output_spikes, winners)

        lr = torch.zeros_like(self.conv_layer.weight)  # Tao tensor toc do hoc

        for i in range(len(winners)):
            winner = winners[i][0]  # Lay chi so cua map dac trung thang cuoc
            pair = pairings[i].clone().detach().to(device)  # Lay cap pre-post
            lr0 = self.learning_rate[0].clone().detach().to(device)
            lr1 = self.learning_rate[1].clone().detach().to(device)

            # Cap nhat toc do hoc cho trong so tuong ung
            lr[winner.item()] = torch.where(pair, lr0, lr1)

        # Cap nhat trong so theo quy tac STDP
        self.conv_layer.weight += lr * ((self.conv_layer.weight - self.lower_bound) * (
            self.upper_bound - self.conv_layer.weight) if self.use_stabilizer else 1)

        # Giu trong so trong khoang cho phep
        self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)

    # Ham thay doi toc do hoc
    def update_learning_rate(self, ap, an):
        self.learning_rate = tuple([ap, an])

