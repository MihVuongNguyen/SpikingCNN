import torch
import torch.nn as nn
from models import snn, functional as sf


class Network(nn.Module):

    def __init__(self, device='cpu'):
        super(Network, self).__init__()
        self.device = device

        # Lop convolution thu nhat
        self.conv1 = snn.Convolution(
            in_channels=2, out_channels=32, kernel_size=5)
        self.conv1_threshold = 10  # nguong phat tin hieu cho conv1
        self.conv1_kwinners = 5  # so noron thang cuoc duoc chon tu conv1
        self.conv1_inhibition_rad = 2  # ban kinh uc che cho conv1

        # Lop convolution thu hai
        self.conv2 = snn.Convolution(
            in_channels=32, out_channels=150, kernel_size=2)
        self.conv2_threshold = 1  # nguong phat tin hieu cho conv2
        self.conv2_kwinners = 8  # so noron thang cuoc duoc chon tu conv2
        self.conv2_inhibition_rad = 1  # ban kinh uc che cho conv2

        # Khoi tao STDP (hoc dong bo thoi gian)
        self.stdp1 = snn.STDP(conv_layer=self.conv1)
        self.stdp2 = snn.STDP(conv_layer=self.conv2)

        # Gia tri hoc toi da cho STDP
        self.max_ap = torch.Tensor([0.15]).to(self.device)

        # Bo nho de luu tru cac thong tin trong lan truy xuat
        self.ctx = {"input_spikes": None, "potentials": None,
                    "output_spikes": None, "winners": None}
        self.spk_cnt = 0  # dem so lan training da chay

    # Ham luu du lieu cua mot lan lan truyen
    def save_data(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners

    # Ham lan truyen tien
    def forward(self, input, layer_idx=None):
        # Them padding cho anh dau vao
        input = sf.pad(input.float(), (2, 2, 2, 2))

        if self.training:
            # Trong che do training, xu ly tung khung thoi gian
            B, T, C, H, W = input.shape  # Lay kich thuoc batch
            spk_list = []  # Danh sach luu tin hieu
            pot_list = []  # Danh sach luu gia tri dien the

            for t in range(T):
                frame = input[:, t, :, :, :]  # Lay anh o buoc thoi gian t
                pot = self.conv1(frame)  # Tinh dien the dau ra
                spk, pot_t = sf.fire(pot, threshold=self.conv1_threshold, return_thresholded_potentials=True)
                spk_list.append(spk)
                pot_list.append(pot_t)

            # Gop ket qua lai thanh tensor [B, T, C, H, W]
            spk = torch.stack(spk_list, dim=1)
            pot = torch.stack(pot_list, dim=1)

            # Xu ly neu chi huong den conv1 (layer_idx == 1)
            if layer_idx == 1:
                self.spk_cnt += 1
                # Cap nhat STDP moi 500 lan
                if self.spk_cnt >= 500:
                    self.spk_cnt = 0
                    ap = self.stdp1.learning_rate[0].clone().detach().to(self.device) * 2
                    ap = torch.min(ap, self.max_ap)  # Gioi han ap
                    an = ap * -0.75  # He so hoc am
                    self.stdp1.update_learning_rate(ap, an)  # Cap nhat toc do hoc

                pot = sf.pointwise_inhibition(pot)  # Ap dung uc che diem
                spk = pot.sign()  # Tinh tin hieu phat tu pot
                winners = sf.get_k_winners(
                    pot, self.conv1_kwinners, self.conv1_inhibition_rad, spk)
                self.save_data(input, pot, spk, winners)  # Luu ket qua

            else:
                # Neu khong phai conv1, xu ly qua conv2
                spk_pooling = sf.pooling(spk, 2, 2, 1)  # Pooling tin hieu
                spk_in = sf.pad(spk_pooling, (1, 1, 1, 1))  # Them padding
                spk_in = sf.pointwise_inhibition(spk_in)  # Ap dung uc che

                potentials = self.conv2(spk_in)  # Tinh dien the
                spk, pot = sf.fire(potentials, self.conv2_threshold, True)  # Phat tin hieu
                pot = sf.pointwise_inhibition(pot)  # Uc che diem
                spk = pot.sign()
                winners = sf.get_k_winners(
                    pot, self.conv2_kwinners, self.conv2_inhibition_rad, spk)
                self.save_data(spk_in, pot, spk, winners)  # Luu ket qua

        else:
            # Trong che do danh gia (eval)
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_threshold, True)
            pooling = sf.pooling(spk, 2, 2, 1)
            padded = sf.pad(pooling, (1, 1, 1, 1))

            pot = self.conv2(padded)
            spk, pot = sf.fire(pot, self.conv2_threshold, True)

            # Tinh output cuoi cung bang pooling
            spk_out = sf.pooling(spk, 2, 2, 1)
            return spk_out

    # Ham thuc hien STDP (hoc)
    def stdp(self, layer_idx):
        if layer_idx == 1:
            stdpn = self.stdp1
        else:
            stdpn = self.stdp2
        stdpn(self.ctx["input_spikes"],
              self.ctx["output_spikes"], self.ctx["winners"])

