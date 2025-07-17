import torch
import torch.nn.functional as fn
import numpy as np

# Ham kiem tra nguong va phat sinh tin hieu (spike)
def fire(potentials, threshold, return_thresholded_potentials=False):
    thresholded = potentials.clone().detach()
    fn.threshold_(thresholded, threshold, 0)  # Cat nguong duoi threshold, gia tri nho hon se tro thanh 0
    if return_thresholded_potentials:
        return thresholded.sign(), thresholded  # Tra ve tin hieu va gia tri nguong
    return thresholded.sign()  # Chi tra ve tin hieu (dau)

# Ham them padding xung quanh anh dau vao
def pad(input, pad, value=0):
    return fn.pad(input, pad, value=value)

# Ham pooling cuc dai voi kich thuoc kernel va buoc nhay
def pooling(input, kernel_size, stride=None, padding=0):
    return fn.max_pool2d(input, kernel_size, stride, padding)

# Ham chuan hoa cuc bo tren tung vung trong anh
def local_normalization(input, normalization_radius, eps=1e-12):
    length = normalization_radius * 2 + 1
    kernel = torch.ones(1, 1, length, length, device=input.device).float() / ((length) ** 2)  # Tao kernel trung binh
    y = input.squeeze(0)
    y.unsqueeze_(1)  # Chuyen ve (B, 1, H, W)
    means = fn.conv2d(y, kernel, padding=normalization_radius) + eps  # Tinh gia tri trung binh xung quanh
    y = y / means  # Chia tung diem cho trung binh
    y.squeeze_(1)
    y.unsqueeze_(0)
    return y

# Chi cho phep mot noron tai moi vi tri phat tin hieu (duyet qua cac map dac trung)
def pointwise_inhibition(thresholded_potentials):
    # Tim gia tri cuc dai tai moi vi tri qua cac buoc thoi gian
    maximum = torch.max(thresholded_potentials, dim=1, keepdim=True)
    # Lay dau cua gia tri cuc dai (phat hien tin hieu dau tien)
    clamp_pot = maximum[0].sign()
    # Tinh toan chi so cua tin hieu xuat hien som nhat
    clamp_pot_max_1 = (clamp_pot.size(0) - clamp_pot.sum(dim=0, keepdim=True)).long()
    clamp_pot_max_1.clamp_(0, clamp_pot.size(0) - 1)
    # Lay tin hieu o buoc thoi gian cuoi
    clamp_pot_max_0 = clamp_pot[-1:, :, :, :]
    # Tim vi tri thang cuoc (co gia tri cuc dai som nhat)
    winners = maximum[1].gather(0, clamp_pot_max_1)
    # Tao he so de ap dung su uc che
    coef = torch.zeros_like(thresholded_potentials[0]).unsqueeze_(0)
    coef.scatter_(1, winners, clamp_pot_max_0)
    # Ap dung su uc che bang nhan broadcasting
    return torch.mul(thresholded_potentials, coef)

# Tim cac vi tri co gia tri cao nhat (nguoi thang), dua tren thoi diem va gia tri tin hieu
def get_k_winners(potentials, kwta=1, inhibition_radius=0, spikes=None):
    # Tim thoi diem phat tin hieu som nhat tai moi vi tri
    maximum = (spikes.size(0) - spikes.sum(dim=0, keepdim=True)).long()
    maximum.clamp_(0, spikes.size(0)-1)
    values = potentials.gather(dim=0, index=maximum)  # Lay gia tri ung voi tin hieu som nhat
    # Nhan gia tri tin hieu cho tin hieu spike tai moi buoc thoi gian
    truncated_pot = spikes * values
    # Tang gia tri tai cac diem co spike de dam bao uu tien chon
    v = truncated_pot.max() * potentials.size(0)
    truncated_pot.addcmul_(spikes, v)
    # Tong hop gia tri theo thoi gian
    total = truncated_pot.sum(dim=0, keepdim=True)

    total.squeeze_(0)
    global_pooling_size = tuple(total.size())  # Kich thuoc cua feature map
    winners = []
    for k in range(kwta):
        max_val, max_idx = total.view(-1).max(0)  # Tim diem co gia tri lon nhat
        if max_val.item() != 0:
            # Chuyen chi so tu 1 chieu ve 3 chieu (feature, row, column)
            max_idx_unraveled = np.unravel_index(max_idx.item(), global_pooling_size)
            winners.append(max_idx_unraveled)  # Them vao danh sach nguoi thang
            # Khong cho feature map hien tai duoc chon tiep
            total[max_idx_unraveled[0], :, :] = 0
            # Uc che khu vuc xung quanh (de hoc dac trung da dang hon)
            if inhibition_radius != 0:
                rowMin, rowMax = max(0, max_idx_unraveled[-2]-inhibition_radius), min(
                    total.size(-2), max_idx_unraveled[-2]+inhibition_radius+1)
                colMin, colMax = max(0, max_idx_unraveled[-1]-inhibition_radius), min(
                    total.size(-1), max_idx_unraveled[-1]+inhibition_radius+1)
                total[:, rowMin:rowMax, colMin:colMax] = 0
        else:
            break  # Neu khong con gia tri nao khac 0 thi dung lai
    return winners

