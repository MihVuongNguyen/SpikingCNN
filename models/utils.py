import torch
import torch.nn.functional as fn
import numpy as np
import math
import os

# Ham chuyen du lieu thanh cap (tuple 2 gia tri)
def to_pair(data):
    if isinstance(data, tuple):
        return data[0:2]
    return (data, data)


# Lop co so cho cac kernel loc
class FilterKernel:
    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self):
        pass


# Lop loc DoG (Difference of Gaussian)
class DoGKernel(FilterKernel):
    def __init__(self, window_size, sigma1, sigma2):
        super(DoGKernel, self).__init__(window_size)
        self.sigma1 = sigma1  # do lech chuan cua Gaussian thu nhat
        self.sigma2 = sigma2  # do lech chuan cua Gaussian thu hai

    def __call__(self):
        w = self.window_size // 2
        x, y = np.mgrid[-w: w+1: 1, -w: w+1: 1]  # tao luoi toa do
        a = 1.0 / (2 * math.pi)
        prod = x ** 2 + y ** 2
        f1 = (1 / (self.sigma1 ** 2)) * np.exp(-0.5 * (1 / (self.sigma1 ** 2)) * prod)
        f2 = (1 / (self.sigma2 ** 2)) * np.exp(-0.5 * (1 / (self.sigma2 ** 2)) * prod)
        dog = a * (f1 - f2)  # hieu cua hai Gaussian
        dog = (dog - np.mean(dog)) / np.max(dog)  # chuan hoa
        dog_tensor = torch.from_numpy(dog)
        return dog_tensor.float()


# Lop ap dung bo loc cho anh
class Filter:
    def __init__(self, filter_kernels, padding=0, threshold=None):
        self.max_window_size = filter_kernels[0].window_size
        self.kernels = torch.stack([kernel().unsqueeze(0) for kernel in filter_kernels])
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        self.threshold = threshold

    def __call__(self, input):
        # Ap dung bo loc bang ham convolution
        output = fn.conv2d(input, self.kernels, padding=self.padding).float()
        # Ap dung nguong: neu gia tri nho hon nguong thi gan bang 0
        output = torch.where(output < self.threshold, torch.tensor(0.0, device=output.device), output)
        return output


# Chuyen cuong do anh sang thoi gian phat tin hieu
class Intensity2Latency:
    def __init__(self, timesteps, to_spike=False):
        self.timesteps = timesteps  # so buoc thoi gian
        self.to_spike = to_spike  # co chuyen sang dang spike hay khong

    def transform(self, intensities):
        bins_intensities = []
        nonzero_cnt = torch.nonzero(intensities).size()[0]  # dem so diem khac 0
        bin_size = nonzero_cnt // self.timesteps  # kich thuoc moi buoc thoi gian
        intensities_flattened = torch.reshape(intensities, (-1,))
        intensities_flattened_sorted = torch.sort(intensities_flattened, descending=True)
        sorted_bins_value, sorted_bins_idx = torch.split(intensities_flattened_sorted[0], bin_size), torch.split(intensities_flattened_sorted[1], bin_size)
        spike_map = torch.zeros_like(intensities_flattened_sorted[0])

        for i in range(self.timesteps):
            spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])  # gan gia tri cuong do
            spike_map_copy = spike_map.clone().detach()
            spike_map_copy = spike_map_copy.reshape(tuple(intensities.shape))
            bins_intensities.append(spike_map_copy.squeeze(0).float())
        return torch.stack(bins_intensities)

    def __call__(self, image):
        if self.to_spike:
            return self.transform(image).sign()  # chuyen sang dang spike (dau)
        return self.transform(image)


# Dataset co cache: luu ket qua xu ly vao bo nho
class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, cache_address=None):
        self.dataset = dataset
        self.cache_address = cache_address
        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        # Neu chua co trong cache
        if self.cache[index] is None:
            sample, target = self.dataset[index]
            if self.cache_address is None:
                self.cache[index] = sample, target
            else:
                save_path = os.path.join(self.cache_address, str(index))
                torch.save(sample, save_path + ".cd")  # luu input
                torch.save(target, save_path + ".cl")  # luu label
                self.cache[index] = save_path
        else:
            # Neu da cache roi
            if self.cache_address is None:
                sample, target = self.cache[index]
            else:
                sample = torch.load(self.cache[index] + ".cd")
                target = torch.load(self.cache[index] + ".cl")
        return sample, target

    # Xoa cache hien tai
    def reset_cache(self):
        if self.cache_address is not None:
            for add in self.cache:
                os.remove(add + ".cd")
                os.remove(add + ".cl")
        self.cache = [None] * len(self)

    def __len__(self):
        return len(self.dataset)

