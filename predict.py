import torch
import numpy as np
import os
import logging

# Cai dat logging de hien thi thong tin debug
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def pass_through_network(model, loader, device='cpu'):
    """
    Ham nay truyen toan bo du lieu tu data loader qua model de trich xuat dac trung (features).
    Dau vao:
        - model: mang hoc (da duoc train)
        - loader: DataLoader chua du lieu (batch_size x ...)
        - device: CPU hoac GPU
    Dau ra:
        - features: mang numpy chua cac dac trung cua tung mau (N, D)
        - targets: mang numpy chua cac nhan tuong ung (N,)
    """

    X_path = 'tmp/test_x.npy'
    y_path = 'tmp/test_y.npy'

    # Neu da co file feature va label luu tru, tai truc tiep
    if os.path.isfile(X_path):
        features = np.load(X_path)
        targets = np.load(y_path)
    else:
        os.makedirs('tmp', exist_ok=True)
        logger.info('into else ')
        features = []
        targets = []
        for i, (data, target) in enumerate(loader):
            logger.info('into loop ')
            # data: (batch_size, T, C, H, W) neu dung spike tensor
            out = pass_batch_through_network(model, data, device)
            # out: (batch_size, D) duoi dang numpy array
            np.save(f"tmp/fl32_features_batch_{i}.npy", out.astype(np.float32))
            logger.info('into loop2 ')
            np.save(f"tmp/fl32_targets_batch_{i}.npy", target.numpy().astype(np.float32))
            logger.info('into loop3 ')
        # Doc lai cac file da luu de ghep lai toan bo features
        features = np.concatenate([np.load(f"tmp/fl32_features_batch_{i}.npy") for i in range(100)])
        logger.info('into else2 ')
        targets = np.concatenate([np.load(f"tmp/fl32_targets_batch_{i}.npy") for i in range(100)])
        logger.info('into else3 ')
        np.save(X_path, features)
        logger.info('into else4 ')
        np.save(y_path, targets)
        logger.info('into else5 ')
    return features, targets


def pass_batch_through_network(model, batch, device='cpu'):
    """
    Truyen mot batch spike qua model va luu anh feature map.
    Dau vao:
        - model: mang hoc
        - batch: (batch_size, T, C, H, W)
        - device: CPU/GPU
    Dau ra:
        - ans: mang numpy (batch_size, D) voi D la tong so dac trung cua model
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    save_dir = "logs/feature_maps"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        ans = []
        logger.info('into batch 1 ')
        for idx, data in enumerate(batch):
            logger.info('into batch loop ')
            # data: (T, C, H, W) → mot mau (1 sample)
            data_in = data.to(device)
            # output: (T, C, H, W) → dau ra model theo tung thoi diem
            output = model(data_in)
            print(f"Output shape from model: {output.shape}")  # VD: (15, 150, 9, 9)
            ans.append(output.reshape(-1).cpu().numpy())  # reshape thanh (T * C * H * W,) → 1 chieu

            # Tong hop theo truc thoi gian (max), giu lai (C, H, W)
            fmap = torch.max(output, dim=0)[0]  # fmap: (C, H, W)
            print(f"Aggregated fmap shape: {fmap.shape}")  # VD: (150, 9, 9)
            # Chon 32 kenh dau tien de ve hinh
            fmap = fmap[:32]
            # Chuan hoa ve [0, 1] de ve anh
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            # Resize ve (28, 28) cho de quan sat
            fmap = F.interpolate(fmap.unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=False).squeeze(0)
            for j in range(fmap.shape[0]):
                channel_map = fmap[j].cpu().numpy()  # (28, 28)
                print(f"channel_map[{j}] shape: {channel_map.shape}")
                plt.imshow(channel_map, cmap='viridis')
                plt.axis('off')
                plt.title(f"Sample {idx} - Feature {j}")
                plt.savefig(f"{save_dir}/sample{idx}_fmap{j}.png", bbox_inches='tight')
                plt.close()

        # ans: list cac vector (T*C*H*W,) → chuyen thanh array co shape (batch_size, D)
        return np.array(ans)


def eval(X, y, predictions):
    """
    Danh gia do chinh xac tren tap test.
    Dau vao:
        - X: (N, D) dac trung cua tung mau (spike, co the co nhieu dong toan 0)
        - y: (N,) nhan dung
        - predictions: (N,) nhan du doan
    Dau ra: bo 3
        - accuracy: so mau dung tren tong so mau
        - error rate: ti le sai
        - silence rate: ti le mau im lang (khong co spike)
    """
    # Xet xem dong nao trong X khac 0 (tuc la co spike)
    non_silence_mask = np.count_nonzero(X, axis=1) > 0
    # Kiem tra du doan dung
    correct_mask = predictions == y
    # Chi lay cac mau vua co spike vua dung
    correct_non_silence = np.logical_and(correct_mask, non_silence_mask)
    correct = np.count_nonzero(correct_non_silence)
    silence = np.count_nonzero(~non_silence_mask)
    # acc, loi, ti le im lang
    return (
        correct / len(X),
        (len(X) - (correct + silence)) / len(X),
        silence / len(X)
    )

