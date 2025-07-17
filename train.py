import os
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
from data.preprocess import S1Transform  # Bien doi S1: DoG + spike
import matplotlib.pyplot as plt
from models import utils
from sklearn.svm import LinearSVC 
from sklearn.linear_model import SGDClassifier
import torchvision.datasets as datasets
from models.model import Network
import logging
from predict import pass_batch_through_network, eval
from sklearn.metrics import accuracy_score
import random
from PIL import Image

# Cau hinh logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ham chinh de huan luyen toan bo pipeline
def train(dataset, device, model_directory, classifier_name, data_directory, args):
    # Tao kernel DoG (Difference of Gaussian) kich thuoc 7x7
    kernels = [utils.DoGKernel(7, 1, 2), utils.DoGKernel(7, 2, 1)]
    filter = utils.Filter(kernels, padding=3, threshold=50)  # Bo loc S1

    logging.info(f'step 1')
    s1_transform = S1Transform(filter)  # Bien doi anh ve dang spike
    logging.info(f'step 2')

    model = Network(device).to(device)  # Khoi tao mang than kinh
    logging.info(f'step 3')

    loader = get_loader(dataset, data_directory, s1_transform)  # Tai du lieu
    logging.info(f'step 4')

    # Huan luyen 2 tang (layer) khong co giam sat
    train_layer(1, model=model, loader=loader, model_directory=model_directory, device=device)
    train_layer(2, model=model, loader=loader, model_directory=model_directory, device=device)

    # Huan luyen phan lop bang SVM
    train_eval_classifier(model, loader, device, model_directory, classifier_name, C=2.4)

# Ham doc du lieu va bien doi bang S1
def get_loader(dataset, data_directory, s1_transform):
    if dataset == 'MNIST':
        logging.info(f'into get_loader')
        train = utils.CacheDataset(torchvision.datasets.MNIST(
            root=data_directory,
            train=True,
            download=True,
            transform=s1_transform))
    else:
        train = datasets.ImageFolder(root=dataset, transform=s1_transform)

    return DataLoader(train, batch_size=10, shuffle=False)  # Tra ve data loader voi batch=10

# Huan luyen 1 layer don bang STDP
def train_layer(num_layer, model, loader, model_directory, device='cpu'):
    model.train()

    name = 'first' if num_layer == 1 else 'second'
    net_path = model_directory + "saved_l" + str(num_layer) + ".net"

    logger.info("\nTraining the {} layer ...".format(name))
    if os.path.isfile(net_path):
        # Tai model da luu neu co
        model.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
    else:
        layer_name = 'conv' + str(num_layer) + '.weight'
        learning_convergence = 1  # Khoi tao do hoi tu
        epoch = 1
        while learning_convergence > 0.01:  # Lap cho den khi hoi tu
            logger.info(f"======================== Epoch {epoch} ========================")
            logger.info(f"======================== Layer {num_layer} ========================")
            for data, _ in loader:
                train_unsupervised(model, data, num_layer, device)
            epoch += 1
            weights = model.state_dict()[layer_name]
            learning_convergence = calculate_learning_convergence(weights)

        torch.save(model.state_dict(), net_path)  # Luu trong so sau khi hoc

# Huan luyen khong co giam sat 1 mau du lieu bang STDP
def train_unsupervised(model, data, layer_idx, device):
    for i in range(len(data)):
        data_in = data[i].to(device)  # [T, H, W] → Tensor spike
        model(data_in, layer_idx)     # Lan truyen
        model.stdp(layer_idx)         # Cap nhat trong so bang STDP

# Ham tinh do hoi tu cua trong so
def calculate_learning_convergence(weights):
    n_w = weights.numel()  # So phan tu trong trong so
    sum_wf_i = torch.sum(weights * (1 - weights))  # Tinh tong theo cong thuc STDP
    c_l = sum_wf_i / n_w  # Do hoi tu
    return c_l.item()

# Huan luyen va danh gia phan loai SVM
def train_eval_classifier(model, loader, device, model_directory, classifier_name, C=2.4, max_iter=1000):
    logger.info('Training the classifier...')
    pt_path = model_directory + classifier_name

    model.eval()  # De chay forward mode
    logger.info('into eval classifier...1')

    features = []  # Danh sach feature dau ra cua model
    targets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]  # Nhung target label (gia su 10 mau)

    os.makedirs('tmp', exist_ok=True)
    for i, (data, target) in enumerate(loader):
        logger.info('into loop ')
        out = pass_batch_through_network(model, data, device)  # Lay feature [N, D]
        features.append(out)

    features = np.concatenate(features)  # [num_samples, feature_dim]
    targets = np.arange(10)  # Tao target [0,1,2,...,9]

    # Huan luyen phan lop bang Linear SVM
    clf = LinearSVC(C=C, max_iter=max_iter)
    clf.fit(features, targets)

    predictions = clf.predict(features)  # Du doan
    score = clf.decision_function(features)  # Lay diem tin tuong

    # Luu ket qua
    vis_dir = "logs/classifier_predictions"
    os.makedirs(vis_dir, exist_ok=True)

    for i in range(len(predictions)):
        pred = predictions[i]
        label = targets[i]
        print(f"[Sample {i}] Ground truth: {label}, Prediction: {pred}")
        with open(f"{vis_dir}/sample_{i}_result.txt", "w") as f:
            f.write(f"Ground truth: {label}\nPrediction: {pred}\n")

    # Danh gia ket qua (accuracy, error, silence)
    accuracy, error, silence = eval(features, targets, predictions)
    logger.info(
        f'\n-------- Accuracy : {accuracy} score: {score} --------\n-------- Error : {error} --------\n-------- Silence : {silence} --------')

