import torch
import torchvision
from data.preprocess import S1Transform  # Bien doi S1: DoG + spike encoding
from models import utils  # Cac ham ho tro nhu tao kernel DoG, Filter
from torchvision import datasets
import torch.utils.data as data
from torch.utils.data import DataLoader
from predict import pass_through_network, eval  # Ham forward toan bo dataset va danh gia
import logging
from models.model import Network  # Mo hinh mang than kinh SNN

# Cau hinh ghi log
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ham test: tai model, phan loai va danh gia tren tap test
def test(dataset, device, model_directory, weights_name, classifier_name, data_directory, args):
    pt_path = model_directory + classifier_name  # Duong dan file phan loai da huan luyen (SGD/SVM)
    net_path = model_directory + weights_name    # Duong dan file model (.pt hoac .net)

    # Tao kernel DoG 7x7
    kernels = [utils.DoGKernel(7, 1, 2),     #(kernel_size, sigma1, sigma2)
               utils.DoGKernel(7, 2, 1)]
    
    # Tao Filter su dung DoG kernels
    filter = utils.Filter(kernels, padding=3, threshold=50)

    # Bien doi anh thanh dang spike (bang DoG + threshold)
    s1_transform = S1Transform(filter)

    # Tai tap test da duoc bien doi
    loader = get_loader(dataset, data_directory, s1_transform)

    # Khoi tao model va tai trong so
    model = Network()  # MANG THAN KINH (2 conv layers)
    model.load_state_dict(torch.load(net_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()  # Set ve mode danh gia

    # Tai bo phan loai da huan luyen (SVM/SGDClassifier)
    clf = torch.load(pt_path, map_location=device)

    # Chay tap test qua mang de lay feature
    test_X, test_y = pass_through_network(model=model, loader=loader, device=device)
    # test_X: [N, D], D = kich thuoc vector dac trung (flatten conv2)

    # Du doan ket qua
    predictions = clf.predict(test_X)

    # Tinh toan accuracy, error, silence
    accuracy, error, silence = eval(test_X, test_y, predictions)
    logger.info(
        f'\n-------- Accuracy : {accuracy} --------\n-------- Error : {error} --------\n-------- Silence : {silence} --------')

# Ham get_loader dung de tai du lieu test (MNIST hoac ImageFolder)
def get_loader(dataset, data_directory, s1_transform):
    if dataset == 'MNIST':
        # Su dung bo du lieu test cua MNIST (60000 train, 10000 test)
        test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_directory,
                                                             train=False, download=True,
                                                             transform=s1_transform))
    else:
        # Neu dataset tuy chinh, su dung ImageFolder
        test = datasets.ImageFolder(root=dataset, transform=s1_transform)
    
    # Tra ve DataLoader
    return DataLoader(test, batch_size=128, shuffle=False)

