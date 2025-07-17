from torchvision import transforms
from models import utils, functional as sf
import logging

# Cau hinh logging de hien thi thong tin trong qua trinh tien xu ly
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Lop bien doi anh S1 (giai doan dau tien trong mo hinh SNN)
class S1Transform:
    def __init__(self, filter, timesteps=15):
        # Bien doi anh PIL sang tensor PyTorch
        self.to_tensor = transforms.ToTensor()
        # Chuyen anh mau sang anh xam
        self.to_gray = transforms.Grayscale()
        # Bo loc DoG hoac cac filter khac se duoc truyen tu ngoai vao
        self.filter = filter
        # Bien doi tu cuong do anh sang spike theo thoi gian (15 buoc thoi gian)
        self.temporal_transform = utils.Intensity2Latency(
            timesteps, to_spike=True)
        # Bien dem so anh da xu ly
        self.cnt = 1

    def __call__(self, image):
        # Cac buoc tien xu ly se duoc thuc hien khi goi instance nay voi anh dau vao
        # In log sau moi 10000 anh xu ly
        if self.cnt % 10000 == 0:
            logging.info(f'Preprocessed {self.cnt} images')
        self.cnt += 1
        # Chuyen anh PIL sang tensor va scale tu [0,1] -> [0,255]
        image = self.to_tensor(image) * 255
        # Chuyen sang anh xam
        image = self.to_gray(image)
        # Them chieu batch size: tu [1, H, W] -> [1, 1, H, W]
        image.unsqueeze_(0)
        # Ap dung bo loc (vi du: DoG kernel)
        image = self.filter(image)
        # Chuan hoa cuc bo voi ban kinh = 8 (local normalization)
        image = sf.local_normalization(image, 8)
        # Bien doi cuong do anh sang dang spike theo thoi gian
        temporal_image = self.temporal_transform(image)
        # Ep kieu du lieu sang byte (uint8) de giam dung luong
        return temporal_image.byte()

