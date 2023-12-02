from PIL import Image
import PIL

import os
import os.path

from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import RandomRotation

class TexNIST(VisionDataset):
    def __init__(self, root='./', transform=None, fraction=1.0):
        classes = list(range(0, 10))

        self.data = []
        self.targets = []
        self.transform = transform

        print('TexNIST loading {0} into memory...'.format(root))
        for c in classes:
            im_dir = os.path.join(root, str(c))

            max_samples = int(len([name for name in os.listdir(im_dir) if name.endswith(".PNG")]) * fraction)
            curr_samples = 0

            for filename in os.listdir(im_dir):
                if filename.endswith(".PNG") and curr_samples < max_samples:
                    img = Image.open(os.path.join(im_dir, filename))
                    self.data.append(img.copy())
                    img.close()
                    self.targets.append(c)
                    curr_samples += 1

        print('TexNIST loaded {0} samples from {0} into memory.'.format(len(self.data), root))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        
    def __len__(self):
        return len(self.data)

    def randomRotate(self):
        for i in range(len(self.data)):
            self.data[i] = RandomRotation((0, 360), resample=PIL.Image.BILINEAR)(self.data[i])
