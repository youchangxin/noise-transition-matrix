import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


from utils.syntheticData import Flip
from torch.utils.data import Dataset, DataLoader


class Dataset_MNIST(Dataset):
    def __init__(self, noise_rate=0.2, flip_type='symmetric', random_seed=20):
        img_npy = np.load('data/mnist/train_images.npy')
        label_npy = np.load('data/mnist/train_labels.npy')

        self.images = np.expand_dims(img_npy, axis=1)
        self.images = self.images.transpose((0, 2, 3, 1))
        flip = Flip(noise_rate, num_classes=10, random_state=random_seed)
        if flip_type == 'symmetric':
            noise_label, self.transition_matrix = flip.symmetric(label_npy)
        elif flip_type == 'asymmetric':
            noise_label, self.transition_matrix = flip.asymmetric(label_npy)
        elif flip_type == 'pair':
            noise_label, self.transition_matrix = flip.pair(label_npy)
        print(label_npy)
        print(noise_label)
        self.label = torch.from_numpy(noise_label).long()
        self.tranform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, idx):
        img, label = self.images[idx], self.label[idx]
        img = self.tranform(img)

        return img, label

    def __len__(self):
        return len(self.label)


class Dataset_MNIST_TEST(Dataset):
    def __init__(self):
        img_npy = np.load('data/mnist/train_images.npy')
        label_npy = np.load('data/mnist/train_labels.npy')

        self.images = np.expand_dims(img_npy, axis=1)
        self.images = self.images.transpose((0, 2, 3, 1))
        self.label = torch.from_numpy(label_npy).long()
        self.tranform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))])

    def __getitem__(self, idx):
        img, label = self.images[idx], self.label[idx]
        img = self.tranform(img)

        return img, label

    def __len__(self):
        return len(self.label)


class Dataset_CIFAR(Dataset):
    def __init__(self, num_classes, noise_rate=0.2, flip_type='symmetric', random_seed=20):
        self.num_classes = num_classes
        if self.num_classes == 10:
            img_npy = np.load('data/cifar10/train_images.npy')
            label_npy = np.load('data/cifar10/train_labels.npy')
        elif self.num_classes == 100:
            img_npy = np.load('data/cifar100/train_images.npy')
            label_npy = np.load('data/cifar100/train_labels.npy')

        self.images = img_npy.reshape((-1, 3, 32, 32))
        self.images = self.images.transpose((0, 2, 3, 1))

        flip = Flip(noise_rate, num_classes=self.num_classes, random_state=random_seed)
        if flip_type == 'symmetric':
            noise_label, self.transition_matrix = flip.symmetric(label_npy)
        elif flip_type == 'asymmetric':
            noise_label, self.transition_matrix = flip.asymmetric(label_npy)
        elif flip_type == 'pair':
            noise_label, self.transition_matrix = flip.pair(label_npy)

        self.label = torch.from_numpy(noise_label).long()
        self.tranform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])

    def __getitem__(self, idx):
        img, label = self.images[idx], self.label[idx]
        img = Image.fromarray(img)
        img = self.tranform(img)

        return img, label

    def __len__(self):
        return len(self.label)


class Dataset_CIFAR_TEST(Dataset):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        if self.num_classes == 10:
            img_npy = np.load('data/cifar10/train_images.npy')
            label_npy = np.load('data/cifar10/train_labels.npy')
        elif self.num_classes == 100:
            img_npy = np.load('data/cifar100/train_images.npy')
            label_npy = np.load('data/cifar100/train_labels.npy')

        self.images = img_npy.reshape((-1, 3, 32, 32))
        self.images = self.images.transpose((0, 2, 3, 1))

        self.label = torch.from_numpy(label_npy).long()
        self.tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                            (0.2023, 0.1994, 0.2010))])

    def __getitem__(self, idx):
        img, label = self.images[idx], self.label[idx]
        img = Image.fromarray(img)
        img = self.tranform(img)

        return img, label

    def __len__(self):
        return len(self.label)



if  __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_data = Dataset_MNIST_TEST()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=1,
                              shuffle=True,
                              drop_last=False)
    for batch_x, batch_y in train_loader:
        img = batch_x.numpy()[0]
        img = img.transpose((1, 2, 0))
        plt.imshow(img)
        print(batch_y)
        plt.show()
        print(batch_y)