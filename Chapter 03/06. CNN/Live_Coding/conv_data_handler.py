import torch as T
from torchvision import datasets, transforms

img_path = '../Datasets/cat_vs_dog/'

train_transform = transforms.Compose([transforms.Resize((120, 120)),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(100),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transform = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(img_path + 'train', transform=train_transform)
test_data = datasets.ImageFolder(img_path + 'test', transform=test_transform)
