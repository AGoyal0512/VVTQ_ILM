import json
import os
import ssl
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torchvision import datasets, transforms

ssl._create_default_https_context = ssl._create_unverified_context

from .const import GTSRB_LABEL_MAP, IMAGENETNORMALIZE

from torch.utils.data import Dataset, DataLoader

# custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels= None, transforms = None):
        self.labels = labels
        self.images = images
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        data = self.images[index][:]
        
        if self.transforms:
            data = self.transforms(data)
            
        return (data, self.labels[index])
    
def sample_n_shots(args, train_data):
    # Create an empty list to store the sampled indices
    sampled_indices = []

    # Iterate over the unique classes in the dataset
    if args.dataset in ["svhn"]:
        unique_classes = np.unique(np.asarray(train_data.labels))
    elif args.dataset in ["gtsrb"]:
        gtsrb_labels = [
            train_data._samples[i][1] for i in range(len(train_data._samples))
        ]
        unique_classes = np.unique(np.asarray(gtsrb_labels))
    elif args.dataset in ["dtd", "flowers102", "oxfordpets"]:
        unique_classes = np.unique(np.asarray(train_data._labels))
    elif args.dataset in ["eurosat"]:
        eurosat_labels = [
            train_data.samples[i][1] for i in range(len(train_data.samples))
        ]
        unique_classes = np.unique(np.asarray(eurosat_labels))
    else:
        unique_classes = np.unique(np.asarray(train_data.targets))
    for class_label in unique_classes:
        # Find the indices of samples belonging to the current class
        if args.dataset in ["svhn"]:
            class_indices = np.where(train_data.labels == class_label)[0]
        elif args.dataset in ["gtsrb"]:
            class_indices = np.where(gtsrb_labels == class_label)[0]
        elif args.dataset in ["dtd", "flowers102", "oxfordpets"]:
            class_indices = np.where(train_data._labels == class_label)[0]
        elif args.dataset in ["eurosat"]:
            class_indices = np.where(eurosat_labels == class_label)[0]
        else:
            class_indices = np.where(train_data.targets == class_label)[0]

        # shuffle the indices
        np.random.shuffle(class_indices)

        # Sample n_shots samples from the current class
        n_samples = int(args.n_shot)
        sampled_indices.extend(class_indices[:n_samples])

    # shuffle again
    np.random.shuffle(sampled_indices)

    # Create a Subset from the sampled indices
    subset = Subset(train_data, sampled_indices)

    return subset


def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace("_", " ").replace("-", " ")
    return class_names


def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())


def prepare_expansive_data(args, dataset, data_path):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=preprocess
        )
        test_data = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=preprocess
        )

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": refine_classnames(test_data.classes),
            "mask": np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.CIFAR100(
            root=data_path, train=True, download=True, transform=preprocess
        )
        test_data = datasets.CIFAR100(
            root=data_path, train=False, download=True, transform=preprocess
        )

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": refine_classnames(test_data.classes),
            "mask": np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.GTSRB(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.GTSRB(
            root=data_path, split="test", download=True, transform=preprocess
        )

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": refine_classnames(list(GTSRB_LABEL_MAP.values())),
            "mask": np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.SVHN(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.SVHN(
            root=data_path, split="test", download=True, transform=preprocess
        )

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": [f"{i}" for i in range(10)],
            "mask": np.zeros((32, 32)),
        }
    elif dataset == "dtd":
        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.DTD(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.DTD(
            root=data_path, split="test", download=True, transform=preprocess
        )

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": refine_classnames(test_data.classes),
            "mask": np.zeros((128, 128)),
        }
    elif dataset == "flowers102":
        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.Flowers102(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.Flowers102(
            root=data_path, split="test", download=True, transform=preprocess
        )
        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": [f"{i}" for i in range(102)],
            "mask": np.zeros((128, 128)),
        }
    elif dataset == "eurosat":
        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        D = datasets.EuroSAT(root=data_path, download=True, transform=preprocess)
        X_train, X_test, y_train, y_test = train_test_split(
            D.samples, D.targets, test_size=0.1, stratify=D.targets, random_state=1
        )

        train_data = datasets.EuroSAT(root=data_path, transform=preprocess)
        train_data.data = X_train
        train_data.targets = y_train

        test_data = datasets.EuroSAT(root=data_path, transform=preprocess)
        test_data.data = X_test
        test_data.targets = y_test

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=8
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=8),
        }
        configs = {
            "class_names": refine_classnames(test_data.classes),
            "mask": np.zeros((128, 128)),
        }
    elif dataset == "abide":
        preprocess = transforms.ToTensor()
        D = ABIDE(root=data_path)
        X_train, X_test, y_train, y_test = train_test_split(
            D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1
        )
        train_data = ABIDE(root=data_path, transform=preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root=data_path, transform=preprocess)
        test_data.data = X_test
        test_data.targets = y_test

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 64, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": ["non ASD", "ASD"],
            "mask": D.get_mask(),
        }
    elif dataset == "oxfordpets":
        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.OxfordIIITPet(
            root=data_path, split="trainval", download=True, transform=preprocess
        )
        test_data = datasets.OxfordIIITPet(
            root=data_path, split="test", download=True, transform=preprocess
        )

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, 64, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": refine_classnames(test_data.classes),
            "mask": np.zeros((128, 128)),
        }
    elif dataset in [
        "food101",
        "sun397",
        "ucf101",
        "stanfordcars",
    ]:
        preprocess = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        train_data = COOPLMDBDataset(
            root=data_path, split="train", transform=preprocess
        )
        test_data = COOPLMDBDataset(root=data_path, split="test", transform=preprocess)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=8
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=8),
        }
        configs = {
            "class_names": refine_classnames(test_data.classes),
            "mask": np.zeros((128, 128)),
        }
    elif dataset == "caltech101":
        data = np.load(args.caltech_path,allow_pickle=True)
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_val"]
        y_test = data["y_val"]


        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Lambda(lambda x: x.convert("RGB")),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )


        train_data = CustomDataset(X_train, y_train, preprocess)
        test_data = CustomDataset(X_test, y_test, preprocess) 
        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=8
            ),
            "test": DataLoader(test_data, 128, shuffle=False, num_workers=8),
        }
        configs = {
            "class_names": [f"{i}" for i in range(101)],
            "mask": np.zeros((128, 128)),
        }

    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs


def prepare_additive_data(args, dataset, data_path, preprocess):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=preprocess
        )
        test_data = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=preprocess
        )
        class_names = refine_classnames(test_data.classes)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(
            root=data_path, train=True, download=False, transform=preprocess
        )
        test_data = datasets.CIFAR100(
            root=data_path, train=False, download=False, transform=preprocess
        )
        class_names = refine_classnames(test_data.classes)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.SVHN(
            root=data_path, split="test", download=True, transform=preprocess
        )
        class_names = [f"{i}" for i in range(10)]

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "dtd":
        train_data = datasets.DTD(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.DTD(
            root=data_path, split="test", download=True, transform=preprocess
        )
        class_names = refine_classnames(test_data.classes)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "flowers102":
        train_data = datasets.Flowers102(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.Flowers102(
            root=data_path, split="test", download=True, transform=preprocess
        )
        class_names = [f"{i}" for i in range(102)]

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "eurosat":
        D = datasets.EuroSAT(root=data_path, download=True, transform=preprocess)
        X_train, X_test, y_train, y_test = train_test_split(
            D.samples, D.targets, test_size=0.1, stratify=D.targets, random_state=1
        )

        train_data = datasets.EuroSAT(root=data_path, transform=preprocess)
        train_data.data = X_train
        train_data.targets = y_train

        test_data = datasets.EuroSAT(root=data_path, transform=preprocess)
        test_data.data = X_test
        test_data.targets = y_test

        class_names = refine_classnames(test_data.classes)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "oxfordpets":
        train_data = datasets.OxfordIIITPet(
            root=data_path, split="trainval", download=True, transform=preprocess
        )
        test_data = datasets.OxfordIIITPet(
            root=data_path, split="test", download=True, transform=preprocess
        )
        class_names = refine_classnames(test_data.classes)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset in [
        "food101",
        "sun397",
        "ucf101",
        "stanfordcars",
    ]:
        train_data = COOPLMDBDataset(
            root=data_path, split="train", transform=preprocess
        )
        test_data = COOPLMDBDataset(root=data_path, split="test", transform=preprocess)
        class_names = refine_classnames(test_data.classes)

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=8
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.GTSRB(
            root=data_path, split="test", download=True, transform=preprocess
        )
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
    elif dataset == "abide":
        D = ABIDE(root=data_path)
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    IMAGENETNORMALIZE["mean"], IMAGENETNORMALIZE["std"]
                ),
            ]
        )
        X_train, X_test, y_train, y_test = train_test_split(
            D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1
        )
        train_data = ABIDE(root=data_path, transform=preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root=data_path, transform=preprocess)
        test_data.data = X_test
        test_data.targets = y_test

        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=2
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=2),
        }
        class_names = ["non ASD", "ASD"]
        
    elif dataset == "caltech101":
        data = np.load(args.caltech_path,allow_pickle=True)
        X_train = data["x_train"]
        y_train = data["y_train"]
        X_test = data["x_val"]
        y_test = data["y_val"]


        # preprocess = transforms.Compose(
        #     [
        #         transforms.Lambda(lambda x: x.convert("RGB")),
        #         transforms.Resize((128, 128)),
        #         transforms.ToTensor(),
        #     ]
        # )


        train_data = CustomDataset(X_train, y_train, preprocess)
        test_data = CustomDataset(X_test, y_test, preprocess) 
        if args.n_shot > 0:
            train_data = sample_n_shots(args, train_data)
        loaders = {
            "train": DataLoader(
                train_data, args.batch_size, shuffle=True, num_workers=8
            ),
            "test": DataLoader(test_data, args.batch_size, shuffle=False, num_workers=8),
        }
        class_names = [f"{i}" for i in range(101)]

    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names


def prepare_gtsrb_fraction_data(data_path, fraction, batch_size,preprocess=None):
    data_path = os.path.join(data_path, "gtsrb")
    assert 0 < fraction <= 1
    new_length = int(fraction * 26640)
    indices = torch.randperm(26640)[:new_length]
    sampler = SubsetRandomSampler(indices)
    if preprocess == None:
        preprocess = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        train_data = datasets.GTSRB(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.GTSRB(
            root=data_path, split="test", download=True, transform=preprocess
        )
        loaders = {
            "train": DataLoader(train_data, batch_size, sampler=sampler, num_workers=2),
            "test": DataLoader(test_data, batch_size, shuffle=False, num_workers=2),
        }
        configs = {
            "class_names": refine_classnames(list(GTSRB_LABEL_MAP.values())),
            "mask": np.zeros((32, 32)),
        }
        return loaders, configs
    else:
        train_data = datasets.GTSRB(
            root=data_path, split="train", download=True, transform=preprocess
        )
        test_data = datasets.GTSRB(
            root=data_path, split="test", download=True, transform=preprocess
        )
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            "train": DataLoader(train_data, batch_size, sampler=sampler, num_workers=2),
            "test": DataLoader(test_data, batch_size, shuffle=False, num_workers=2),
        }
        return loaders, class_names