import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class GreppDataLoader(DataLoader):
    def __init__(self):
        pass

    def __call__(self,
                 path,
                 imgsz=[64,64],
                 batch_size=1,
                 rgb=True,
                 shuffle=False,
                 world_size=1,
                 rank=0,
                 num_workers=0):

        self.dataset = GreppDataset(
            path, imgsz, rgb
        )
        self.sampler = DistributedSampler(
            self.dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,          # Important to set shuffle to False when using DistributedSampler
            drop_last=True,
            pin_memory=False,
            num_workers=num_workers,
            sampler=self.sampler,
        )
        return self.dataloader, self.sampler

    def _aug(self):
        pass


class GreppDataset(Dataset):
    def __init__(self,
                 path,
                 imgsz=[64,64],
                 rgb=True) -> None:
        super(GreppDataset, self).__init__()

        self.imgsz = imgsz
        self.path = path
        self.rgb = rgb

        self.transform = T.Compose([
            T.Resize(self.imgsz),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.classes, self.labels, self.frames = self.get_items()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        label = self.labels[idx]
        frame = Image.open(self.frames[idx])

        if self.rgb:
            frame = frame.convert("RGB")
        else:
            frame = frame.convert("L")

        frame = self.transform(frame)
        return frame, label

    def get_items(self):
        classes = os.listdir(self.path)

        labels = []
        frames = []
        for c in classes:

            frame_dir = os.path.join(self.path, c)
            for f in os.listdir(frame_dir):
                frame_path = os.path.join(frame_dir, f)

                frames.append(frame_path)
                labels.append(classes.index(c))

        return classes, labels, frames


def test():

    DataLoaderInstance = GreppDataLoader()
    train_loader, train_sampler = DataLoaderInstance(
            path='/workspace/grepp/data/train',
            imgsz=(64,64),
            batch_size=6,
            rgb=True,
            shuffle=True,
            world_size=3,
            rank=0,
            num_workers=2 * 1,
    )

    for i, batch in enumerate(train_loader):
        X, y = batch
        print(i, X, y.shape)

        break

        # y_ = y[0][0].unsqueeze(dim=0)
        # to_pil = ToPILImage()
        # img = to_pil(y_)

        # img.save(f'./output/results/_.jpg')


if __name__=='__main__':
    test()