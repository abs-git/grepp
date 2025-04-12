import os
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import yaml
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from grepp.model.model import ConvNet, End2End
from grepp.train.utils.dataloader import GreppDataLoader
from grepp.train.utils.draw_metric import save_history
from grepp.train.utils.grad import GradCAM, get_overlay
from grepp.train.utils.loss import CrossEntropy_Loss, SoftmaxFocal_Loss
from grepp.train.utils.metric import get_metric, outcome
from grepp.train.utils.optim import get_optimizer, get_scheduler


class Train(object):
    def __init__(self, cfg):

        world_size = torch.cuda.device_count()
        rank = torch.distributed.get_rank()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        base_model = ConvNet()
        if cfg.default.checkpoint != None:
            ckpt = torch.load(cfg.default.checkpoint, weights_only=True)
            base_model.load_state_dict(ckpt)

        model = End2End(base_model=base_model,
                        num_classes=3)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )

        optimizer = get_optimizer(cfg.default.optimizer, model, cfg.default.lr)
        scheduler = get_scheduler(cfg.default.scheduler,
                                  optimizer,
                                  int(cfg.default.epochs / 3),
                                  cfg.default.gamma)
        scaler = torch.amp.GradScaler(enabled=cfg.default.amp)

        ce_loss = CrossEntropy_Loss()
        fc_loss = SoftmaxFocal_Loss()
        focal_ratio = cfg.default.focal_ratio

        if rank == 0:
            save_dir = self._check_dir(cfg)

        # data loader
        DataLoaderInstance = GreppDataLoader()
        train_loader, train_sampler = DataLoaderInstance(
            path=cfg.data.train,
            imgsz=cfg.default.imgsz,
            batch_size=cfg.default.batch_size,
            rgb=cfg.default.rgb,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            num_workers=cfg.default.num_workers * world_size
        )
        val_loader, val_sampler = DataLoaderInstance(
            path=cfg.data.val,
            imgsz=cfg.default.imgsz,
            batch_size=cfg.default.batch_size,
            rgb=cfg.default.rgb,
            world_size=world_size,
            rank=rank,
            shuffle=True,
            num_workers=cfg.default.num_workers * world_size,
        )

        test_dataset = defaultdict(list)
        for c in os.listdir(cfg.data.test):
            dataset_dir = os.path.join(cfg.data.test, c)
            for i in sorted(os.listdir(dataset_dir)):
                test_dataset[c].append(os.path.join(dataset_dir, i))

        # grad cam
        conv_layers = {
            "conv1": model.module.base.conv1,
            "conv2": model.module.base.conv2,
            "conv3": model.module.base.conv3,
        }
        transform = T.Compose([
            T.Resize((64,64)),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        cam = GradCAM(model, conv_layers)


        # setting
        self.cfg = cfg

        self.world_size = world_size
        self.rank = rank
        self.device = device

        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

        self.train_loader = train_loader
        self.train_sampler = train_sampler
        self.val_loader = val_loader
        self.val_sampler = val_sampler
        self.test_dataset = test_dataset

        self.conv_layers = conv_layers
        self.transform = transform
        self.cam = cam

        self.ce_loss = ce_loss
        self.fc_loss = fc_loss
        self.focal_ratio = focal_ratio

        self.epochs = cfg.default.epochs
        self.best_val_loss = float("inf")
        self.val_f1 = 0

        self.save_dir = save_dir
        self.save_interval = cfg.output.interval

    def __call__(self):
        self._save_config(self.cfg)

        train_history = {"loss": [],
                         "precision": [],
                         "recall": [],
                         "f1": [],
                         "miss_rate": [],
                         "accuracy": [],
                         "TP": [],
                         "FP": [],
                         "TN": [],
                         "FN": []
                        }

        val_history = {"loss": [],
                      "precision": [],
                      "recall": [],
                      "f1": [],
                      "miss_rate": [],
                      "accuracy": [],
                      "TP": [],
                      "FP": [],
                      "TN": [],
                      "FN": []
                     }

        for e in range(self.epochs):

            print(f'\n Epoch: {e}, lr: {self.scheduler.get_last_lr()[0]}, ')

            train_results = self.train()

            self.scheduler.step()

            for k in train_history.keys():
                train_history[k].append(train_results[k])

            if self.rank == 0:
                val_results = self.validation()

                for k in val_history.keys():
                    val_history[k].append(val_results[k])

                if e % self.save_interval == 0:
                    self._save_artifacts(e)

        save_history(train_history, val_history, self.save_dir)

        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()

    def train(self):
        self.model.train()

        running_loss = 0.0
        TP, FP, TN, FN = .0, .0, .0, .0

        pbar = tqdm(self.train_loader, unit=' batch', ncols=180)
        for batch_index, (X, y) in enumerate(pbar):
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.cfg.default.amp):
                y_pred = self.model(X)
                loss = self.ce_loss(y, y_pred) * (1 - self.focal_ratio) + self.fc_loss(y,y_pred) * self.focal_ratio

            self.optimizer.zero_grad()

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # metrics
            running_loss += loss.item() * X.size(0)

            tp, fp, tn, fn = outcome(y, y_pred)
            TP += tp
            FP += fp
            TN += tn
            FN += fn
            precision, recall, f1, miss_rate, acc = get_metric(TP, FP, TN, FN)

            # logging
            log = "Train loss: {:.7f} | TP: {}, FP: {}, TN: {}, FN: {} | Prec: {:.3f}, Recall: {:.3f}, f1: {:.3f}, Miss Rate: {:.3f}".format(
                running_loss / ((batch_index + 1) * X.shape[0]),
                TP, FP, TN, FN,
                precision, recall, f1, miss_rate, acc
            )

            pbar.set_description(log)

        train_loss = running_loss / len(self.train_loader)

        return {"loss": train_loss,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "miss_rate": miss_rate,
                "accuracy": acc,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN}


    def validation(self):
        self.model.eval()

        loss_sum = 0.0
        TP, FP, TN, FN = .0, .0, .0, .0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, ncols=180, unit=' batch')
            for batch_index, (X, y) in enumerate(pbar):
                X, y = X.to(self.device), y.to(self.device)

                y_pred = self.model(X)
                loss = self.ce_loss(y, y_pred) * (1 - self.focal_ratio) + self.fc_loss(y,y_pred) * self.focal_ratio

                tp, fp, tn, fn = outcome(y, y_pred)
                TP += tp
                FP += fp
                TN += tn
                FN += fn
                precision, recall, f1, miss_rate, acc = get_metric(TP, FP, TN, FN)

                loss_sum += loss.item() * X.size(0)

                # logging
                log = "Valid loss: {:.7f} | TP: {}, FP: {}, TN: {}, FN: {} | Prec: {:.3f}, Recall: {:.3f}, f1: {:.3f}, Miss Rate: {:.3f}".format(
                    loss_sum / ((batch_index + 1) * X.shape[0]),
                    TP, FP, TN, FN,
                    precision, recall, f1, miss_rate, acc
                )
                pbar.set_description(log)

            val_loss = loss_sum / len(self.val_loader)

        tqdm.write(f"\n--- Saving weights to: {self.save_dir}/last.pt ---")
        torch.save(self.model.module.state_dict(), f"{self.save_dir}/last.pt")
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            tqdm.write(f"--- Saving weights to: {self.save_dir}/best.pt ---")
            torch.save(self.model.module.state_dict(), f"{self.save_dir}/best.pt")

        return {"loss": val_loss,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "miss_rate": miss_rate,
                "accuracy": acc,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN}


    def _check_dir(self, cfg):
        tag=0
        while True:
            save_dir = os.path.join(cfg.output.root, f'output_{tag}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                return save_dir
            tag += 1


    def _save_artifacts(self, epoch):

        ckpt_dir = os.path.join(self.save_dir, f'save_{epoch}')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)

        for c, items in self.test_dataset.items():

            overlays_row = []
            for p in items:
                image = Image.open(p)
                inp_image = self.transform(image).unsqueeze(0)
                raw_image = np.array(image.resize((64, 64)))

                cam_maps = self.cam.generate(inp_image)

                overlays_col = []
                for name, _ in self.conv_layers.items():
                    cam_map = cam_maps[name]
                    overlay = get_overlay(cam_map, raw_image)
                    overlays_col.append(overlay)
                    combined_col = np.concatenate(overlays_col, axis=1)

                overlays_row.append(combined_col)

            combined_images = np.concatenate(overlays_row, axis=0)

            cv2.imwrite(f"{ckpt_dir}/{c}_gradcam_{epoch}.jpg",  combined_images)

    def _save_config(self, cfg):

        def edict_to_dict(d):
            if isinstance(d, dict):
                return {k: edict_to_dict(v) for k, v in d.items()}
            return d

        with open(f"{self.save_dir}/config.yaml", "w") as f:
            yaml.safe_dump(edict_to_dict(cfg), f, default_flow_style=False)

