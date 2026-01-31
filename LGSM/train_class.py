import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.sys.path.append('./')
os.sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import glob
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from tensorboardX import SummaryWriter
import json
import copy

# datasets and splits
from datasets.loader_downstream import MoleculeDataset, DownstreamMoleculeDataset
from datasets.utils.data_utils import mol_frag_collate
from datasets.utils.splitters import scaffold_split, random_split, random_scaffold_split
from utils import EarlyStopping

# models
from models.mamba_fuser import FusionMamba
# utils
from config import cfg
from datasets.utils.logger import setup_logger, set_seed, dict_to_markdown


def compute_roc_auc(y_true, y_pred, epoch, logger=None):
    """
    计算多任务 ROC-AUC，并处理 nan/inf 与异常任务。
    参数：
        y_true: numpy.ndarray, shape [N, num_tasks]
        y_pred: numpy.ndarray, shape [N, num_tasks]
        epoch: 当前 epoch
        logger: 可选，用于记录警告
    返回：
        mean_auc: float，平均 ROC-AUC
        skipped_tasks: list[str]，被跳过的任务
    """
    roc_list = []
    skipped_tasks = []
    for i in range(y_true.shape[1]):
        num_pos = np.sum(y_true[:, i] == 1)
        num_neg = np.sum(y_true[:, i] == 0)
        if num_pos > 0 and num_neg > 0:
            is_valid = y_true[:, i] >= 0
            if np.isnan(y_pred[is_valid, i]).any():
                print(f'Epoch {epoch}, Task {i}, nan in y_pred')
                skipped_tasks.append(f'task_{i}_nan')
                continue
            if np.isinf(y_pred[is_valid, i]).any():
                print(f'Epoch {epoch}, Task {i}, inf in y_pred')
            try:
                s = roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i])
                roc_list.append(s)
            except ValueError as e:
                print(f'Epoch {epoch}, Task {i}, Error computing AUC: {e}')
                skipped_tasks.append(f'task_{i}_error')
        else:
            skipped_tasks.append(f'task_{i}_no_both_classes(pos={num_pos},neg={num_neg})')
            print(f'Epoch {epoch}, Task {i} is invalid: pos={num_pos}, neg={num_neg}')

    if len(roc_list) == 0:
        if logger:
            logger.warning(f'Epoch {epoch}, ROC list is empty! Skipped tasks: {skipped_tasks}')
        return 0.0, skipped_tasks

    mean_auc = sum(roc_list) / len(roc_list)
    return mean_auc, skipped_tasks


if __name__ == '__main__':
    set_seed(cfg.runseed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    logger = setup_logger(f"{cfg.model}", cfg.save_dir)
    logger.info('-' * 60)
    logger.info(f'The setup args are:\n{dict_to_markdown(cfg.to_dict(), max_str_len=120)}')

    cfg.save_yaml(filename=cfg.save_dir + 'config.yaml')

    for filename in glob.glob(os.path.join(cfg.save_dir, 'events.out.*')):
        os.remove(filename)

    writer = SummaryWriter(
        log_dir=cfg.save_dir,
        comment=f'cfg.dataset.lower()',
        filename_suffix=f'{cfg.dataset.lower()}_{cfg.model.lower()}'
    )

    # 加载数据集
    train_dataset = DownstreamMoleculeDataset(dataset=cfg.dataset, split='train',
                                              root=os.path.join(cfg.data_dir, cfg.dataset), cutoff=cfg.cutoff)
    valid_dataset = DownstreamMoleculeDataset(dataset=cfg.dataset, split='valid',
                                              root=os.path.join(cfg.data_dir, cfg.dataset), cutoff=cfg.cutoff)
    test_dataset = DownstreamMoleculeDataset(dataset=cfg.dataset, split='test',
                                             root=os.path.join(cfg.data_dir, cfg.dataset), cutoff=cfg.cutoff)

    num_tasks = train_dataset[0].y.shape[-1]

    logger.info(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of tasks: {num_tasks}")

    trainloader = DataLoader(train_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate,
                             shuffle=True, num_workers=0)
    valloader = DataLoader(valid_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate,
                           shuffle=False, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate,
                            shuffle=False, num_workers=0)

    model = FusionMamba(cfg, num_tasks=num_tasks)

    if cfg.pretrain_path:
        check_points = torch.load(cfg.pretrain_path, map_location=device)
        if 'gnn' in check_points.keys():
            model.gnn.load_state_dict(check_points['gnn'])
        else:
            model.gnn.load_state_dict(check_points['mol_gnn'])

    model.to(device)
    best_model = model
    logger.info(f"Model: {cfg.model}, Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    best_val = 0
    best_test = None
    best_epoch = 0

    early_stopping = EarlyStopping(patience=10, mode='max', restore_best_weights=True)
    train_metrics = []
    val_metrics = []
    test_metrics = []


    total_steps = len(trainloader)
    for epoch in range(cfg.epoch):
        model.train()
        cum_loss = 0
        cum_loss_frag_div = 0
        cum_loss_frag = 0
        cum_loss_tree = 0
        cum_loss_task = 0
        cum_loss_mask = 0
        y_pred = []
        y_true = []

        for step, batch in enumerate(tqdm(trainloader)):
            batch = batch.to(device)
            pred = model(batch)

            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

            # loss_frag_div, loss_frag, loss_tree, loss_mask = losses
            y = batch.y.view(pred.shape).to(torch.float64)
            is_valid = y >= 0
            loss_mat = criterion(pred.double(), y)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

            optimizer.zero_grad()
            loss_task = torch.sum(loss_mat) / torch.sum(is_valid)
            loss = loss_task

            loss.backward()
            optimizer.step()

            cum_loss += float(loss.cpu().item())
            # cum_loss_frag_div += float(loss_frag_div.cpu().item())
            # cum_loss_frag += float(loss_frag.cpu().item())
            # cum_loss_tree += float(loss_tree.cpu().item())
            # cum_loss_mask += float(loss_mask.cpu().item())
            cum_loss_task += float(loss_task.cpu().item())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        train_res, _ = compute_roc_auc(y_true, y_pred, epoch, logger)

        print_loss = cum_loss / len(trainloader)
        writer.add_scalar('Training Loss', print_loss, epoch)
        writer.add_scalar('Train ROC', train_res, epoch)

        # ---------- VALID ----------
        model.eval()
        y_pred, y_true = [], []
        for step, batch in enumerate(valloader):
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch['y'].view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        val_res, _ = compute_roc_auc(y_true, y_pred, epoch, logger)
        writer.add_scalar('Valid ROC', val_res, epoch)

        # ---------- TEST ----------
        model.eval()
        y_pred, y_true = [], []
        for step, batch in enumerate(testloader):
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        test_res, _ = compute_roc_auc(y_true, y_pred, epoch, logger)
        writer.add_scalar('Test ROC', test_res, epoch)

        train_metrics.append(train_res)
        val_metrics.append(val_res)
        test_metrics.append(test_res)

        if early_stopping(val_res, model):
            logger.info(f"Early stopping at epoch {epoch}")
            break

        logger.info(
            f"Epoch: {epoch:03d}/{cfg.epoch:03d}, Loss: {print_loss:.4f}, "
            f"Train ROC: {train_res:.4f}, Val ROC: {val_res:.4f}, Test ROC: {test_res:.4f}"
        )

        if val_res > best_val:
            best_val = val_res
            best_test = test_res
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            logger.info(f'Best step at epoch {epoch}')

    best_val_metric = np.max(val_metrics)
    best_test_metric = np.max(test_metrics)
    true_test_metric = test_metrics[np.argmax(val_metrics)]

    # logger.info(f"Best Val ROC: {best_val}, \tBest Test ROC: {best_test}")
    logger.info(f'best_val_metric:{best_val_metric:.4f}\t'
              f'best_test_metric:{best_test_metric:.4f}\t'
              f'true_test_metric:{true_test_metric:.4f}')
    if cfg.save_model:
        save_path = os.path.join(cfg.save_dir, f'{cfg.model}_{best_epoch:03d}.pth')
        torch.save(best_model.state_dict(),save_path)
        print("模型保存到：", save_path)
    writer.close()
