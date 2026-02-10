import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datasets.loader_downstream import MoleculeDataset, DownstreamMoleculeDataset

from datasets.utils.data_utils import mol_frag_collate
from utils import EarlyStopping

# models
from models.mamba_fuser import FusionMamba
# utils
from config_reg import cfg
from datasets.utils.logger import setup_logger, set_seed, dict_to_markdown

class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # 确保预测值非负
        y_pred = F.relu(y_pred)

        log_pred = torch.log(y_pred + 1)
        log_true = torch.log(y_true + 1)

        return torch.mean(torch.pow(log_pred - log_true, 2))

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

    for filename in glob.glob(os.path.join(cfg.save_dir, f'{cfg.model}_*')):
        os.remove(filename)

    writer = SummaryWriter(log_dir=cfg.save_dir, comment=f'cfg.dataset.lower()',
                           filename_suffix=f'{cfg.dataset.lower()}_{cfg.model.lower()}')
    # load dataset
    # 传递cutoff参数，用于预计算radius_graph的边和距离
    train_dataset = DownstreamMoleculeDataset(dataset=cfg.dataset, split='train', root=os.path.join(cfg.data_dir, cfg.dataset), cutoff=cfg.cutoff)
    valid_dataset = DownstreamMoleculeDataset(dataset=cfg.dataset, split='valid', root=os.path.join(cfg.data_dir, cfg.dataset), cutoff=cfg.cutoff)
    test_dataset = DownstreamMoleculeDataset(dataset=cfg.dataset, split='test', root=os.path.join(cfg.data_dir, cfg.dataset), cutoff=cfg.cutoff)


    num_tasks = train_dataset[0].y.shape[-1]

    logger.info(
        f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Number of tasks: {num_tasks}")

    trainloader = DataLoader(train_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=True,
                             num_workers=0, drop_last=True)

    valloader = DataLoader(valid_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False,
                           num_workers=0, drop_last=True)

    testloader = DataLoader(test_dataset, batch_size=cfg.batchsize, collate_fn=mol_frag_collate, shuffle=False,
                            num_workers=0, drop_last=True)

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
    criterion = MSLELoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    best_val = 10000.
    best_test = None

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

            y = batch.y.view(pred.shape).to(torch.float64)
            loss_task = criterion(pred.double(), y)
            loss = loss_task

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            cum_loss += float(loss.cpu().item())
            cum_loss_task += float(loss_task.cpu().item())

        print_loss = cum_loss / len(trainloader)
        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        train_mae = torch.mean(torch.abs(y_true - y_pred))
        writer.add_scalar('Training Loss', print_loss, epoch)
        writer.add_scalar('Train MAE', train_mae, epoch)


        # VAL
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(valloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        val_mae = torch.mean(torch.abs(y_true - y_pred))
        writer.add_scalar('Valid MAE', val_mae, epoch)

        # TEST
        model.eval()
        y_pred = []
        y_true = []
        for step, batch in enumerate(testloader):
            # batch.y[batch.y == -1] = 0
            batch = batch.to(device)
            with torch.no_grad():
                pred = model(batch)

                # loss_frag_div, loss_frag, loss_tree, loss_mask, loss_task = losses

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        test_mae = torch.mean(torch.abs(y_true - y_pred))
        writer.add_scalar('Test MAE', test_mae, epoch)

        logger.info(
            f"Epoch: {epoch:03d}/{cfg.epoch:03d}, Total Loss: {print_loss:.4f}, "
            f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f}"
        )

        if val_mae < best_val:
            best_val = val_mae
            best_test = test_mae
            best_epoch = epoch
            logger.info(f'Best step at epoch {epoch}')

    best_val_metric = np.min(val_metrics)
    best_test_metric = np.min(test_metrics)
    true_test_metric = test_metrics[np.argmin(val_metrics)]

    logger.info(f'best_val_metric:{best_val_metric:.4f}\t'
              f'best_test_metric:{best_test_metric:.4f}\t'
              f'true_test_metric:{true_test_metric:.4f}')

    if cfg.save_model:
        torch.save(best_model.state_dict(), os.path.join(cfg.save_dir, f'{cfg.model}_{best_epoch:03d}.pth'))
    logger.info(f"Best Val MAE: {best_val.item()}, \tBest Test MAE: {best_test.item()}")
    writer.close()
