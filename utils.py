import copy

import numpy as np


class EarlyStopping:
    """
    通用早停机制，支持最大化和最小化指标。
    同时内部保存最佳模型权重。
    """

    def __init__(self, patience=7, min_delta=0, mode='min', restore_best_weights=True):
        """
        Args:
            patience (int): 在被判断为没有提升后，可以容忍的 epoch 数量。
            min_delta (float): 判断为“提升”所需的最小变化量。
            mode (str): 'min' 或 'max'。'min' 模式下，监控的指标越小越好（如 loss）；
                        'max' 模式下，监控的指标越大越好（如 AUC, F1-score）。
            restore_best_weights (bool): 如果为True，在停止训练时，会将模型的权重恢复到最佳状态。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.counter = 0
        self.best_score = None
        self.best_weights = None

        if self.mode == 'min':
            self.val_op = np.less  # 使用 numpy.less 比较 a < b
            self.best_score = np.inf
        elif self.mode == 'max':
            self.val_op = np.greater  # 使用 numpy.greater 比较 a > b
            self.best_score = -np.inf
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Choose 'min' or 'max'.")

    def __call__(self, metric_value, model):
        """
        Args:
            metric_value (float): 当前 epoch 的验证指标值 (e.g., val_loss or val_auc)。
            model (torch.nn.Module): 待监控的模型。

        Returns:
            bool: 如果为 True，则表示应该停止训练。
        """
        # 使用 self.val_op 来判断当前指标是否优于历史最佳
        # min 模式: metric_value < self.best_score - self.min_delta
        # max 模式: metric_value > self.best_score + self.min_delta
        delta = self.min_delta if self.mode == 'max' else -self.min_delta

        if self.val_op(metric_value, self.best_score + delta):
            # 发现更优的指标
            self.best_score = metric_value
            self.counter = 0
            self.save_best_weights(model)
        else:
            # 未发现更优的指标
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs of no improvement.")
            if self.restore_best_weights:
                print(f"Restoring model weights to the end of the best epoch.")
                model.load_state_dict(self.best_weights)
            return True  # 返回 True，表示应该停止

        return False  # 返回 False，表示继续训练

    def save_best_weights(self, model):
        """保存最佳模型权重"""
        # 使用 copy.deepcopy 确保权重被完全复制，而不是引用
        self.best_weights = copy.deepcopy(model.state_dict())