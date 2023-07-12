import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics.functional import accuracy
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def torch_rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 tensor, original batch of point clouds
        Return:
          BxNx3 tensor, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = torch.rand(1) * 2 * np.pi

    rotated_data = torch.zeros(batch_data.shape, dtype=torch.float32)
    for k in range(batch_data.shape[0]):
        cosval = torch.cos(rotation_angle)
        sinval = torch.sin(rotation_angle)
        rotation_matrix = torch.tensor([[cosval, sinval, 0],
                                        [-sinval, cosval, 0],
                                        [0, 0, 1]]).cuda()

        shape_pc = batch_data[k, ...]  # [4096, 3]
        rotated_data[k, ...] = torch.mm(shape_pc,
                                        rotation_matrix)  # .t()  # Transpose the result back to original shape

    return rotated_data

    ####################################################################################################################


class Classifier(LightningModule):

    def __init__(self, num_classes,
                 encoder,
                 num_training_samples,
                 batch_size,
                 max_epochs=50,
                 learning_rate=0.001,
                 dropout=0.3):
        super(Classifier, self).__init__()

        self.encoder = encoder

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.num_samples = num_training_samples
        self.train_iters_per_epoch = num_training_samples // batch_size
        self.loss_fn = nn.NLLLoss()

        self.fc_1 = nn.Linear(1024, 512)
        self.fc_2 = nn.Linear(512, 256)
        self.fc_3 = nn.Linear(256, num_classes)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.bn_1(self.fc_1(x)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)

        return F.log_softmax(self.fc_3(x), dim=1)

    def training_step(self, batch, batch_idx):
        x1, x2, label = batch  # x1 torch.Size([32, 4096, 3])
        y_hat = self(x1)
        loss = self.loss_fn(y_hat, label)
        self.log("barlow_cls_train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, label = batch  # x1 torch.Size([32, 4096, 3])
        y_hat = self(x1)
        loss = self.loss_fn(y_hat, label)
        self.log("barlow_cls_val_loss", loss, on_step=False, on_epoch=True)
        # Accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, label, task='binary')
        self.log("barlow_cls_val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x1, x2, labels = batch  # x1 torch.Size([32, 4096, 3])
        y_hat = self.forward(x1)  ##[batch, 2]
        # Loss
        loss = self.loss_fn(y_hat, labels)
        # Accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy(preds, labels, task='binary')
        correct = torch.sum(preds == labels)

        return {'test_loss': loss, 'test_acc': acc, 'preds': preds, 'targets': labels}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        avg_acc = torch.stack([x['test_acc'].float() for x in outputs]).sum() / len(outputs)  # self.num_samples

        preds_stack = torch.stack([x['preds'] for x in outputs])
        labels_stack = torch.stack([x['targets'] for x in outputs])

        # calculate F1 score
        f1 = f1_score(labels_stack.cpu().numpy(), preds_stack.cpu().detach().numpy())

        # Compute fpr, tpr, thresholds and roc auc
        fpr, tpr, thresholds = roc_curve(labels_stack.cpu().numpy(), preds_stack.cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)

        logs = {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'F1_score': f1, 'ROC AUC': roc_auc}
        self.log_dict(logs)

        return {'avg_test_loss': avg_loss, 'avg_test_acc': avg_acc, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    ####################################################################################################################
