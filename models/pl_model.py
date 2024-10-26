import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

import utils.train as utils_train
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D

from graphbap.bapnet import BAPNet


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


class MoleculeTrainer(pl.LightningModule):
    def __init__(self, config, protein_featurizer, ligand_featurizer, net_cond):
        super(MoleculeTrainer, self).__init__()
        self.config = config
        self.protein_featurizer = protein_featurizer
        self.ligand_featurizer = ligand_featurizer
        self.net_cond = net_cond

        self.model = ScorePosNet3D(
            config.model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        )
        self.save_hyperparameters()

        # For training logging
        self.train_iterations = 0
        # For validation logging
        self.all_pred_v, self.all_true_v = [], []
        self.sum_loss, self.sum_loss_pos, self.sum_loss_v, self.sum_n = 0, 0, 0, 0

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = utils_train.get_optimizer(self.config.train.optimizer, self.model)
        scheduler = utils_train.get_scheduler(self.config.train.scheduler, optimizer)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'validation/loss'
                }
            }
        else:
            return [optimizer], [scheduler]

    def on_train_epoch_start(self):
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        self.model.train()
        optimizer = self.optimizers()

        all_batch = batch
        topk_prompt = self.config.data.topk_prompt

        all_batch = [b.to(self.device) for b in all_batch]
        assert len(all_batch) == topk_prompt + 1, "wrong value of topk_prompt"

        prompt_batch_2, prompt_batch_3 = None, None
        if topk_prompt == 1:
            batch, prompt_batch = all_batch
        elif topk_prompt == 2:
            batch, prompt_batch, prompt_batch_2 = all_batch
        elif topk_prompt == 3:
            batch, prompt_batch, prompt_batch_2, prompt_batch_3 = all_batch
        else:
            raise ValueError(topk_prompt)

        gt_protein_pos = batch.protein_pos

        results = self.model.get_diffusion_loss(
            net_cond=self.net_cond,
            protein_pos=gt_protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,

            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,

            prompt_ligand_pos=prompt_batch.ligand_pos,
            prompt_ligand_v=prompt_batch.ligand_atom_feature_full,
            prompt_batch_ligand=prompt_batch.ligand_element_batch,

            prompt_ligand_pos_2=prompt_batch_2.ligand_pos if prompt_batch_2 is not None else None,
            prompt_ligand_v_2=prompt_batch_2.ligand_atom_feature_full if prompt_batch_2 is not None else None,
            prompt_batch_ligand_2=prompt_batch_2.ligand_element_batch if prompt_batch_2 is not None else None,

            prompt_ligand_pos_3=prompt_batch_3.ligand_pos if prompt_batch_3 is not None else None,
            prompt_ligand_v_3=prompt_batch_3.ligand_atom_feature_full if prompt_batch_3 is not None else None,
            prompt_batch_ligand_3=prompt_batch_3.ligand_element_batch if prompt_batch_3 is not None else None
        )
        loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']
        loss = loss / self.config.train.n_acc_batch

        self.train_iterations += 1
        if self.train_iterations % self.config.train.train_report_iter == 0:
            self.log('iteration', self.train_iterations)
            self.log('train/loss', loss)
            self.log('train/loss_pos', loss_pos)
            self.log('train/loss_v', loss_v)
        return loss

    def on_validation_epoch_start(self):
        self.all_pred_v, self.all_true_v = [], []
        self.sum_loss, self.sum_loss_pos, self.sum_loss_v, self.sum_n = 0, 0, 0, 0
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        all_batch = batch
        topk_prompt = self.config.data.topk_prompt

        all_batch = [b.to(self.device) for b in all_batch]

        prompt_batch_2, prompt_batch_3 = None, None
        if topk_prompt == 1:
            batch, prompt_batch = all_batch
        elif topk_prompt == 2:
            batch, prompt_batch, prompt_batch_2 = all_batch
        elif topk_prompt == 3:
            batch, prompt_batch, prompt_batch_2, prompt_batch_3 = all_batch
        else:
            raise ValueError(topk_prompt)

        batch_size = batch.num_graphs
        for t in np.linspace(0, self.model.num_timesteps - 1, 10).astype(int):
            time_step = torch.tensor([t] * batch_size).to(self.device)
            results = self.model.get_diffusion_loss(
                net_cond=self.net_cond,
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,

                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,

                prompt_ligand_pos=prompt_batch.ligand_pos,
                prompt_ligand_v=prompt_batch.ligand_atom_feature_full,
                prompt_batch_ligand=prompt_batch.ligand_element_batch,

                prompt_ligand_pos_2=prompt_batch_2.ligand_pos if prompt_batch_2 is not None else None,
                prompt_ligand_v_2=prompt_batch_2.ligand_atom_feature_full if prompt_batch_2 is not None else None,
                prompt_batch_ligand_2=prompt_batch_2.ligand_element_batch if prompt_batch_2 is not None else None,

                prompt_ligand_pos_3=prompt_batch_3.ligand_pos if prompt_batch_3 is not None else None,
                prompt_ligand_v_3=prompt_batch_3.ligand_atom_feature_full if prompt_batch_3 is not None else None,
                prompt_batch_ligand_3=prompt_batch_3.ligand_element_batch if prompt_batch_3 is not None else None,

                time_step=time_step
            )
            loss, loss_pos, loss_v = results['loss'], results['loss_pos'], results['loss_v']

            self.sum_loss += float(loss) * batch_size
            self.sum_loss_pos += float(loss_pos) * batch_size
            self.sum_loss_v += float(loss_v) * batch_size
            self.sum_n += batch_size
            self.all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
            self.all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = self.sum_loss / self.sum_n

        return avg_loss

    def on_validation_epoch_end(self):
        avg_loss = self.sum_loss / self.sum_n
        avg_loss_pos = self.sum_loss_pos / self.sum_n
        avg_loss_v = self.sum_loss_v / self.sum_n
        atom_auroc = get_auroc(np.concatenate(self.all_true_v), np.concatenate(self.all_pred_v, axis=0),
                               feat_mode=self.config.data.transform.ligand_atom_mode)
        self.log('epoch', self.current_epoch)
        self.log('validation/loss', avg_loss)
        self.log('validation/loss_pos', avg_loss_pos)
        self.log('validation/loss_v', avg_loss_v * 1000)
        self.log('validation/atom_auroc', atom_auroc)
