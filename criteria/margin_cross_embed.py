import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer
import os
os.environ['PYTHONBREAKPOINT']='ipdb.set_trace'
"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = True

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.margin             = opt.loss_margin_margin
        self.nu                 = opt.loss_margin_nu
        self.beta_constant      = opt.loss_margin_beta_constant
        self.beta_val           = opt.loss_margin_beta

        if opt.loss_margin_beta_constant:
            self.beta = opt.loss_margin_beta
        else:
            self.beta = torch.nn.Parameter(torch.ones(opt.n_classes)*opt.loss_margin_beta)

        self.batchminer = batchminer
        self.name  = 'margin_cross'
        self.lr    = opt.loss_margin_beta_lr

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM



    def forward(self, embeds, labels, batch, model, **kwargs):
        sampled_triplets = self.batchminer(embeds, labels)

        if len(sampled_triplets):
            d_ap, d_an = [],[]
            for triplet in sampled_triplets:

                train_triplet = {'Anchor': batch[triplet[0],:].unsqueeze(0), 'Positive':batch[triplet[1],:].unsqueeze(0), 'Negative':batch[triplet[2]].unsqueeze(0)}
                anchor_0, positive_0 = model(train_triplet['Anchor'], train_triplet['Positive'])
                anchor_1, negative_1 = model(train_triplet['Anchor'], train_triplet['Negative'])
                if kwargs['cat_global']:
                    # breakpoint()
                    anchor_0 = torch.cat([anchor_0, embeds[triplet[0]].unsqueeze(0)], dim=-1)
                    anchor_1 = torch.cat([anchor_1, embeds[triplet[0]].unsqueeze(0)], dim=-1)
                    positive_0 = torch.cat([positive_0, embeds[triplet[1]].unsqueeze(0)], dim=-1)
                    negative_1 = torch.cat([negative_1, embeds[triplet[2]].unsqueeze(0)], dim=-1)
                    anchor_0 = torch.nn.functional.normalize(anchor_0, dim=-1)
                    anchor_1 = torch.nn.functional.normalize(anchor_1, dim=-1)
                    positive_0 = torch.nn.functional.normalize(positive_0, dim=-1)
                    negative_1 = torch.nn.functional.normalize(negative_1, dim=-1)

                pos_dist = ((anchor_0-positive_0).pow(2).sum()+1e-8).pow(1/2)
                neg_dist = ((anchor_1-negative_1).pow(2).sum()+1e-8).pow(1/2)

                d_ap.append(pos_dist)
                d_an.append(neg_dist)
            d_ap, d_an = torch.stack(d_ap), torch.stack(d_an)

            if self.beta_constant:
                beta = self.beta
            else:
                beta = torch.stack([self.beta[labels[triplet[0]]] for triplet in sampled_triplets]).to(torch.float).to(d_ap.device)

            pos_loss = torch.nn.functional.relu(d_ap-beta+self.margin)
            neg_loss = torch.nn.functional.relu(beta-d_an+self.margin)

            pair_count = torch.sum((pos_loss>0.)+(neg_loss>0.)).to(torch.float).to(d_ap.device)

            if pair_count == 0.:
                loss = torch.sum(pos_loss+neg_loss)
            else:
                loss = torch.sum(pos_loss+neg_loss)/pair_count

            if self.nu: loss = loss + beta_regularisation_loss.to(torch.float).to(d_ap.device)
        else:
            loss = torch.tensor(0.).to(torch.float).to(batch.device)

        return loss
