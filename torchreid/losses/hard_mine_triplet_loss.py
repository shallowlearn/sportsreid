from __future__ import division, absolute_import
import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, num_instances = None, use_gpu=True, topk = 1, bottomk=1):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.num_instances = num_instances
        self.use_gpu = use_gpu
        self.topk = topk
        self.bottomk = bottomk

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            # This topk code is equal to commented lines when topk = 1. Verified
            '''
            Motivation is a comment in a pytorch forum thread I found that 
            said using k hard negatives instead of just the hard one worked
            better. If k = 1, it becomes usual batch hard strategy
            '''
            dist_ap.append(dist[i][mask[i]].topk(self.topk).values)
            # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            '''
            Pytorch does not have a bottom k function. So multiply by -1 to find the min
            Make sure you multiple the final dist_an tensor outside
            this for loop with another compensating -1.
            '''
            dist_an.append((-1*dist[i][mask[i] == 0]).topk(self.bottomk).values)
            # dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        # Multiply back with -1 used to calculate the min
        dist_an = -1*torch.cat(dist_an)

        # dist_ap and dist_an might not have equal lengths
        # depending on topk and bottomk. Usually
        # length of ap < length. We repeat the last value
        # to make them the same length
        if len(dist_ap) < len(dist_an):
            dist_ap = torch.cat([dist_ap,
                                 dist_ap[-1:].repeat(len(dist_an) - len(dist_ap))])
        elif len(dist_an) < len(dist_ap):
            dist_an = torch.cat([dist_an,
                                 dist_an[-1:].repeat(len(dist_ap) - len(dist_an))])
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        # return self.ranking_loss(dist_an, dist_ap, y)
        triplet_loss = self.ranking_loss(dist_an, dist_ap, y)

        # For each anchor also compute centroid of intra class and other classes
        # and reduce distance from former and increase distance from latter.
        # Also make first cluster tight
        dist_cp, dist_cn = [], []
        cc_loss = 0#torch.tensor(0)
        for i in range(n):
            intra_class_centroid = torch.mean(inputs[mask[i]], dim=0, keepdim=True)
            other_class_centroid = torch.mean(inputs[mask[i] == 0], dim=0, keepdim=True)

            dist_cp.append(torch.cdist(intra_class_centroid, inputs[i:i+1]))
            dist_cn.append(torch.cdist(other_class_centroid, inputs[i:i+1]))

            # Distance between intra class centroid and other class centroid
            cc_loss = cc_loss + torch.cdist(intra_class_centroid, other_class_centroid)
        # We want to increase cc_loss or decrease -cc_loss
        # and take the mean
        cc_loss *= -1/n

        dist_cp = torch.cat(dist_cp)
        dist_cn = torch.cat(dist_cn)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_cn)
        centroid_triplet_loss = self.ranking_loss(dist_cn, dist_cp, y)

        return triplet_loss, centroid_triplet_loss, cc_loss[0,0]
