import torch
import torch.nn as nn
import cv2
# import training.ops
import numpy as np
import random
Default = {'margin': 0.5, 'alpha': None, 'beta': None, 'n_neg': 10}

# ops下的函数
def extract_kpt_vectors(tensor, kpts, rand_batch=False):
    """
    Pick channel vectors from 2D location in tensor.
    E.g. tensor[b, :, y1, x1]

    :param tensor: Tensor to extract from [b, c, h, w]
    :param kpts: Tensor with 'n' keypoints (x, y) as [b, n, 2]
    :param rand_batch: Randomize tensor in batch the vector is extracted from
    :return: Tensor entries as [b, n, c]
    """
    batch_size, num_kpts = kpts.shape[:-1]  # [b, n]

    # Reshape as a single batch -> [b*n, 2]
    tmp_idx = kpts.contiguous().view(-1, 2).long()

    # Flatten batch number indexes  -> [b*n] e.g. [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    b_num = torch.arange(batch_size)
    b_num = b_num.repeat((num_kpts, 1)).view(-1)
    b_num = torch.sort(b_num)[0] if not rand_batch else b_num[torch.randperm(len(b_num))]

    # Perform indexing and reshape to [b, n, c]
    return tensor[b_num, :, tmp_idx[:, 1], tmp_idx[:, 0]].reshape([batch_size, num_kpts, -1])

class PixelwiseContrastiveLoss(nn.Module):
    """
    Implementation of "pixel-wise" contrastive loss. Contrastive loss typically compares two whole images.
            L = (Y) * (1/2 * d**2) + (1 - Y) * (1/2 * max(0, margin - d)**2)
    In this instance, we instead compare pairs of features within those images.
    Positive matches are given by ground truth correspondences between images.
    Negative matches are generated on-the-fly based on provided parameters.
    Attributes:
        margin (float): Target margin distance between positives and negatives
        alpha (int): Minimum distance from original positive KeyPoint
        beta (int): Maximum distance from original positive KeyPoint
        n-neg (int): Number of negative samples to generate
    Methods:
        forward: Compute pixel-wise contrastive loss
        forward_eval: Detailed forward pass for logging
    """
    def __init__(self, margin=0.5, alpha=None, beta=None, n_neg=10):
        super(PixelwiseContrastiveLoss,self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.n_neg = n_neg
        
        self._dist = nn.PairwiseDistance()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.margin}, {self.alpha}, {self.beta}, {self.n_neg})'
        
    def __str__(self):
        return f'Min{self.alpha or 0}_Max{self.beta or "Inf"}'

    @staticmethod
    def create_parser(parser):
        parser.add_argument('--margin', default=0.5, help='Target distance between negative feature embeddings.')
        parser.add_argument('--alpha', default=None, type=float, help='Minimum distance from positive KeyPoint')
        parser.add_argument('--beta', default=None, type=float, help='Maximum distance from positive KeyPoint')
        parser.add_argument('--n_neg', default=10, help='Number of negative samples to generate')

    def forward(self, predicted, targetim):
        """ Compute pixel-wise contrastive loss.
        :param features: Vertically stacked feature maps (b, n-dim, h*2, w)
        :param labels: Horizontally stacked correspondence KeyPoints (b, n-kpts, 4) -> (x1, y1, x2, y2)
        :return: Loss
        """

        
        source = predicted
        target = targetim
        #source=torch.tensor(source).float()       
        #source.unsqueeze_(0)
        #source.unsqueeze_(0)
        dim=source.size()
        #print(source.size())
        point_x=random.sample(range(1,dim[3]),30)
        point_y=random.sample(range(1,dim[2]),30)
        #print(len(point_x))
        source_kpts=[]
        target_kpts=[]
        source_kpts1=[]
        target_kpts1=[]
        for i in range(6):          
              for j in range(len(point_x)):
                  coords=(point_x[j],point_y[j])
                  #print(coords)
                  source_kpts1.append(coords)
                  target_kpts1.append(coords)
              #print(len(source_kpts1))
              source_kpts.append(source_kpts1)
              target_kpts.append(target_kpts1)

        source_kpts=torch.tensor(source_kpts).float()
        #source_kpts.unsqueeze_(0)
        #print(source_kpts.size())
        #target=torch.tensor(target).float()
        #target.unsqueeze_(0)
        #target.unsqueeze_(0)
        
        target_kpts=torch.tensor(target_kpts).float()
        #target_kpts.unsqueeze_(0)

        #loss1 = self._positive_loss(source, target, source_kpts, target_kpts)[0]

        #loss2 = self._negative_loss(source, target, source_kpts, target_kpts)[0]

        loss3 = self._positive_loss(source, source, source_kpts, target_kpts)[0]

        loss4 = self._positive_loss(source, target, source_kpts, target_kpts)[0]
 
        if loss4>loss3:

           loss = loss4
        else:
           loss = loss3


        return loss

    def _calc_distance(self, source, target, source_kpts, target_kpts):
        
        source_descriptors = extract_kpt_vectors(source, source_kpts).permute([0, 2, 1]) #ops.
        target_descriptors = extract_kpt_vectors(target, target_kpts).permute([0, 2, 1])
        return self._dist(source_descriptors, target_descriptors)

    def _positive_loss(self, source, target, source_kpts, target_kpts):
        dist = self._calc_distance(source, target, source_kpts, target_kpts)
        loss = (dist**2).mean() / 2
        return loss, dist

    def _negative_loss(self, source, target, source_kpts, target_kpts):
        dsource_kpts, dtarget_kpts = self._generate_negative_like(source, source_kpts, target_kpts)

        dist = self._calc_distance(source, target, dsource_kpts, dtarget_kpts)
        margin_dist = (self.margin - dist).clamp(min=0.0)
        loss = (margin_dist ** 2).mean() / 2
        return loss, dist

    def _generate_negative_like(self, other, source_kpts, target_kpts):
        # Source points remain the same
        source_kpts = source_kpts.repeat([1, self.n_neg, 1])

        # Target points + offset according to method
        target_kpts = target_kpts.repeat([1, self.n_neg, 1])
        target_kpts = self._permute_negatives(target_kpts, other.shape)
        return source_kpts, target_kpts

    def _permute_negatives(self, kpts, shape):
        h, w = shape[-2:]
        # (max(h, w) - low) means that even after getting the remainder points will be further away than low
        low = self.alpha if self.alpha else 0
        high = self.beta if self.beta else (max(h, w) - low)

        # Generate random shift for each KeyPoint
        shift = torch.randint_like(kpts, low=low, high=high)
        shift *= torch.sign(torch.rand_like(shift, dtype=torch.float)-0.5).short()  # Random + or - shift

        # Initial shift to satisfy max distance
        new_kpts = kpts + shift
        new_kpts %= torch.tensor((w, h), dtype=torch.float, device=new_kpts.device)#short

        # Shift to satisfy min distance
        diffs = new_kpts - kpts
        diff_clamp = torch.clamp(diffs, min=-high, max=high)
        new_kpts += (diff_clamp - diffs)

        return new_kpts
