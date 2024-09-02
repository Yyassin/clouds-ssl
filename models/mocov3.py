import torch
import torch.nn as nn
import copy
import einops
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

def temporal_similarity(t1, t2, temp, range):
    """Compute the temporal-wise relative similarity for an instance
    pair given their timestamps. The similarity is computed as a
    function of the difference between the timestamps.

    Args:
        t1: Timestamp of the first instance.
        t2: Timestamp of the pair.
        temp: Soft temperature (tau_T) parameter for the sigmoid function.
        range: Range of the relative similarity, i.e., the output is in [0, range].

    Returns:
        The relative similarity between the two instances in [0, range].
    """
    return 2 * range * torch.sigmoid(-temp * torch.abs(t1 - t2))

def instance_similarity(x1, x2, temp, range):
    """Compute the instance-wise relative similarity for an instance pair
    given their feature vectors. The similarity is computed as a function
    of the difference between features.

    Args: 
        x1: Feature vector for the first instance.
        x2: Feature vector of the pair.
        temp: Soft temperature (tau_I) parameter for the sigmoid function.
        range: Range of the relative similarity, i.e., the output is in [0, range].

    Returns:
        The relative similarity between the two instance in [0, range].
    """
    diff = torch.abs(x1 - x2) 
    diff_norm = torch.linalg.vector_norm(diff, ord=2, dim=-1)
    return 2 * range * torch.sigmoid(-temp * diff_norm)

def compute_temporal_sim_mat(timestamps, flights, temp=0.9, range_=1, device="cuda:0"):
    """Compute the temporal-wise similarity matrix for a batch of instances.

    Args:
        timestamps: Tensor of shape (b, 1) with the timestamps of the instances.
        flights: Tensor of shape (b, 1) with the flight numbers of the instances.
        temp: Soft temperature (tau_T) parameter for the sigmoid function.
        range: Range of the relative similarity, i.e., the output is in [0, range].
        device: The device to place intermediates on.

    Returns:
        The temporal-wise similarity matrix.
    """

    with torch.no_grad():
        # Initialize a (b x b) 0 block matrix.
        b = timestamps.shape[0]
        S = torch.full((b, b), 0).to(device).to(dtype=torch.bfloat16)

        # For each instance (each row in the mat), we find the pairs (column indices)
        # with the same flight and calculate the temporal-wise relative similarities
        # for these entries.
        for row in range(b):
            timeR = timestamps[row]
            flightR = flights[row]
            
            # Compare flights to see which cell similarities should be computed. The rest are 0
            same_flight_mask = flights == flightR
            
            # Compute weights only for the same flight entries
            timesC = timestamps[same_flight_mask]
            S[row, same_flight_mask] = temporal_similarity(timeR, timesC, temp, range_)
    return S

def compute_instance_sim_mat(sims, temp=0.9, range_=1, device="cuda:0"):
    """Compute the instance-wise similarity matrix for a batch of instances.

    Args:
        sims: Tensor of shape (b, feature-dim) with the features of the instances.
        temp: Soft temperature (tau_I) parameter for the sigmoid function.
        range: Range of the relative similarity, i.e., the output is in [0, range].
        device: The device to place intermediates on.

    Returns:
        The instance-wise similarity matrix.
    """
    with torch.no_grad():
        # Initialize a (b x b) 0 block matrix.
        b = sims.shape[0]
        S = torch.full((b, b), 0).to(device).to(dtype=torch.bfloat16)

        # For each instance (each row in the mat), we compute the
        # relative similarity with all other instance pairs.
        for row in range(b):
            # Unsqueeze to broadcast the difference along sequence dim
            simsR = sims[row].to(device).to(dtype=torch.bfloat16).unsqueeze(0)
            S[row] = instance_similarity(simsR, sims, temp, range_)
    return S


    
class MoCoV3(nn.Module):
    """Standard MoCoV3 module from
    """
    def __init__(self, 
                 encoder,
                 is_ViT,
                 dim=768,
                 queue_size=None,
                 momentum=0.999, 
                 temp=0.07,
                 use_project=False,
                 device="cuda:0"):
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.is_ViT = is_ViT
        self.momentum = momentum
        self.temp = temp
        self.use_project = use_project
        self.device = device
        
        # Clone the encoder to make a twin that will predict keys. 
        # We update the twin with momentum
        self.encoder_k = copy.deepcopy(encoder)
        self.encoder_q = encoder

        # fq has an extra prediction head
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        ).to(device)

        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.requires_grad = False       # encoder_k learns via momentum

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder, doesn't 
        include the prediction head.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), 
                                    self.encoder_k.parameters()):
            param_k.data = (param_k.data * self.momentum + 
                            param_q.data * (1. - self.momentum))
            
    def ctr(self, q, k):
        logits = torch.mm(q, k.t()) # (N, N)
        N = q.size(0)
        labels = range(N) # positives in diagonal
        labels = torch.LongTensor(labels).to(self.device)
        loss = self.criterion(logits / self.temp, labels)
        return 2 * self.temp * loss
    
    def soft_ctr(self, q, k, S):
        logits = torch.mm(q, k.t()) # (N, N)
        N = q.size(0)
        soft_labels = S

        # Each logit row is log(exp / sumexp)
        log_softmax = F.log_softmax(logits / self.temp, dim=1)
        # Sum along the rows (CE)
        log_softmax = (log_softmax * soft_labels).sum(dim=1)
        # Then mean across rows
        loss = -log_softmax.mean()
        return 2 * self.temp * loss
    
    def forward(self, x1, x2, use_checkpointing, T=None, I=None, lmbda=None):
        '''
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            loss
        '''
        if use_checkpointing:
            q1 =  checkpoint(self.encoder_q.forward_features, x1, use_reentrant=False)
            q2 =  checkpoint(self.encoder_q.forward_features, x2, use_reentrant=False)
        else:
            q1 = self.encoder_q.forward_features(x1)
            q2 = self.encoder_q.forward_features(x2)
        
        if self.is_ViT:
            q1 = q1[:, 1:, :].mean(dim=1)  # global average pool for ViTs
            q2 = q2[:, 1:, :].mean(dim=1)  # global average pool for ViTs
        else:
            q1 = einops.rearrange(q1, 'b d h w -> b d (h w)').mean(dim=-1)  # global average pool for ConvNext
            q2 = einops.rearrange(q2, 'b d h w -> b d (h w)').mean(dim=-1)  # global average pool for ConvNext

        # fq has additional predictor
        if use_checkpointing:
            q1 =  checkpoint(self.predictor, q1, use_reentrant=False)
            q2 =  checkpoint(self.predictor, q2, use_reentrant=False)
        else:
            q1 = self.predictor(q1)
            q2 = self.predictor(q2)

        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            k1 = self.encoder_k.forward_features(x1)
            k2 = self.encoder_k.forward_features(x2)
            
            if self.is_ViT:
                k1 = k1[:, 1:, :].mean(dim=1)  # global average pool for ViTs
                k2 = k2[:, 1:, :].mean(dim=1)  # global average pool for ViTs
            else:
                k1 = einops.rearrange(k1, 'b d h w -> b d (h w)').mean(dim=-1)  # global average pool for ConvNext
                k2 = einops.rearrange(k2, 'b d h w -> b d (h w)').mean(dim=-1)  # global average pool for ConvNext
            
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

        # Compute soft contrastive objectives if the relative similarity matrices are provided.
        if T is not None:
            loss_temporal = self.soft_ctr(q1, k2, T) + self.soft_ctr(q2, k1, T)
        if I is not None:
            loss_instance = self.soft_ctr(q1, k2, I) + self.soft_ctr(q2, k1, I)
            loss_instance /= 10
        
        # This function is only used with BOTH temporal and instance-wise objectives.
        if T is not None and I is not None:
            loss = loss_temporal * lmbda + loss_instance * (1 - lmbda)

        # Otherwise, use the regular (hard) MocoV3 loss
        if T is None and I is None:
            loss = self.ctr(q1, k2) + self.ctr(q2, k1)
        return loss
