import torch
import torch.nn as nn
from x_transformers import Decoder
import numpy as np
import copy
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

class Predictor(nn.Module):
    """The predictor network for JEPA. It takes the context encoding and 
    the target masks as input and predicts the target blocks.
    """
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        self.predictor = Decoder(dim = embed_dim, depth = depth, heads = num_heads)

    def forward(self, context_encoding, target_masks):
        """ Concatenate context patch encodings, and the target masks. Pass
        them through the predictor network to get the predictions for the target
        blocks.
        """
        x = torch.cat((context_encoding, target_masks), dim = 1)
        x = self.predictor(x)
        # return last `len(target_masks)` tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1]:, :]

class JEPA(nn.Module):
    """ The Joint Encoding and Prediction Augmentation (JEPA) model."""
    def __init__(self, enc, M, device="cuda:1"):
        super().__init__()
        self.device = device

        # The student encoder is the encoder being trained, it generates the context encoding
        self.student_encoder = enc
        # The teacher encoder is the encoder used to generate the target blocks. We'll update
        # this via a momentum update.
        self.enc = copy.deepcopy(self.student_encoder).to(self.device)

        # The predictor network takes the context encoding and the target masks as input
        # and predicts the target blocks.
        self.predictor = Predictor(self.enc.embed_dim, num_heads=8, depth=8).to(self.device)
        
        # The number of target blocks
        self.M = M
        # Mask token to learn.
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.enc.embed_dim)).to(self.device)

        self.patch_dim = int(np.sqrt(self.enc.patch_embed.num_patches))
        self.patch_dim = (self.patch_dim, self.patch_dim)
        nn.init.trunc_normal_(self.mask_token, 0.02)
        
        self.ctr = torch.nn.MSELoss()
        self.criterion_contrast = torch.nn.CrossEntropyLoss()

        self.temp = 0.07
        # fq has an extra prediction head
        self.predictor_contrast = nn.Sequential(
            nn.Linear(self.enc.embed_dim, self.enc.embed_dim),
            nn.BatchNorm1d(self.enc.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.enc.embed_dim, self.enc.embed_dim),
        ).to(device)


    @torch.no_grad()
    def get_target_block(self, x, patch_dim, aspect_ratio, scale, M):
        """Gets the target blocks from the input image x. A target block is a block of patches
        that the model will predict.
        
        Args:
            x: The input image.
            patch_dim: Image dimensions in patches.
            aspect_ratio: Aspect ratio of the target block.
            scale: Scale of the target block.
            M: Number of target blocks.
        """

        #get the patch dimensions
        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the patches in the target block
        target_block = torch.zeros((M, x.shape[0], block_h*block_w, x.shape[2]))
        target_patches = []
        all_patches = []
        for z in range(M):
            #get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = []
            #get the patches in the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)
                    
            #get the target block
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        return target_block.to(self.device), target_patches, all_patches
    
    @torch.no_grad()
    def update_momentum(self, m):
        """Update the teacher encoder with the student encoder using momentum update.
        
        Args:
            m: The momentum parameter.
        """
        student_model = self.student_encoder.eval()
        teacher_model = self.enc.eval()
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)

        self.student_encoder.train()
        self.enc.train()

    
    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        """Gets the context block from the input image x. A context block is a block of patches
        that the model will use to predict the target blocks.
        
        Args:
            x: The input image. Shape: (B, N, E)
            patch_dim: Image dimensions in patches. Shape: (H, W)
            aspect_ratio: Aspect ratio of the context block.
            scale: Scale of the context block.
            target_patches: The patches in the target block; these patches will be removed from the context block.

        Returns:
            The context block.
        """

        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        #get the patches in the context_block
        patches = []
        for i in range(block_h):
            for j in range(block_w):
                if start_patch + i * patch_w + j not in target_patches: #remove the target patches
                    patches.append(start_patch + i * patch_w + j)

        return x[:, patches, :]
    
    def ctr_contrast(self, q, k):
        """Compute the contrastive loss between the query and key features.
        
        Args:
            q: The query features.
            k: The key features.
        """
        logits = torch.mm(q, k.t()) # (N, N)
        N = q.size(0)
        labels = range(N) # positives in diagonal
        labels = torch.LongTensor(labels).to(self.device)
        loss = self.criterion_contrast(logits / self.temp, labels)
        return 2 * self.temp * loss
    
    def soft_ctr(self, q, k, S):
        """Compute the soft contrastive loss between the query and key features.

        Args:
            q: The query features.
            k: The key features.
            S: The relative similarity matrix.
        """
        logits = torch.mm(q, k.t()) # (N, N)
        soft_labels = S
        # Each logit row is log(exp / sumexp)
        log_softmax = F.log_softmax(logits / self.temp, dim=1)
        # Sum along the rows (CE)
        log_softmax = (log_softmax * soft_labels).sum(dim=1)
        # Then mean across rows
        loss = -log_softmax.mean()
        return 2 * self.temp * loss

    def contrast(self, x1, x2, use_checkpointing=True, T=None, I=None, lmbda=None):
        """Compute the contrastive loss between the query and key features.
        
        Args:
            x1: The first set of views:
            x2: The second set of views.
            use_checkpointing: Whether to use checkpointing.
            T: The relative similarity matrix for the temporal objective.
            I: The relative similarity matrix for the instance-wise objective.
            lmbda: The weight for the temporal objective.
        """
        torch.cuda.empty_cache()
        if use_checkpointing:
            q1 =  checkpoint(self.student_encoder.forward_features, x1, use_reentrant=False)
            q2 =  checkpoint(self.student_encoder.forward_features, x2, use_reentrant=False)
        else:
            q1 = self.student_encoder.forward_features(x1)
            q2 = self.student_encoder.forward_features(x2)

        q1 = q1[:, 1:, :].mean(dim=1)  # global average pool for ViTs
        q2 = q2[:, 1:, :].mean(dim=1)  # global average 

        # fq has additional predictor
        if use_checkpointing:
            q1 =  checkpoint(self.predictor_contrast, q1, use_reentrant=False)
            q2 =  checkpoint(self.predictor_contrast, q2, use_reentrant=False)
        else:
            q1 = self.predictor_contrast(q1)
            q2 = self.predictor_contrast(q2)

        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():
            k1 = self.enc.forward_features(x1)
            k2 = self.enc.forward_features(x2)
            
            k1 = k1[:, 1:, :].mean(dim=1)  # global average pool for ViTs
            k2 = k2[:, 1:, :].mean(dim=1)  # global average pool for ViTs

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
    
    def forward(self, x_, m, target_aspect_ratio=1, target_scale=1, context_aspect_ratio=1, context_scale=1):
        """Forward pass of the JEPA model.
        
        Args:
            x_: The input image.
            m: The momentum parameter.
            target_aspect_ratio: The aspect ratio of the target block.
            target_scale: The scale of the target block.
            context_aspect_ratio: The aspect ratio of the context block.
            context_scale: The scale of the context block.
        """
        # x_: (64, 3, 224, 224)
        torch.cuda.empty_cache()

        with torch.no_grad():
            self.update_momentum(m)
            x = self.enc.forward_features(x_)
            x = x[:, 1:, :] # remove cls

        b, n, e = x.shape
        target_blocks, target_patches, all_patches = \
            self.get_target_block(x, self.patch_dim, target_aspect_ratio, target_scale, self.M)
        
        m, b, n, e = target_blocks.shape
        x_patches = self.student_encoder.patch_embed(x_)
        x_patches = self.student_encoder._pos_embed(x_patches)
        x_patches = self.student_encoder.norm_pre(x_patches)
        # Remove cls
        x_patches = x_patches[:, 1:, :]

        context_block = self.get_context_block(x_patches, self.patch_dim, context_aspect_ratio, context_scale, all_patches)
        context_encoding = self.student_encoder.blocks(context_block)
        context_encoding = self.student_encoder.norm(context_encoding)

        prediction_blocks = torch.zeros((m, b, n, e)).to(self.device)
        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1)
            target_pos_embedding = self.student_encoder.pos_embed[:, target_patches[i], :]
            target_masks = target_masks + target_pos_embedding
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        loss = self.ctr(prediction_blocks, target_blocks)
        return loss