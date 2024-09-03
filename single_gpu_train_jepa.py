import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import timm
from eval_util import knn
from itertools import product
import gc

from v2.ChannelDataset import ChannelDataset
from v2.TorchDataset import TorchDataset
from models.remoco import JEPA
from models.mocov3 import compute_temporal_sim_mat, compute_instance_sim_mat
from training import adjust_learning_rate, get_learning_rate
from common import get_strong_augment, get_weak_augment, get_dataviews

# Main train loop
index = 0
gpu_id = f"cuda:{index}"
if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Setup hyperparameters
    channel = "H50"
    batch_size = 2 # 28

    m=0.996 #momentum
    m_start_end = (.996, 1.)

    target_aspect_ratio = (0.75,1.5)
    target_scale = (0.15, .2)
    context_aspect_ratio = 1
    context_scale = (0.85,1.0)

    size = 384
    weak_augment = get_weak_augment(size)
    strong_augment = get_strong_augment(size)

    # Get instance-wise features
    sims_ = torch.load(os.path.abspath("./filtered_data/stats/stats_384.pt"))

    # Iterate over hyperparameters
    for cfgi, (temp, base_lr, epochs) in enumerate(product([1], [5e-5], [25])):
        torch.cuda.empty_cache()
        save_dir = "./saved_models/moco"
        if os.path.exists(os.path.abspath(f'{save_dir}/{base_lr}_{epochs}_{temp}_{channel}.pt')):
            print(f"skipping [{cfgi+1}]")
            continue

        # Prepare data
        actual_lr = base_lr * batch_size / 256
        dvs = get_dataviews()
        train_dataset = TorchDataset(ChannelDataset(dvs), sims=sims_, weak_aug=weak_augment, strong_aug=strong_augment, use_self=False)
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
        
        # Prepare model
        model = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True).to(gpu_id)

        model = model.train()
        jepa = JEPA(enc=model, M=4, device=gpu_id).to(gpu_id)
        jepa.train()
        optimizer = torch.optim.AdamW(jepa.parameters())
        
        # Number of steps for entire training run
        total_steps = int(epochs * len(train_loader))
        warmup_steps = int(total_steps * 0.1)
        step_counter = 0
        sched_config = {
            'max_lr': actual_lr,
            'min_lr': 0,
            'total_steps': total_steps,
            'warmup_steps': warmup_steps,
        }
        print(len(train_loader), channel)
        
        # Train
        train_losses = []
        for epoch in range(epochs):
            for batch_idx, batch in enumerate(train_loader):
                try:
                    torch.cuda.empty_cache()
                    target_aspect_ratio_ = np.random.uniform(target_aspect_ratio[0], target_aspect_ratio[1])
                    target_scale_ = np.random.uniform(target_scale[0], target_scale[1])
                    context_scale_ = np.random.uniform(context_scale[0], context_scale[1])

                    # Adjust learning rate based on step
                    # Get learning rate according to the schedule
                    new_learning_rate = adjust_learning_rate(step=step_counter, sched_config=sched_config)
                    for g in optimizer.param_groups:
                        g['lr'] = new_learning_rate  # Update every parameter group with this new learning rate
                    current_lr = get_learning_rate(optimizer=optimizer)  # Confirm that we updated correctly
                
                    # Get batch
                    batch_1 = batch[0].to(gpu_id).to(dtype=torch.bfloat16)  # (bsz, 3, 384, 384)
                    batch_2 = batch[1].to(gpu_id).to(dtype=torch.bfloat16)  # (bsz, 3, 384, 384)
                    timestep_indices = batch[2].to(gpu_id).to(dtype=torch.bfloat16) # (bsz, 1)
                    flight_indices = batch[3].to(gpu_id).to(dtype=torch.bfloat16) # (bsz, 1)
                    sims = batch[4].to(gpu_id).to(dtype=torch.bfloat16) # (bsz, feature_dim)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss_jepa = jepa(
                            batch_2, 
                            m,
                            target_aspect_ratio=target_aspect_ratio_, 
                            target_scale=target_scale_, 
                            context_aspect_ratio=context_aspect_ratio, 
                            context_scale=context_scale_)
                        loss_jepa /= 6

                        torch.cuda.empty_cache()

                        # Soft similarity matrices
                        temporal_sim_mat = compute_temporal_sim_mat(timestep_indices, flight_indices, temp=temp, device=gpu_id)
                        instance_sim_mat = compute_instance_sim_mat(sims, temp=temp, device=gpu_id)

                        # Loss
                        loss_moco = jepa.contrast(
                            batch_1, batch_2, use_checkpointing=True, T=temporal_sim_mat, I=instance_sim_mat, lmbda=0.75
                        )  
                        loss = (loss_jepa + loss_moco) / 2   

                    print("jepa", loss_jepa.item(), "moco", loss_moco.item())
                    del loss_jepa
                    del loss_moco
                    gc.collect()
                    torch.cuda.empty_cache()
                        
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                except Exception:
                    print("Memory error, continuing")
                    loss_jepa, loss_moco, loss, temporal_sim_mat, instance_sim_mat  = 0, 0, 0, 0, 0
                    optimizer.zero_grad()
                    del loss_jepa
                    del loss_moco
                    del loss
                    del temporal_sim_mat
                    del instance_sim_mat
                    del batch_1
                    del batch_2
                    del timestep_indices
                    del flight_indices
                    del sims
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue 

                # Linearly increase momentum
                m += ((m_start_end[1] - m_start_end[0]) / (epochs * len(train_loader)))
                
                train_losses.append(loss.cpu().item())
                step_counter += 1

                # Eval
                if step_counter % 10 == 0:
                    with torch.no_grad():
                        loss_mean = torch.tensor(train_losses).mean().item()
                        mae = knn(jepa)
                        print(f'[{cfgi + 1}] epoch: {epoch}, step: {step_counter}, loss: {loss_mean}, mae: {mae}, current_lr: {current_lr}')
                        train_losses = []            
        
        if not os.path.exists(os.path.abspath(save_dir)):
            os.makedirs(os.path.abspath(save_dir))
        torch.save(obj=jepa.state_dict(), f=f'{save_dir}/{base_lr}_{epochs}_{temp}_{channel}.pt')

        model = None
        del model
        jepa = None
        del jepa
        optimizer = None
        del optimizer
