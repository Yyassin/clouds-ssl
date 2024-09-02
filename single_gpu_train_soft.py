import os
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import torch
import timm
from eval_util import knn
from itertools import product

from v2.ChannelDataset import ChannelDataset
from v2.TorchDataset import TorchDataset
from models.mocov3 import MoCoV3, compute_temporal_sim_mat, compute_instance_sim_mat
from training import adjust_learning_rate, get_learning_rate
from common import get_strong_augment, get_weak_augment, get_dataviews

# Main train loop
if __name__ == "__main__":
    channel = "H50"
    batch_size = 2 # 228
    size=384

    weak_augment = get_weak_augment(size=size)
    strong_augment = get_strong_augment(size=size)

    # Iterate over hyperparameters
    for cfgi, (temp, base_lr, epochs) in list(enumerate(product([1., 0.95, 0.8], [1e-4, 5e-5, 1e-5], [1, 3, 5, 10]))):
        save_dir = "./saved_models/moco"
        if os.path.exists(os.path.abspath(f'{save_dir}/{base_lr}_{epochs}_{temp}_{channel}.pt')):
            print(f"skipping [{cfgi+1}]")
            continue

        # Prepare data
        actual_lr = base_lr * batch_size / 256
        dvs = get_dataviews()

        # Get instance-wise features
        sims = torch.load(os.path.abspath("./filtered_data/stats/stats_384.pt"))

        train_dataset = TorchDataset(
            ChannelDataset(dvs), 
            sims=sims,
            weak_aug=weak_augment, 
            strong_aug=strong_augment, 
            use_self=False)
        train_dataset.sims_range.to("cuda:1").to(dtype=torch.bfloat16)
        train_loader = DataLoader(dataset=train_dataset, 
                                batch_size=batch_size,
                                shuffle=True, pin_memory=True, num_workers=8)
        
        # Prepare model
        model = timm.create_model('vit_small_patch16_384.augreg_in21k_ft_in1k', pretrained=True).to("cuda:1")

        model = model.train()
        moco = MoCoV3(encoder=model,
                    is_ViT=True,
                    dim=384, device="cuda:1").to("cuda:1")
        moco = moco.train()
        optimizer = torch.optim.AdamW(moco.parameters())
        
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
                # Adjust learning rate based on step
                # Get learning rate according to the schedule
                new_learning_rate = adjust_learning_rate(step=step_counter, sched_config=sched_config)
                for g in optimizer.param_groups:
                    g['lr'] = new_learning_rate  # Update every parameter group with this new learning rate
                current_lr = get_learning_rate(optimizer=optimizer)  # Confirm that we updated correctly
            
                # Get batch
                batch_1 = batch[0].to("cuda:1").to(dtype=torch.bfloat16)  # (bsz, 3, 384, 384)
                batch_2 = batch[1].to("cuda:1").to(dtype=torch.bfloat16)  # (bsz, 3, 384, 384)
                timestep_indices = batch[2].to("cuda:1").to(dtype=torch.bfloat16) # (bsz, 1)
                flight_indices = batch[3].to("cuda:1").to(dtype=torch.bfloat16) # (bsz, 1)
                sims = batch[4].to("cuda:1").to(dtype=torch.bfloat16) # (bsz, feature_dim)
                
                # Soft similarity matrices
                temporal_sim_mat = compute_temporal_sim_mat(timestep_indices, flight_indices, temp=temp, device="cuda:1")
                instance_sim_mat = compute_instance_sim_mat(sims, temp=temp, device="cuda:1")

                # Loss
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss = moco(
                        batch_1, 
                        batch_2, 
                        use_checkpointing=True,
                        T=temporal_sim_mat,
                        I=instance_sim_mat,
                        lmbda=0.75
                    ) # temporal is weighted
                    
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                train_losses.append(loss.cpu().item())
                step_counter += 1

                # Eval
                if step_counter % 10 == 0:
                    with torch.no_grad():
                        loss_mean = torch.tensor(train_losses).mean().item()
                        mae = knn(moco, is_moco=True, size=size)
                        print(f'[{cfgi+1}] epoch: {epoch}, step: {step_counter}, loss: {loss_mean}, mae: {mae}, current_lr: {current_lr}')
                        train_losses = []
                            
        if not os.path.exists(os.path.abspath(save_dir)):
            os.makedirs(os.path.abspath(save_dir))
        torch.save(obj=moco.state_dict(), f=os.path.abspath(f'{save_dir}/{base_lr}_{epochs}_{temp}_{channel}.pt'))
