import os
from torch.utils.data import DataLoader
import torch
import timm
from eval_util import knn
from itertools import product

from v2.ChannelDataset import ChannelDataset
from v2.TorchDataset import TorchDataset
from models.mocov3 import MoCoV3
from training import adjust_learning_rate, get_learning_rate
from common import get_strong_augment, get_weak_augment, get_dataviews

# Main train loop
if __name__ == "__main__":
    channel = "H50"
    batch_size = 2 # 228
    size=384
    
    weak_augment = get_weak_augment(size)
    strong_augment = get_strong_augment(size)
    configs = [
        {
            "weak_augment": weak_augment,
            "strong_augment": strong_augment,
            "use_self": False
        }
    ]

    # Iterate over augment configs
    for i, config in enumerate(configs):
        weak_augment = config["weak_augment"]
        strong_augment = config["strong_augment"]
        use_self = config["use_self"]

        save_dir = "./saved_models/moco"

        # Iterate over hyperparameters
        for cfgi, (base_lr, epochs) in enumerate(product([1e-4, 5e-5, 1e-5], [1, 3, 5, 10])):
            torch.cuda.empty_cache()
            if os.path.exists(os.path.abspath(f'{save_dir}/{base_lr}_{epochs}_{channel}_conf={i}.pt')):
                print(f"skipping [{cfgi+1}]", i)
                continue

            # Prepare data
            actual_lr = base_lr * batch_size / 256
            dvs = get_dataviews()
            train_dataset = TorchDataset(ChannelDataset(dvs), weak_aug=weak_augment, strong_aug=strong_augment, use_self=use_self)
            train_loader = DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size,
                                    shuffle=True, pin_memory=True, num_workers=8)
            
            # Prepare hard MoCoV3 model
            model = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=True, img_size=size).to("cuda:0")

            model = model.train()
            moco = MoCoV3(encoder=model,
                        is_ViT=True,
                        dim=384, device="cuda:0").to("cuda:0")
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
                    batch_1 = batch[0].to("cuda:0", non_blocking=True).to(dtype=torch.bfloat16)  # (bsz, 3, 384, 384)
                    batch_2 = batch[1].to("cuda:0", non_blocking=True).to(dtype=torch.bfloat16)  # (bsz, 3, 384, 384)
                    
                    # Loss
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss = moco(batch_1, batch_2, use_checkpointing=True)
                        
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    train_losses.append(loss.cpu().item())
                    step_counter += 1

                    # Eval
                    if step_counter % 10 == 0:
                        loss_mean = torch.tensor(train_losses).mean().item()
                        mae = knn(moco, is_moco=True, size=size)
                        print(f'[{cfgi+1}] epoch: {epoch}, step: {step_counter}, loss: {loss_mean}, mae: {mae}, current_lr: {current_lr}')
                        train_losses = []
                        
            # Save
            if not os.path.exists(os.path.abspath(save_dir)):
                os.makedirs(save_dir)
            torch.save(obj=moco.state_dict(), f=os.path.abspath(f'{save_dir}/{base_lr}_{epochs}_{batch_size}_{channel}_conf={i}.pt'))
