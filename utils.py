import torch
import numpy as np
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt
from torchvision import transforms

from modules import NPZDataset

@torch.no_grad()
def refine_latent_offline(actor, x0, steps=8, step_scale=0.1):
    x = x0.clone()
    for _ in range(steps):
        mu, _ = actor(x)
        x = x + step_scale * torch.tanh(mu)
    return x

def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_train_test(
    path,
    device="cpu",
    batch_size=32,
    resize=256,
    test_size=0.6,
    ratio_0=20/85,
    seed=42
):
    data = np.load(path, mmap_mode="r")
    labels = data["labels"]
    
    idx_1 = np.where(labels == 1)[0]
    idx_0 = np.where(labels == 0)[0]

    n0 = int(len(idx_0) * ratio_0)
    np.random.seed(seed)
    idx_0 = np.random.choice(idx_0, n0, replace=False)

    indices = np.concatenate([idx_1, idx_0])
    np.random.shuffle(indices)

    
    if test_size > 0:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels[indices],
            random_state=seed
        )
    else:
        train_idx = indices
        test_idx = []

    train_dataset = NPZDataset(path, train_idx, resize, device)
    test_dataset  = NPZDataset(path, test_idx,  resize, device) if len(test_idx) > 0 else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size) if test_dataset else None

    return train_dataset, train_loader, test_dataset, test_loader

def count_classes(dataset):
    count0 = 0
    count1 = 0

    for i in range(len(dataset)):
        lbl = dataset[i][1]    
        lbl = int(lbl.item())  

        if lbl == 0:
            count0 += 1
        else:
            count1 += 1
    return count0, count1

def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch).squeeze()
            all_preds.append(y_pred.cpu())
            all_labels.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()


    binary_pred = (preds > threshold).astype(int)

    print("Classification Report:")
    print(classification_report(labels, binary_pred, labels=[0,1], target_names=["Class 0","Class 1"]))

    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, preds)
        print(f"AUC-ROC: {auc:.4f}")
    else:
        print("AUC-ROC: Undefined (only one class present)")

def train_cnn_ae(model, loader, opt, device):
    model.train()
    total = 0
    for x,y in loader:
        x = x.to(device)
        y = y.unsqueeze(1).to(device)
        recon = model.decode(model.encode(x,y), y)
        loss = torch.mean(torch.abs(x - recon))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

def train_detector(model, loader, opt, criterion, device):
    model.train()
    total = 0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        out = model(x)
        loss = criterion(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

def One_Step_To_Feasible_Action(generator, detector, x_orig, device,
                                previously_generated=None, alpha=1.0,
                                lambda_div=0.1, lr=0.001, steps=50):
    generator.eval()
    detector.eval()
    if previously_generated is None:
        previously_generated = []

    x_orig = x_orig.to(device).unsqueeze(0)
    y_class1 = torch.full((1,1),0.8,device=device)

    with torch.no_grad():
        z = generator.encode(x_orig, y_class1)
    z = z.clone().detach().requires_grad_(True)

    optimizer_z = optim.Adam([z], lr=lr)

    for step in range(steps):
        optimizer_z.zero_grad()
        x_syn = generator.decode(z, y_class1)
        prob_class1 = detector(x_syn)

        if len(previously_generated) > 0:
            old = torch.stack(previously_generated).to(device)
            diff = x_syn.unsqueeze(1)-old.unsqueeze(0)
            dist_min = diff.norm(p=2,dim=(2,3,4)).min(dim=1).values
            diversity_term = torch.exp(-alpha*dist_min).mean()
        else:
            diversity_term = torch.tensor(0.0, device=device)

        reward = prob_class1.mean() + lambda_div*diversity_term
        reward.backward()
        optimizer_z.step()

    with torch.no_grad():
        x_adv = generator.decode(z, y_class1).detach().cpu()

    return x_adv

def Gen_with_PPO(
    actor,
    generator,
    x_orig,
    device,
    episodes,
    steps=8
):
    actor.eval()
    generator.eval()

    conf = 0.8 + 0.1 / (1 + math.exp(-0.1 * (episodes - 10)))
    y_class1 = torch.full(
        (1, 1),
        random.uniform(max(conf - 0.02, 0.8), conf),
        device=device
    )

    with torch.no_grad():
        z0 = generator.encode(
            x_orig.unsqueeze(0).to(device),
            y_class1
        )

    z_adv = refine_latent_offline(
        actor,
        z0.squeeze(0),
        steps=steps
    )

    if not check_valid_variant(z_adv, z0.squeeze(0)):
        return None

    with torch.no_grad():
        x_synthetic = generator.decode(
            z_adv.unsqueeze(0),
            y_class1
        )

    return x_synthetic

def check_valid_variant(x, x_prime, threshold=0.3):
    if x.dim() > 1:
        x_flat = x.view(x.size(0), -1)
        x_prime_flat = x_prime.view(x_prime.size(0), -1)
        cos_sim = F.cosine_similarity(x_flat, x_prime_flat, dim=1)
    else:
        cos_sim = F.cosine_similarity(x, x_prime, dim=0)
    
    cos_dist = 1 - cos_sim
    
    is_valid = (cos_dist < threshold).all()
    
    return is_valid.item()

def load_z_space(generator, loader, device):
    generator.eval()

    all_z = []
    all_y = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.unsqueeze(1).to(device)

            z = generator.encode(x, y)
            
            all_z.append(z.cpu())
            all_y.append(y.cpu())

    all_z = torch.cat(all_z, dim=0)
    all_y = torch.cat(all_y, dim=0)

    return all_z, all_y

def preview_images(loader, model=None, device='cpu', num_images=5):
    for images, labels in loader:
        images = images[:num_images]
        labels = labels[:num_images]

        if model is None:
            imgs_np = images.cpu().numpy()
            labels_np = labels.cpu().numpy()
            imgs_np = np.clip(imgs_np, 0, 1)
            n = imgs_np.shape[0]
            fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
            if n == 1:
                axes = [axes]
            for j in range(n):
                label = int(labels_np[j])
                axes[j].imshow(imgs_np[j].transpose(1, 2, 0))
                axes[j].set_title(f'[{label}] Image {j}')
                axes[j].axis('off')
            plt.tight_layout()
            plt.show()
        else:
            model.eval()
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)
                reconstructed, _, _ = model(images, labels.unsqueeze(1))

            orig = images.cpu().numpy()
            recon = reconstructed.cpu().numpy()
            labels_np = labels.cpu().numpy()
            orig = np.clip(orig, 0, 1)
            recon = np.clip(recon, 0, 1)

            n = orig.shape[0]
            fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
            if n == 1:
                axes = axes.reshape(2, 1)

            for j in range(n):
                label = int(labels_np[j])
                axes[0, j].imshow(orig[j].transpose(1, 2, 0))
                axes[0, j].set_title(f'[{label}] Org-{j}')
                axes[0, j].axis('off')

                axes[1, j].imshow(recon[j].transpose(1, 2, 0))
                axes[1, j].set_title(f'[{label}] Recon-{j}')
                axes[1, j].axis('off')
            plt.tight_layout()
            plt.show()

        break

def load_data(data, batch_num=64, is_shuffle=True):
    images = data['images']
    labels = data['labels']

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor() 
    ])

    X = torch.stack([transform(img) for img in images])

    y = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_num, shuffle=is_shuffle)

    return loader



