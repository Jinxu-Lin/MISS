import numpy as np
import torch
from torch import Tensor
import os 
from tqdm import tqdm
from scipy.stats import spearmanr
from pathlib import Path
from trak.utils import get_matrix_mult

def get_xtx(grads: Tensor) -> Tensor:
    proj_dim = grads.shape[1]
    result = torch.zeros(
        proj_dim, proj_dim, dtype=torch.float16, device='cuda'
    )
    blocks = torch.split(grads, split_size_or_sections=20000, dim=0)

    for block in blocks:
        result += block.T @ block

    return result

def get_xtx_inv(xtx: torch.Tensor, lambda_reg: float) -> torch.Tensor:
    xtx_reg = xtx + lambda_reg * torch.eye(
        xtx.size(0), device=xtx.device, dtype=xtx.dtype
    )
    xtx_inv = torch.linalg.inv(xtx_reg.to(torch.float32))

    xtx_inv /= xtx_inv.abs().mean()

    return xtx_inv.to(torch.float16)

def get_x_xtx_inv(grads: torch.Tensor, xtx_inv: torch.Tensor) -> Tensor:

    blocks = torch.split(grads, split_size_or_sections=20000, dim=0)
    result = torch.empty(
        grads.shape[0], xtx_inv.shape[1], dtype=torch.float16, device='cuda'
    )

    for i, block in enumerate(blocks):
        start = i * 20000
        end = min(grads.shape[0], (i + 1) * 20000)
        result[start:end] = block @ xtx_inv
    
    return result

def get_xtx_inv_x(grads: torch.Tensor, xtx_inv: torch.Tensor) -> torch.Tensor:

    blocks = torch.split(grads, split_size_or_sections=20000, dim=0)
    result = torch.empty(
        grads.shape[0], xtx_inv.shape[1], dtype=torch.float16, device='cuda'
    )

    for i, block in enumerate(blocks):
        start = i * 20000
        end = min(grads.shape[0], (i + 1) * 20000)
        result[start:end] = block @ xtx_inv
    
    return result


def get_preds(scores, mask):
    deleted_mask = ~mask

    # 使用掩码对 scores 进行行加和
    prediction_change = scores[:, deleted_mask].sum(dim=1)

    return prediction_change


def compute_influence_matrix(xtx_inv, xtx_inv_x, mask):

    # Step 3: compute for each subset


    # 获取被删除的样本索引（未被掩盖的样本）
    subset_indices = torch.nonzero(~mask).squeeze()

    # initialize the influence matrix I(S) = (G^T G + λ I)^(-1)
    influence_matrix = xtx_inv.clone()

    # add the contribution of the subset samples to the influence matrix
    for idx in subset_indices:
        single_influence = xtx_inv_x[idx].unsqueeze(1)  # i(x_j)，形状为 (p, 1)
        influence_matrix += single_influence @ single_influence.T

    return influence_matrix


def get_scores(
        features: Tensor, target_grads: Tensor, accumulator: Tensor
    ) -> Tensor:
        train_dim = features.shape[0]
        target_dim = target_grads.shape[0]

        accumulator += (
            get_matrix_mult(features=features, target_grads=target_grads).detach().cpu()
        )

def compute_scores(test_grad, num_train_samples, influence_matrix, device,block_size=5000):
    # Step 3: 分块计算 test_grad @ intermediate
    m, p = test_grad.shape
    scores = torch.empty(m, num_train_samples, device=device)

    for start in range(0, m, block_size):
        end = min(start + block_size, m)
        scores[start:end] = test_grad[start:end] @ influence_matrix

    return scores


def main():

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load train_grad
    train_grad = np.memmap(
        '/home/jinxulin/MISS/Classification/saved/grad/cifar10/seed-0/train-4096.npy', 
        dtype=np.float16, 
        mode='r',
        shape=(50000, 4096)
    )
    train_grad = torch.from_numpy(train_grad).to('cuda')

    # Load test_grad
    test_grad = np.memmap(
        '/home/jinxulin/MISS/Classification/saved/grad/cifar10/seed-0/test-4096.npy', 
        dtype=np.float16, 
        mode='r',
        shape=(10000, 4096)
    )
    test_grad = torch.from_numpy(test_grad).to('cuda')

    # Load train_error
    train_error = np.memmap(
        '/home/jinxulin/MISS/Classification/saved/grad/cifar10/seed-0/error.npy', 
        dtype=np.float16, 
        mode='r',
        shape=(50000, 1)
    )
    train_error = torch.from_numpy(train_error).to('cuda')


    # Load masks and margins
    tmp_path = '/home/jinxulin/MISS/Classification/tmp'
    masks_path = Path(tmp_path).joinpath('mask.npy')
    masks = torch.as_tensor(np.load(masks_path, mmap_mode='r')).float()

    margins_path = Path(tmp_path).joinpath('val_margins.npy')
    margins = torch.as_tensor(np.load(margins_path, mmap_mode='r'))

    # Evaluation!

    # Step 1: Compute (G^T G)
    xtx = get_xtx(train_grad)

    best_rs = 0
    best_lamba = 0
    lamba_list = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    for lamba in lamba_list:

        # Initialize preds_matrix
        filename = os.path.join('/home/jinxulin/MISS/Classification/saved/', f'preds_matrix_{lamba}.npy')
        preds_matrix = np.memmap(filename, 
            dtype=np.float16, 
            mode='w+', 
            shape=(len(masks), test_grad.shape[0])) 

        # Step 2: Compute (G^T G + λ I)^(-1)
        xtx = get_xtx(train_grad)
        xtx_inv = get_xtx_inv(xtx, lamba)
        # Step 3: Compute (G^T G + λ I)^(-1) G
        xtx_inv_x = get_xtx_inv_x(train_grad, xtx_inv)

        # Step 4: Compute residual
        residual = train_grad * train_error

        # Step 5: compute for Influence Matrix
        for i, mask in enumerate(masks):
            print(f'{i} / {len(masks)}')
            mask = mask.bool()
            influence_matrix = compute_influence_matrix(xtx_inv, xtx_inv_x, mask)
        
            # Step 6: compute G_test * I(S) G^T R
            influence_matrix = influence_matrix @ residual.T
            scores = compute_scores(test_grad, train_grad.shape[0], influence_matrix, device)
            pred = get_preds(scores, mask)
            preds_matrix[i] = pred.cpu().numpy()
            preds_matrix.flush()

        
        print('preds_matrix started')

        # Step 7: Evaluate
        val_inds = np.arange(test_grad.shape[0])
        rs, ps = [], []

        for ind, j in tqdm(enumerate(val_inds), desc="Evaluating Rank Correlation"):
            preds = preds_matrix[:, j].numpy()
            r, p = spearmanr(preds, margins[:, j])
            rs.append(r)
            ps.append(p)
        
        rs, ps = np.array(rs), np.array(ps)
        print(f'Correlation: {rs.mean():.3f} (avg p value {ps.mean():.6f})')

        if rs.mean() > best_rs:
            best_rs = rs.mean()
            best_lamba = lamba

    print(f'Best lambda: {best_lamba}, Best rs: {best_rs}')
    # save the best lambda
    with open(os.path.join('/home/jinxulin/MISS/Classification/saved/', f'best_lambda.txt'), 'w') as f:
        f.write(f'{best_lamba}/{best_rs}')



if __name__ == '__main__':
    main()