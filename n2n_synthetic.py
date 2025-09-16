 

"""
Noise2Noise Reconstruction for Photoluminescence Data

This script demonstrates the application of Noise2Noise (N2N) reconstruction
techniques to artificial photoluminescence datasets. It compares different 
reconstruction methods including:
- Pointwise linear fitting
- Regularized reconstruction (optional)
- N2N neural network reconstruction

The script loads artificial data cubes, adds noise, and performs reconstruction
to estimate photoluminescence parameters (amplitude and lifetime).
"""


#Enter device here, 'cuda' for GPU, and 'cpu' for CPU
device = 'cuda'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import random
from utils_N2N import *
import os
from pathlib import Path

# CONFIGURE MATPLOTLIB PLOTS
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))


 
print("Starting N2N reconstruction pipeline")

# ============================================
# SETUP AND DATA LOADING
# ============================================
    
# Setup device
device, dtype = setup_device(prefer_cuda=True)
    
# Load dataset
dataset = 2
try:
    mat_data = load_artificial_dataset(dataset )
except (FileNotFoundError, RuntimeError) as e:
    print(f"Failed to load dataset: {e}")
    
# Define time indices to use
time_indices = range(151, 210)

# BATCH SIZE OF THE CUBE
batch_size = 15    

# MAGNITUDE OF THE NOISE
noise_magnitude = 1e-2

# Prepare noisy data
noisy_data_torch = prepare_noisy_data(
    mat_data['cube'], time_indices, batch_size, noise_magnitude, dtype
)
    
batch_size, channels, height, width = noisy_data_torch.shape
print(f"Data shape: {noisy_data_torch.shape}")
    
# Prepare time points
time_ns = mat_data['timeNs']
time_torch = torch.from_numpy(time_ns[0, time_indices]).type(dtype).detach()
    
# Choose random input index
input_index = random.randint(0, batch_size - 1)
print(f"Using batch index {input_index} as input")
    
# Prepare ground truth
a_gt = np.log(mat_data["A1Map"])
b_gt = -1/mat_data["lifetimeMap"]
ground_truth = np.stack((a_gt,b_gt))
    
a_gt_torch = np_to_torch(a_gt).type(dtype).detach()
b_gt_torch = np_to_torch(b_gt).type(dtype).detach()
    
# Create mask (empty for now, can be customized for specific regions)
mask = np.zeros_like(a_gt, dtype=bool)

# Create output directory
base_name = f"dataset_{dataset}_batch_{batch_size}_noise_{noise_magnitude}"
counter = 1
while True:
    output_dir = Path(f"{base_name}_{counter}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        break
    counter += 1

print(f"Output directory created: {output_dir}")


# ============================================
# VISUALIZATION OF GROUND TRUTH
# ============================================
    
print("Plotting ground truth")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_with_colorbar(a_gt, mask, axes[0])
axes[0].set_title("$\\log(\\delta)$ - Ground Truth")
plot_with_colorbar(b_gt, mask, axes[1])
axes[1].set_title("$\\tau$ - Ground Truth")
plt.tight_layout()
# plt.savefig(output_dir / "ground_truth.png", dpi=300, bbox_inches='tight')
# plt.show()



# ============================================
# POINTWISE RECONSTRUCTION
# ============================================
    
print("Computing pointwise reconstruction")
input_data_np = noisy_data_torch.detach().cpu().numpy()
a_pw, b_pw = pointwise_reconstruction(
        input_data_np, time_ns[0, time_indices]
    )
    
# Plot pointwise results
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_with_colorbar(a_pw, mask, axes[0])
axes[0].set_title("$\\log(\\delta)$ - Pointwise")
plot_with_colorbar(-1/b_pw, mask, axes[1])
axes[1].set_title("$\\tau$ - Pointwise")
plt.tight_layout()
# plt.savefig(output_dir / "pointwise_reconstruction.png", dpi=300, bbox_inches='tight')
# plt.show()
    
pointwise_results = np.stack((a_pw, b_pw))

    
# ============================================
# REGULARIZED RECONSTRUCTION (if available)
# ============================================
    
regularized_results = load_regularized_solution(dataset, noise_magnitude, reg_id = 2)[0]
    
if regularized_results is not None:
        print("Regularized solution loaded")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_with_colorbar(regularized_results[0], mask, axes[0])
        axes[0].set_title("$\\log(\\delta)$ - Regularized")
        plot_with_colorbar(-1/regularized_results[1], mask, axes[1])
        axes[1].set_title("$\\tau$ - Regularized")
        plt.tight_layout()
        # plt.savefig(output_dir / "regularized_reconstruction.png", dpi=300, bbox_inches='tight')
        # plt.show()
else:
        print("No regularized solution available")

 
# ============================================
# N2N RECONSTRUCTION
# ============================================
    
print("Starting N2N reconstruction")

lambda_tv = .00001
max_epochs = 50000    # training epochs
lr = 0.005
optimisation_method = 'Adam'
mini_batch_size = 4

n2n_result = n2n(
        noisy_data=noisy_data_torch,
        time_points0=time_torch,
        mask=mask,
        optimizer_type='Adam',
        lr=lr,
        max_epochs=max_epochs,
        GT_a=a_gt_torch,
        GT_b=b_gt_torch,
        lambda_tv=lambda_tv,
        input_index=input_index,
        mini_batch_size=mini_batch_size,
        device=device,
        scale_flag=True,
        output_dir=output_dir
    )
    
# Plot N2N results
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_with_colorbar(n2n_result[0], mask, axes[0])
axes[0].set_title("$\\log(\\delta)$ - N2N")
plot_with_colorbar(-1/n2n_result[1], mask, axes[1])
axes[1].set_title("$\\tau$ - N2N")
plt.tight_layout()
# plt.savefig(output_dir / "n2n_reconstruction.png", dpi=300, bbox_inches='tight')
# plt.show()
    

 
# ============================================
# COMPARISON PLOTS
# ============================================
    
print("Generating comparison plots")
    


plot_maps( 
            pointwise_results, 
            regularized_results,
            n2n_result,
            output_dir=output_dir,
            GT = ground_truth,
        )

  
error_maps(
            ground_truth, 
            pointwise_results, 
            regularized_results,
            n2n_result,
            output_dir=output_dir
        )

plot_training_metrics(
            ground_truth, 
            pointwise_results, 
            regularized_results,
            n2n_result,
            output_dir=output_dir
        )

joint_plot(ground_truth,ground_truth, (100,210),(-3,-0.5),20)
plt.savefig(output_dir / "jointplot_gt.png", dpi=300, bbox_inches='tight')
joint_plot(np.stack((pointwise_results[0],np.minimum(-2e-3,pointwise_results[1]))),ground_truth, (10,480),(-10,12),20)
plt.savefig(output_dir / "jointplot_pw.png", dpi=300, bbox_inches='tight')
joint_plot(regularized_results,ground_truth, (100,210),(-3,-0.5),20)
plt.savefig(output_dir / "jointplot_reg.png", dpi=300, bbox_inches='tight')
joint_plot(n2n_result,ground_truth, (100,210),(-3,-0.5),20)

plt.savefig(output_dir / "jointplot_n2n.png", dpi=300, bbox_inches='tight')


