
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from utils_real_data import *
from utils_N2N import *

#Enter device here, 'cuda' for GPU, and 'cpu' for CPU
device = 'cuda'
# Setup device
device, dtype = setup_device(prefer_cuda=True)
    

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))



# Choose dataset
#       High SNR : 1
#       Med  SNR : 2
#       Low  SNR : 3

sample_id = 1

# Create output directory
base_name = f"real_dataset_0{sample_id}"
counter = 1
while True:
    output_dir = Path(f"{base_name}_{counter}")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        break
    counter += 1

print(f"Output directory created: {output_dir}")

 
print("Starting N2N reconstruction pipeline")


noisy_data_torch, time_torch, metadata = load_photoluminescence_dataset(
    data_directory="FaPbI3\\results\\try_1\\",
    sample_id=sample_id,
    experiment_name="2023_02_10_INRIA", 
    experiment_noise_name="2023_02_10_INRIA_noise",
    time_indices=range(4, 11),  # chosen indices indices
    return_torch=True,
    cuda_flag=True,
    verbose=True
)


B,C,H,W = noisy_data_torch.shape

input_index = random.randint(0,B-1)
noisy_data_mean_np = np.mean(noisy_data_torch.detach().cpu().numpy(), axis = 0)
time_np = time_torch.cpu().numpy()
mask = []

# ========================
# POINTWISE RECONSTRUCTION 
# ========================
print("Computing pointwise reconstruction")

a_pw, b_pw = pointwise_reconstruction(
        noisy_data_mean_np,time_np
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
  
data_dir = Path(".")
    
# Try different file formats and naming conventions
possible_files = [
        data_dir / f"real_dataset_{sample_id}_REG.pkl"
    ]
    
for file_path in possible_files:
    print(f"Loading regularized solution from {file_path}")
    if file_path.exists():
        print(f"Loading regularized solution from {file_path}")
        try:
            with open(file_path, 'rb') as f:
                regularized_results =  pickle.load(f)[0]
        except Exception as e:
                print(f"Failed to load .pkl file: {e}")
                regularized_results= None
                print("No regularized solution found")

   
if regularized_results is not None:
        print("Regularized solution loaded")
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
        plot_with_colorbar(regularized_results[0], mask, axes[0])
        axes[0].set_title("$\\log(\\delta)$ - Regularized")
        plot_with_colorbar(-1/regularized_results[1], mask, axes[1])
        axes[1].set_title("$\\tau$ - Regularized")
        plt.tight_layout()

# ==========================
# NOISE2NOISE RECONSTRUCTION 
# ==========================

print("Starting N2N reconstruction")

lambda_tv = .00001
max_epochs = 5000    # training epochs
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
        GT_a=None,
        GT_b=None,
        lambda_tv=lambda_tv,
        input_index=input_index,
        mini_batch_size=mini_batch_size,
        device=device,
        scale_flag=True,
        output_dir=output_dir
    )



# ============================================
# COMPARISON PLOTS
# ============================================
    
print("Generating comparison plots")
    
plot_maps(
            pointwise_results, 
            regularized_results,
            n2n_result,
            output_dir=output_dir,
        )




joint_plot(np.stack((pointwise_results[0],np.minimum(-2e-3,pointwise_results[1]))), (470,540),(15,17),20)
plt.savefig(output_dir / "jointplot_pw.png", dpi=300, bbox_inches='tight')
joint_plot(regularized_results, (470,540),(15,17),20)
plt.savefig(output_dir / "jointplot_reg.png", dpi=300, bbox_inches='tight')
joint_plot(n2n_result, (470,540),(15,17),20)
plt.savefig(output_dir / "jointplot_n2n.png", dpi=300, bbox_inches='tight')