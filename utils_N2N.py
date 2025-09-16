"""
THIS IS THE UTIL FILE FOR THE N2N RECONSTRUCTION.
IT INVOLVES 

    - HUBER REGULARISATION
    - "FAKE" MINIBATCHING : at each epoch the loss is only computed over a single minibatch of data
    - SHUFFLING : at each epoch the set of target cubes is shuffled so each minibatch is different

"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
import seaborn as sns
import matplotlib.colors as mcolors
import pickle
import scipy.io
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def np_to_torch(img_np):
    """Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    """
    return torch.from_numpy(img_np)[None, :]

def plot_with_colorbar(x, mask,ax,quantiles=None):
    """
    Plot image with colorbar and proper scaling.
    
    Args:
        x (np.ndarray): Image to plot
        mask (np.ndarray): Mask to apply (masked regions set to NaN)
        ax (matplotlib.axes.Axes): Axes to plot on
        quantiles (tuple, optional): Quantiles for color scaling
    """
    x_masked = x.copy()
    x_masked[mask] = np.nan
    if quantiles:
        vmin,vmax = quantiles[0]
    else:
        vmin,vmax = np.nanquantile(x, [0.005,0.995])
    im = ax.imshow(x_masked, cmap = 'viridis', vmin =vmin, vmax = vmax )
    ax.axis('off')    
    plt.colorbar(im, ax=ax, shrink =0.4)


# ============================================================================
# Total Variation Operations (PyTorch)
# ============================================================================

def hor_forward_torch(x):
    """ Horizontal forward finite differences (with Neumann boundary conditions) """
    hor = torch.zeros_like(x)
    hor[:,:,:-1] = x[:,:,1:] - x[:,:,:-1]
    return hor

def ver_forward_torch(x):
    """ Vertical forward finite differences (with Neumann boundary conditions) """
    ver = torch.zeros_like(x)
    ver[:,:-1,:] = x[:,1:,:] - x[:,:-1,:]
    return ver

def dir_op_torch(x):
    """Compute directional derivatives."""
    h = hor_forward_torch(x)
    v = ver_forward_torch(x)
    return torch.stack((h,v), 0)

def TV_op_torch(x, epsilon=1e-6):
    """
    Compute Total Variation regularization term.
    
    Args:
        x (torch.Tensor): Input tensor
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        torch.Tensor: TV regularization value
    """
    return torch.sum(torch.sqrt(epsilon + torch.sum( dir_op_torch(x)**2, dim =0, keepdim=True) ))

def STV_op_torch(x,epsilon = 1e-6):
    """
    Compute Structure Tensor Total Variation regularization.
    
    Args:
        x (torch.Tensor): Input tensor with 2 channels
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        torch.Tensor: Structure Tensor TV value
    """
    # Structure Tensor TV operator
    x0 = dir_op_torch(x[:,0:1,:])
    x1 = dir_op_torch(x[:,1:2,:])
    return torch.sum(torch.sqrt(epsilon + torch.sum( x0**2, dim =0, keepdim=True) + torch.sum( x1**2, dim =0, keepdim=True) ))


# ============================================================================
# Total Variation Operations (NumPy)
# ============================================================================

def hor_forward_np(x):
    """ Horizontal forward finite differences (with Neumann boundary conditions) """
    hor = np.zeros_like(x)
    hor[:,:,:-1] = x[:,:,1:] - x[:,:,:-1]
    return hor

def ver_forward_np(x):
    """ Vertical forward finite differences (with Neumann boundary conditions) """
    ver = np.zeros_like(x)
    ver[:,:-1,:] = x[:,1:,:] - x[:,:-1,:]
    return ver

def dir_op_np( x):
    h = hor_forward_np(x)
    v = ver_forward_np(x)
    return np.stack((h,v), 0)

def TV_op_np(x):
    if x.ndim <3:
        x = x[np.newaxis,:,:]
    return np.sum(np.sqrt(0.000001 + np.sum( dir_op_np(x)**2, axis =0) ))

def STV_op_np(x):
    # Structure Tensor TV operator
    if x.ndim <4:
        x = x[np.newaxis,:,:]
    x0 = dir_op_np(x[:,0,:])
    x1 = dir_op_np(x[:,1,:])
    return np.sum(np.sqrt(0.000001 + np.sum( x0**2, axis =0) + np.sum( x1**2, axis =0) ))

# ============================================================================
# Neural Network Architecture
# ============================================================================

class network(nn.Module):
    """
    Two-branch CNN for reconstruction tasks.
    
    Args:
        n_channels (int): Number of input channels
        embed_channels (int): Number of embedding channels (default: 48)
    """

    def __init__(self,n_chan,chan_embed=48):
        super(network, self).__init__()

        self.act_N1 = nn.LeakyReLU(negative_slope=0.2, inplace=True) # ORIGINAL
        self.act2_N1 = nn.ReLU()

        self.conv1_N1 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2_N1 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3_N1 = nn.Conv2d(chan_embed, 1, 1)

        self.act_N2 = nn.LeakyReLU(negative_slope=0.2, inplace=True) # ORIGINAL 0.2
        self.act2_N2 = nn.ReLU()
 
        self.conv1_N2 = nn.Conv2d(n_chan,chan_embed,3,padding=1)
        self.conv2_N2 = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1)
        self.conv3_N2 = nn.Conv2d(chan_embed, 1, 1)

        self.BN1_N1 = nn.BatchNorm2d(chan_embed)
        self.BN2_N1 = nn.BatchNorm2d(chan_embed)
        self.BN3_N1 = nn.BatchNorm2d(1)

        self.BN1_N2 = nn.BatchNorm2d(chan_embed)
        self.BN2_N2 = nn.BatchNorm2d(chan_embed)
        self.BN3_N2 = nn.BatchNorm2d(1)
    

    def forward(self, x):
        xPad = torch.nn.functional.pad(x, (3,3,3,3), mode='reflect')
        #xPad = self.act(self.conv(xPad))

        x1 = self.act_N1(self.BN1_N1(self.conv1_N1(xPad)))
        x1 = self.act_N1(self.BN2_N1(self.conv2_N1(x1)))
        x1 = ((self.conv3_N1(x1)))

        x2 = self.act_N2(self.BN1_N2(self.conv1_N2(xPad)))
        x2 = self.act_N2(self.BN2_N2(self.conv2_N2(x2)))
        x2 = ((self.conv3_N2(x2)))

        return torch.cat((x1[:,:,3:-3,3:-3],x2[:,:,3:-3,3:-3]),1)

   
# PARAMETER INITIALISATION:
def initialise_weights(model, init_func, *params, **kwargs):
    """
    Initialize model weights with specified function.
    
    Args:
        model (nn.Module): Model to initialize
        init_func: Initialization function
        *args, **kwargs: Arguments for initialization function
    """
    for p in model.parameters():
        init_func(p, *params, **kwargs)


# ============================================================================
# Loss Functions and Metrics
# ============================================================================

def myMSE(GT, SOL):
    """
    Compute normalized MSE for each channel separately.
    
    Returns:
        tuple: Normalized MSE for channel 0 and channel 1
    """    
    with torch.no_grad():
        mse_a = (torch.norm(GT[:,0,]- SOL[:,0,])**2).item()/SOL[:,0,].numel()
        mean_a = (torch.abs(GT[:,0,])).mean().item()
        mse_b = (torch.norm(GT[:,1,]- SOL[:,1,])**2).item()/SOL[:,0,].numel()
        mean_b = (torch.abs(GT[:,1,])).mean().item()

    return mse_a/abs(mean_a), mse_b/abs(mean_b)



def myNMSE(GT, SOL):
    with torch.no_grad():

        mse_a = (torch.norm(GT[:,0,]- SOL[:,0,])**2).item()/SOL[:,0,].numel()
        mean_a = (torch.abs(GT[:,0,])).mean().item()
        mse_b = (torch.norm(GT[:,1,]- SOL[:,1,])**2).item()/SOL[:,0,].numel()
        mean_b = (torch.abs(GT[:,1,])).mean().item()

    return mse_a/abs(mean_a) + mse_b/abs(mean_b)


# LOSS FUNCTION
def reconstruction_loss(gt, pred):
    """
    Compute reconstruction loss.
    """
    loss_fn = torch.nn.HuberLoss(delta = 1)
    return loss_fn(gt,pred)


def compute_total_loss(model,y_input,y_target, time_points, lambda_tv = 0.0):
    """
    Compute total loss including data fidelity and regularization terms.
    
    Args:
        model (nn.Module): Neural network model
        input_data (torch.Tensor): Input data
        target_data (torch.Tensor): Target data
        time_points (torch.Tensor): Time points for reconstruction
        lambda_tv (float): TV regularization weight
        
    Returns:
        torch.Tensor: Total loss
    """
    pred = model(y_input)
    
    # Data fidelity loss
    loss_fid = 0
    for t_i in range(time_points.shape[0]):
        loss_fid        += reconstruction_loss(y_target[:,t_i:t_i+1,:,:],  pred[:,0:1,:] + time_points[t_i]*pred[:,1:2,:])

    loss_fid = loss_fid + lambda_tv*(STV_op_torch(pred) ) 

    return loss_fid

# ============================================================================
# Training Functions
# ============================================================================

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, y_target):
        super(MyDataset, self).__init__()
        
        self.y_target = y_target

    def __getitem__(self, index):
        y_target = self.y_target[index]

        return y_target
    
    def __len__(self):
        return len(self.y_target.size(0))

def forward(model, y_input):

    with torch.no_grad():
        pred = model(y_input)
    return pred



# N2N DENOISER
def n2n(noisy_data, time_points0,mask,optimizer_type='Adam', lr=1e-3, max_epochs=10000, step_size=5000,gamma=0.8, GT_a=None ,GT_b=None, lambda_tv=0.0 , input_index=0, mini_batch_size=1, device = None, scale_flag = True, output_dir = "."):
    """
    Train Noise2Noise denoiser for reconstruction.
    
    Args:
        noisy_data (torch.Tensor): Noisy input data
        time_points (torch.Tensor): Time points for reconstruction
        mask (np.ndarray): Mask for visualization
        optimizer_type (str): Optimizer type ('Adam' or 'SGD')
        learning_rate (float): Learning rate
        max_epochs (int): Maximum number of epochs
        step_size (int): Scheduler step size
        gamma (float): Scheduler gamma
        ground_truth_a (torch.Tensor, optional): Ground truth for channel A
        ground_truth_b (torch.Tensor, optional): Ground truth for channel B
        lambda_tv (float): TV regularization weight
        input_index (int): Index to use as input
        mini_batch_size (int): Mini-batch size
        device (str, optional): Device to use
        
    Returns:
        tuple: Reconstructed channels A and B, and training metrics
    """
                    
    PLOT_FLAG = True
    figsize = (10.5,3.5)
    if device is None:
        device = DEVICE

    # INITIALISATION OF THE METHOD
    B,C,H,W = noisy_data.shape
    model = network(C)

    if scale_flag:
        scale_factor =  torch.max(time_points0)
    else:
        scale_factor = torch.asarray(1)

    time_points = time_points0/scale_factor

    # PREPARE DATA INDICES
    targetIndexList = list(range(B))
    targetIndexList.remove(input_index)
    
    input_data = noisy_data[input_index:input_index+1,:]
    target_data = noisy_data[targetIndexList,:]

    loader = DataLoader(target_data, batch_size=mini_batch_size, shuffle=True,)

    # CHOICE OF THE INITIALISATION - TO BE COMPLETED
    if 0:
        initialise_weights(model, torch.nn.init.normal_, mean=0., std=.01) 
        #initialise_weights(model, torch.nn.init.constant_, .001) 

    model = model.to(DEVICE)

    # TEST NETWORK COMPATIBILITY
    try:
        out = forward(model,input_data)
        print("The input and the network dimensions are compatible")
    except:
        print("There's something wrong with the dimensions of the tensors")
        return


    # OPTIMIZATION METHOD
    match optimizer_type:
        case 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr)
        case 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        case _:
            print("Invalid Choice")
            return 
    

    if GT_a is not None and GT_b is not None:
        GT_combined = torch.cat((GT_a,GT_b),0).unsqueeze(0)
        STV_GT = STV_op_torch(GT_combined).item()

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # TRAINING METRICS
    metrics = {
        'loss': [],
        'loss_rel_change': [1.0],
        'stv': [],
        'solution_rel_change': []
    }

    if GT_a is not None and GT_b is not None:
        metrics['nmse'] = []

    for epoch in tqdm(range(max_epochs)):

        optimizer.zero_grad()

        # GET MINI-BATCH
        target_data_batch = next(iter(loader))

        # COMPUTE LOSS
        loss = compute_total_loss(model, input_data,target_data_batch, time_points, lambda_tv)

        # BACKWARD PASS
        loss.backward()
        optimizer.step()
        scheduler.step()

        # RECORD METRICS
        metrics['loss'].append(loss.item())
        current_output = forward(model,input_data)

        if epoch >0:
            loss_rel_change = abs(metrics['loss'][epoch] - metrics['loss'][epoch-1]) / abs(metrics['loss'][epoch])
            metrics['loss_rel_change'].append(loss_rel_change)

            sol_rel_change = torch.norm(previous_output - current_output) / torch.norm(previous_output)
            metrics['solution_rel_change'].append(sol_rel_change.item())
            
            # STOPPING CRITERIA
            if metrics['solution_rel_change'][-1] < 1e-5 and metrics['loss_rel_change'][-1] < 1e-4:
                print(f"Converged at epoch {epoch}")
                break            

        # CHECK FOR NAN VALUES
        if torch.isnan(current_output).any():
            print("NaN detected in output, stopping training")
            break

        # RESCALE THE OUTPUT USING THE scale_factor
        scaled_output = torch.cat((
            current_output[:,0:1,:], 
            current_output[:,1:2,:]/scale_factor),dim = 1)
        
        
        # COMPUTE ADDITIONAL METRICS
        if GT_a is not None and GT_b is not None:
            nmse = myNMSE(GT_combined,scaled_output)
            metrics['nmse'].append(nmse)

        stv = STV_op_torch(scaled_output).item()
        metrics['stv'].append(stv)

        previous_output = current_output.clone()

        stvPlotLine = 1

        # VISUALISATION
        if epoch % 1000 == 0 and epoch>0 and PLOT_FLAG:

            scale_factor_np = scale_factor.detach().cpu().numpy().squeeze(0)
            output_np  = current_output.detach().cpu().numpy()
            a1 = output_np[:,0,:].squeeze(0)
            b1 = output_np[:,1,:].squeeze(0)
            

            fig, ax = plt.subplots(1, 2)
            plot_with_colorbar(a1, mask, ax[0])
            ax[0].set_title("$\\log(\\delta)$ - N2N")
            plot_with_colorbar(-1/b1*scale_factor_np,mask, ax[1])
            ax[1].set_title("$\\tau$ - N2N")
            plt.savefig(output_dir / f"sol_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
            # plt.show()

            fig01, (ax01) = plt.subplots(3, 1, figsize=figsize)
            fig01.tight_layout()

            # LOSS / NMSE 
            ax01[0].semilogy(metrics['loss'], label = "Loss", color = 'tab:red')
            ax01[0].tick_params(axis='y', labelcolor='tab:red')
            if GT_a is not None and GT_b is not None:
                ax0bis = ax01[0].twinx()
                ax0bis.semilogy(metrics['nmse'], label = "NMSE", color = 'tab:blue')
                ax0bis.tick_params(axis='y', labelcolor='tab:blue')
                ax01[0].set_title('LOSS / NMSE')
            else:
                ax01[0].set_title('LOSS')
            
            # STV
            ax01[1].semilogy(metrics['stv'])
            if GT_a is not None and GT_b is not None:
                ax01[1].axhline(y = STV_GT, color = 'r', linestyle = '-') 
            ax01[1].set_title('STV')
            
            # LOSS RELCHG / SOL RELCHG
            ax01[2].semilogy(metrics['solution_rel_change'],color = 'tab:blue')
            ax01[2].axhline(y = 1e-5, color = 'tab:blue', linestyle = '-') 
            ax01bis = ax01[2].twinx()
            ax01bis.semilogy(metrics['loss_rel_change'], label = "loss_relch", color = 'tab:red')
            ax01bis.axhline(y = 1e-5, color = 'tab:red', linestyle = '-') 
            ax01[2].set_title('Relative Change')
            plt.savefig(output_dir / f"metrics_epoch_{epoch}.png", dpi=300, bbox_inches='tight')
            # plt.show()

    # Final output
    with torch.no_grad():
        final_output = model(input_data)
    
    scale_factor_np = scale_factor.detach().cpu().numpy().squeeze()
    final_output_np = final_output.detach().cpu().numpy()
    
    a = final_output_np[0, 0, :]
    b = final_output_np[0, 1, :] / scale_factor_np

    fig, ax = plt.subplots(1, 2)
    plot_with_colorbar(a, mask, ax[0])
    ax[0].set_title(f"$\\log(\\delta)$ - N2N")
    plot_with_colorbar(-1/b,mask,ax[1])
    ax[1].set_title(f"$\\tau$ - N2N")
    # plt.show()
    
    info_string = optimizer_type +"__"+str(lr)[2:] +"__" + str(mini_batch_size)

    print("Done!")
    return a, b, metrics, info_string


# ============================================================================
# Visualization Functions
# ============================================================================

color_list = ['tab:red','tab:green','tab:orange','tab:blue','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan', ('green', 0.3), "xkcd:sky blue", "xkcd:lavender", "xkcd:mauve", "xkcd:deep pink", "xkcd:yellow orange", "xkcd:jade", "xkcd:dusty green"]

def plot_maps( PW,REG, *n2n_results, output_dir = ".", GT=None):
    

    methods = [ ("Pointwise", PW), ("Regularized", REG)]

    if GT is not None:
        methods.append(("Ground Truth", GT))

    for i, result in enumerate(n2n_results):
        method, lerRat, mBatch= list(filter(None,result[-1].split("_")))
        method_info = method + ", 0." + lerRat + ", " + mBatch
        methods.append((method_info, result[:2]))

    n_methods = len(methods)
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for i, (name, (a_data, b_data)) in enumerate(methods):

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
        plot_with_colorbar(a_data, [], axes[ 0])
        axes[ 0].set_title(f"$\\log(\\delta)$ - {name}")
        
        plot_with_colorbar(-1/b_data, [], axes[ 1])
        axes[ 1].set_title(f"$\\tau$ - {name}")
    
        plt.tight_layout()

        plt.savefig(Path(output_dir) / f"sol - {name}.png", dpi=300, bbox_inches='tight')
        # plt.show()

def error_maps(GT, PW,REG, *n2n_results, output_dir = "."):
    
    methods = [("Ground Truth", GT), ("Pointwise", PW), ("Regularized", REG)]

    for i, result in enumerate(n2n_results):
        method, lerRat, mBatch= list(filter(None,result[-1].split("_")))
        method_info = method + ", 0." + lerRat + ", " + mBatch
        methods.append((method_info, result[:2]))

    n_methods = len(methods)
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)

    for i, (name, (a_data, b_data)) in enumerate(methods[1:]):

        fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.5))
        im0 = axes[0].imshow(a_data - GT[0], cmap = 'RdBu', vmin =-1, vmax = 1)
        axes[0].axis('off')
        plt.colorbar(im0, ax = axes[0])
        axes[0].set_title(f"$\\log(\\delta)$ err - {name}")

        im1 = axes[1].imshow(-1/b_data +1/GT[1], cmap = 'RdBu', vmin =-50, vmax = 50)
        axes[1].axis('off')
        plt.colorbar(im1,ax = axes[1])
        axes[1].set_title(f"$\\tau$ err - {name}")

        plt.savefig(Path(output_dir) / f"err - {name}.png", dpi=300, bbox_inches='tight')
        # plt.show()



def plot_training_metrics( GT,PW, REG, *n2n_results, output_dir = "."):

    figsize = (7.5,2.5)
    methods = []
    for i, result in enumerate(n2n_results):
        method, lerRat, mBatch= list(filter(None,result[-1].split("_")))
        method_info = method + ", 0." + lerRat + ", " + mBatch
        methods.append((method_info, result[2]))

    # LOSS PLOT
    fig, (ax1) = plt.subplots(1 , 1, figsize=figsize)
    fig.tight_layout()
    for index, (legend_string, metrics_dict ) in enumerate(methods):
        ax1.semilogy(metrics_dict['loss'], label = legend_string, color = color_list[index+3] ,markevery=500 )
    ax1.set_title('LOSS')
    ax1.legend()

    plt.savefig(Path(output_dir) / f"metrics_loss.png", dpi=300, bbox_inches='tight')

    # RELCHG PLOT
    fig, (ax1) = plt.subplots(1 , 1, figsize=figsize)
    fig.tight_layout()
    for index, (legend_string, metrics_dict ) in enumerate(methods):
        ax1.semilogy(metrics_dict['solution_rel_change'], label = legend_string, color = color_list[index+3] ,markevery=500 )
    ax1.set_title('RELATIVE CHANGE')
    ax1.legend()

    plt.savefig(Path(output_dir) / f"metrics_relchg.png", dpi=300, bbox_inches='tight')

    # NMSE
    GT_combined =  np_to_torch(GT).to(DEVICE)
    fig01, (ax1) = plt.subplots(1 , 1,  figsize=figsize)
    ax1.axhline(y = myNMSE(GT_combined, np_to_torch(PW).to(DEVICE)), color = color_list[1], linestyle = '--' , label='PW') 
    ax1.axhline(y = myNMSE(GT_combined, np_to_torch(REG).to(DEVICE)), color = color_list[2], linestyle = '--' , label='REG') 
    for index, (legend_string, metrics_dict ) in enumerate(methods):
        ax1.semilogy(metrics_dict['nmse'], label = legend_string, color = color_list[index+3] ,markevery=500 )
    ax1.set_title('NMSE')
    ax1.legend()

    plt.savefig(Path(output_dir) / f"metrics_nmse.png", dpi=300, bbox_inches='tight')

    # STV
    fig01, (ax1) = plt.subplots(1 , 1, figsize=figsize)
    ax1.axhline(y = STV_op_np(GT).item(), color = color_list[0], linestyle = '--', label = 'GT') 
    ax1.axhline(y = STV_op_np(PW), color = color_list[1], linestyle = '--' , label='PW') 
    ax1.axhline(y = STV_op_np(REG), color = color_list[2], linestyle = '--' , label='REG') 
    for index, (legend_string, metrics_dict ) in enumerate(methods):
        ax1.semilogy(metrics_dict['stv'], label = legend_string, color = color_list[index+3] ,markevery=500 )
    ax1.set_title('STV')
    ax1.legend()

    plt.savefig(Path(output_dir) / f"metrics_stv.png", dpi=300, bbox_inches='tight')



def create_teal_colormap(cmap_name='custom_teal_cmap'):
    """
    Creates a colormap with white at level 0, then transitions from a light
    shade of teal (#4CB391) to a darker shade.
    
    Parameters:
    cmap_name (str): Name for the new colormap
    
    Returns:
    matplotlib.colors.LinearSegmentedColormap: The created colormap
    """
    # Convert hex to RGB for the teal color rgb(43, 98, 79)
    teal_color = mcolors.to_rgb('#4CB391')  # Base teal color
    
    # Create a lighter version of teal
    light_teal = tuple(1 - (1 - c) * 0.8 for c in teal_color)  # Lighter teal
    
    # Create a darker version of teal
    dark_teal = tuple(c * .99 for c in teal_color)  # Darker teal
    pastel_pink = (255/255,197/255,211/255)
    dark_teal2 = (0/255,50/255,50/255)
    lele_green = (30/255, 86/255, 49/255)
    palette_green = (68/255, 128/255, 97/255)
    purple = (0.67,0.39,0.71)
    teal2 = (0.2,0.78,0.65)

    # Create colormap with white at exactly level 0, then light to dark teal
    # The small offset (0.0001) ensures white appears only at exactly zero
    colors = [(0, (1, 1, 1)), 
              (0.00001, purple),  # Light teal starts immediately after zero
                (1, teal2)]        # Dark teal at the end
            
    
    
    # Create the colormap with specific positions
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    
    return cmap




def joint_plot(data, GT, ylim,xlim, bins):

    # color1="#4CB391"
    color2 = "#448061"
    color3 = "#6F96AE"
    color4 = "#E74C3C"
    colors = ['#1abc9c', '#3498db', '#9b59b6','#f39c12', '#e74c3c']

    G0 = ((GT[0].reshape(-1)))
    G1 = ((-np.divide(1,GT[1].reshape(-1))))

    # Create the teal colormap
    teal_cmap = create_teal_colormap()
    h = sns.jointplot( x=data[0].reshape(-1), y=-np.divide(1,data[1].reshape(-1)), kind = 'hex',color = color3, cmap = teal_cmap,marginal_kws=dict(bins = bins), ylim=ylim, xlim = xlim, joint_kws=dict())
    h.set_axis_labels("$\\log(\\delta)$ (a.u.)", "$\\tau$ (ns)", fontsize=48)
    h.ax_joint.tick_params(axis='both', labelsize=48)
    h.ax_marg_y.remove()
    h.ax_marg_x.remove()
    cbar_ax = h.figure.add_axes([.85, .06, .05, .76])  # x, y, width, height
    cbar = plt.colorbar(cax=cbar_ax)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.formatter.set_useMathText(True)
    cbar.set_label('Number of Pixels', rotation=90, fontsize=32, labelpad=20)
    cbar.ax.yaxis.offsetText.set_fontsize(24)
    cbar.ax.yaxis.offsetText.set_position((2.6,1))
    cbar.ax.tick_params(labelsize=24)
    h.figure.legend(bbox_to_anchor=(1, 1), loc=2)
    for  i in range(len(set(G0))):
        x = list(set(G0))[i]
        y = list(set(G1[G0==list(set(G0))[i]]))
        h.ax_joint.plot(x,y,'o',ms=25 , mec=color4, mfc='none',mew=5,clip_on=False)



# ============================================================================
# DEVICE SETUP
# ============================================================================




def setup_device(prefer_cuda=True):
    """
    Setup computation device.
    
    Args:
        prefer_cuda (bool): Whether to prefer CUDA if available
        
    Returns:
        str: Device string ('cuda' or 'cpu')
        torch.dtype: Appropriate dtype for the device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = 'cuda'
        dtype = torch.cuda.FloatTensor
        print("Using CUDA GPU for computation")
    else:
        device = 'cpu' 
        dtype = torch.FloatTensor
        print("Using CPU for computation")
    
    return device, dtype



# ============================================================================
# data loading
# ============================================================================


def load_artificial_dataset(dataset_id, data_dir="."):
    """
    Load artificial photoluminescence dataset.
    
    Args:
        dataset_id (int): Dataset identifier
        data_dir (str): Directory containing data files
        
    Returns:
        dict: Loaded dataset containing cube, timeNs, A1Map, lifetimeMap
    """
    data_path = Path(data_dir) / f'artificialDataCube{dataset_id}.mat'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    print(f"Loading dataset {dataset_id} from {data_path}")
    
    try:
        mat_data = scipy.io.loadmat(str(data_path))
        return mat_data
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def prepare_noisy_data(cube_data, time_indices, batch_size, noise_magnitude, dtype):
    """
    Prepare noisy data from the original cube.
    
    Args:
        cube_data (np.ndarray): Original data cube
        time_indices (range): Time indices to use
        batch_size (int): Number of samples in batch
        noise_magnitude (float): Noise standard deviation
        dtype: PyTorch data type
        
    Returns:
        tuple: (noisy_data_torch, time_torch, scale_factor)
    """
    print(f"Preparing noisy data with noise magnitude: {noise_magnitude}")
    
    # Rearrange dimensions: [height, width,time] -> [time, height, width]
    cube_arranged = np.moveaxis(cube_data, [0, 1, 2], [1, 2, 0])
    
    # Generate additive noise
    noise_shape = (batch_size, len(time_indices)) + cube_data.shape[:2]
    noise = np.random.normal(size=noise_shape, scale=noise_magnitude)
    
    # Add noise and apply threshold to avoid log of negative numbers
    noisy_cube = np.maximum(1e-6, cube_arranged[time_indices, :, :] + noise)
    
    # Apply log transform
    log_cube = np.real(np.log(noisy_cube))
    
    # Convert to PyTorch tensor
    log_cube_torch = torch.from_numpy(log_cube).type(dtype).detach()
    
    return log_cube_torch

def load_regularized_solution(dataset_id, noise_magnitude, reg_id = 1, data_dir="."):
    """
    Load pre-computed regularized reconstruction solution.
    
    Args:
        dataset_id (int): Dataset identifier
        noise_magnitude (float): Noise magnitude used
        data_dir (str): Directory containing solution files
        
    Returns:
        np.ndarray or None: Regularized solution if available
    """
    data_dir = Path(data_dir)
    
    # Try different file formats and naming conventions
    possible_files = [
        data_dir / f"data{dataset_id}_reg{reg_id}_{noise_magnitude:.0e}.pkl"
    ]
    
    for file_path in possible_files:
        print(f"Loading regularized solution from {file_path}")
        if file_path.exists():
            print(f"Loading regularized solution from {file_path}")
            try:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                    print(f"Failed to load .pkl file: {e}")
    
    print("No regularized solution found")
    return None


# ============================================================================
# Pointiwise Linear Regression
# ============================================================================

def pointwise_reconstruction(log_data, time_points):
    """
    Perform pointwise linear reconstruction.
    
    Args:
        log_data (np.ndarray): Log-transformed data [time, height, width]
        time_points (np.ndarray): Time points corresponding to each frame
        
    Returns:
        tuple: (a_map, b_map) from linear fitting
    """
    print("Performing pointwise linear reconstruction")
    
    height, width = log_data.shape[1], log_data.shape[2]
    a_map = np.zeros((height, width))
    b_map = np.zeros((height, width))
    
    # Fit linear model for each pixel
    for i in tqdm(range(height), desc="Pointwise fitting"):
        for j in range(width):
            # Linear fit: log(intensity) = b * time + a
            coeffs = np.polyfit(time_points, log_data[:, i, j], 1)
            b_map[i, j] = np.real(coeffs[0])
            a_map[i, j] = np.real(coeffs[1])
    
    return a_map, b_map
