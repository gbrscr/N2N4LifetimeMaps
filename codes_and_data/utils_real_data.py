import spe2py.spe2py as spe
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import glob
import pandas as pd 
import re
import os
import torch

def gw_factor_rel_3ns(gw):
    """Gets the factor of intensity gained when varying the GW relative to the signal at 3ns"""
    gws = [0.480000000000000,0.600000000000000,0.660000000000000,0.900000000000000,0.980000000000000,1.10000000000000,1.20000000000000,1.50000000000000,2,2.50000000000000,3,3.50000000000000,4,4.50000000000000,5,5.50000000000000,6,6.50000000000000,7,7.50000000000000,8,8.50000000000000,9,9.50000000000000,10,11,12,13,14,15,16,17,18,19,20]
    re_3ns = [0.0613730796556675,0.0640008540146135,0.0639978803340475,0.0786335608034963,0.0809888584359928,0.0886247862682889,0.219423455974177,0.295525121171345,0.524015300889828,1.09252616643982,1,1.65699795647824,2.48882046214284,2.95219005088583,3.31269372600207,3.72127512310610,4.07207875543512,4.28890015555523,4.50158944946037,4.75803056222285,4.90978364043009,5.04148473763559,5.29538690577377,5.48541014657545,5.76527129789776,6.22453395901982,7.04356127220531,7.44257302359270,8.01619530431116,8.63907743034972,9.37679894120067,9.85697943043237,10.3712179417789,10.9050409568512,11.7480632558031]
    if gw<10:
        fun = interp1d(gws, re_3ns)
        rep = fun(gw)
    else:
        rep = 0.5809 * gw
    return rep

def get_infos_from_spe_footer(footer_spe):
    """Extract experimental parameters from SPE file footer"""
    infos = {}
    mode_gates = footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.Mode.cdata
    
    if mode_gates == 'Repetitive':
        infos["gate_delay"] = float(footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse["delay"])
        infos["gw"] = float(footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.RepetitiveGate.Pulse["width"])
    else:
        infos["gw"] = float(footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Gating.Sequential.StartingGate.Pulse["width"])
    
    infos["emiccd"] = float(footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Intensifier.EMIccd.Gain.cdata)
    infos["accumulations"] = float(footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.ReadoutControl.Accumulations.cdata)
    infos["flat_field_enabled"] = footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Experiment.OnlineCorrections.FlatfieldCorrection.Enabled.cdata
    infos["flat_field_path"] = footer_spe.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera.Experiment.OnlineCorrections.FlatfieldCorrection.ReferenceFile.cdata

    # Compute the factors
    infos["factor_to_photon_number"] = 1/(0.145*infos["emiccd"])
    infos["factor_to_photon_flux"] = infos["factor_to_photon_number"]/(3*1e-9 * gw_factor_rel_3ns(infos["gw"]) * infos["accumulations"])

    return infos

def get_parameter_from_name(file_name, param_name, default_value):
    """Extract parameter value from filename using regex pattern"""
    pattern = "_" + param_name + "[0-9][.]?[0-9e+-]*"
    match = re.search(pattern, file_name)

    if match is not None:
        match_pattern = match.group()
        no_match_sample = file_name.split(match_pattern)
        value = eval(match_pattern.replace("_" + param_name, ''))
        rest_of_name = ''.join(no_match_sample)
    else:
        value = default_value
        rest_of_name = file_name
    return value, rest_of_name

def analyse_experimental_tree(root_dir, additional_patterns=[]):
    """Analyze experimental file tree and extract metadata from filenames"""
    filelist_with_paths = glob.glob(join(root_dir, "**/*.*"))
    filelist = [os.path.basename(list_item) for list_item in filelist_with_paths]
    filelist_dir = [os.path.dirname(list_item) for list_item in filelist_with_paths]

    real_data_index_in_file_list = []
    real_data_exp_name = []
    
    # Define default patterns
    patterns = [
        {"pattern": 'sample', "name": 'Sample', "base_value": -32, "list_values": []},
        {"pattern": "power", "name": 'Power', "base_value": 0, "list_values": []},
        {"pattern": "rep", "name": 'Repetition', "base_value": 1, "list_values": []},
        {"pattern": "lambda", "name": 'Lambda', "base_value": 532, "list_values": []}
    ]
    
    if len(additional_patterns) > 0:
        patterns = patterns + additional_patterns

    for k, filename in enumerate(filelist):
        if '.spe' in filename:
            real_data_index_in_file_list.append(k)
            rest_of_name = filename.replace('.spe', '')

            for pattern in patterns:
                value, rest_of_name = get_parameter_from_name(rest_of_name, pattern["pattern"], pattern["base_value"])
                pattern["list_values"].append(value)

            real_data_exp_name.append(rest_of_name)

    # Build table
    table_content = None
    var_names = []
    
    for p, pattern in enumerate(patterns):
        if p == 0:
            table_content = np.array(pattern["list_values"])[:, np.newaxis]
        else:
            table_content = np.concatenate((table_content, np.array(pattern["list_values"])[:, np.newaxis]), axis=1)
        var_names.append(pattern["name"])

    var_names.append('IndexInFiles')
    table_content = np.concatenate((table_content, np.array(real_data_index_in_file_list)[:, np.newaxis]), axis=1)

    real_data_table = pd.DataFrame(table_content, columns=var_names)
    real_data_table['Experiment'] = real_data_exp_name
    
    # Sort table
    headers = real_data_table.columns.values
    sort_columns = [col for col in headers if col not in ['Experiment', 'IndexInFiles']]
    real_data_table = real_data_table.sort_values(by=sort_columns)

    return real_data_table, filelist, filelist_dir

def load_photoluminescence_dataset(
    data_directory,
    sample_id,
    experiment_name,
    experiment_noise_name,
    laser_pulse_delay=722.88,
    time_selection_limits=None,
    magnification=50,
    binning=1,
    return_torch=True,
    cuda_flag=True,
    log_transform=True,
    time_indices=None,
    verbose=False
):
    """
    Load photoluminescence lifetime imaging dataset from SPE files.
    
    Parameters:
    -----------
    data_directory : str
        Path to the directory containing SPE files
    sample_id : int or float
        Sample identifier to filter files
    experiment_name : str
        Name of the main experiment (e.g., "2023_02_10_INRIA")
    experiment_noise_name : str
        Name of the noise experiment (e.g., "2023_02_10_INRIA_noise")
    laser_pulse_delay : float, optional
        Temporal position of the pulse in camera temporal space (default: 722.88 ns)
    time_selection_limits : list, optional
        [min_time, max_time] to filter time points (default: None, uses all times)
    magnification : int, optional
        Microscope magnification (default: 50)
    binning : int, optional
        Camera binning (default: 1)
    return_torch : bool, optional
        Whether to return torch tensors (default: True)
    cuda_flag : bool, optional
        Whether to use CUDA tensors (default: True)
    log_transform : bool, optional
        Whether to apply log transform to data (default: True)
    time_indices : list, optional
        Specific time indices to use (default: None, uses time_selection_limits)
    verbose : bool, optional
        Whether to print progress information (default: False)
    
    Returns:
    --------
    data_cube : numpy.ndarray or torch.Tensor
        4D array (frames, time_points, height, width) containing PL data
    time_points : numpy.ndarray or torch.Tensor
        1D array of time points corresponding to each time slice
    metadata : dict
        Dictionary containing experimental metadata
    """
    
    # Define patterns for extracting gate width and delay from filenames
    pattern_gw = {"pattern": "gw", "name": 'GateWidth', "base_value": 3, "list_values": []}
    pattern_delay = {"pattern": "delay", "name": 'GateDelay', "base_value": 720, "list_values": []}
    
    # Analyze file tree
    if verbose:
        print(f"Analyzing files in directory: {data_directory}")
    
    meta_info_files, filelist, filelist_dir = analyse_experimental_tree(
        data_directory, [pattern_gw, pattern_delay]
    )
    
    # Filter files for the specified sample
    sub_list_of_files = meta_info_files[meta_info_files.Sample == sample_id]
    
    if len(sub_list_of_files) == 0:
        raise ValueError(f"No files found for sample {sample_id}")
    
    # Get available gate delays
    available_gate_delays = sorted(sub_list_of_files.GateDelay.unique())
    
    if verbose:
        print(f"Found {len(available_gate_delays)} time points for sample {sample_id}")
    
    # Initialize containers
    first_file = True
    new_cube = None
    new_cube_noise = None
    new_cube_time = []
    mean_pl_over_the_image = []
    gate_widths = []
    
    for k, delay in enumerate(available_gate_delays):
        
        # Load noise file
        selection_noise = [x and y for x, y in zip(
            list(sub_list_of_files.Experiment == experiment_noise_name),
            list(sub_list_of_files.GateDelay == delay)
        )]
        selected_file_meta_noise = sub_list_of_files[selection_noise]
        
        if len(selected_file_meta_noise) == 0:
            if verbose:
                print(f"Warning: No noise file found for delay {delay}")
            mean_image_noise = np.zeros((1, 512, 512))  # Default shape
        else:
            exp_to_open_noise = filelist[int(list(selected_file_meta_noise.IndexInFiles)[0])]
            file_path_noise = os.path.join(filelist_dir[int(list(selected_file_meta_noise.IndexInFiles)[0])], exp_to_open_noise)
            loaded_files_noise = spe.SpeFile(file_path_noise)
            image_noise = np.squeeze(np.array(loaded_files_noise.data))
            mean_image_noise = np.mean(image_noise, axis=0, keepdims=True)
        
        # Load main data file
        selection = [x and y for x, y in zip(
            list(sub_list_of_files.Experiment == experiment_name),
            list(sub_list_of_files.GateDelay == delay)
        )]
        selected_file_meta = sub_list_of_files[selection]
        
        if len(selected_file_meta) == 0:
            if verbose:
                print(f"Warning: No data file found for delay {delay}")
            continue
            
        exp_to_open = filelist[int(list(selected_file_meta.IndexInFiles)[0])]
        file_path = os.path.join(filelist_dir[int(list(selected_file_meta.IndexInFiles)[0])], exp_to_open)
        loaded_files = spe.SpeFile(file_path)
        image = np.squeeze(np.array(loaded_files.data))
        
        # Extract metadata
        infos = get_infos_from_spe_footer(loaded_files.footer)
        
        # Initialize cube on first iteration
        if first_file:
            n_frames, height, width = image.shape
            new_cube = np.zeros((n_frames, len(available_gate_delays), height, width))
            new_cube_noise = np.zeros((len(available_gate_delays), height, width))
            first_file = False
        
        # Store noise
        new_cube_noise[k, :, :] = mean_image_noise
        
        # Process image (remove noise and convert to photon flux)
        image_denoised = (image - mean_image_noise) * infos["factor_to_photon_flux"]
        
        # Compute real time relative to laser pulse
        real_time = infos["gate_delay"] - laser_pulse_delay
        
        # Store processed data
        new_cube[:, k, :, :] = image_denoised
        new_cube_time.append(real_time + infos["gw"] / 2)
        mean_pl_over_the_image.append(np.mean(image_denoised))
        gate_widths.append(infos["gw"])
        
        if verbose:
            print(f"Processed time point {k+1}/{len(available_gate_delays)}: {real_time:.2f} ns")
    
    # Convert time to numpy array
    new_cube_time = np.array(new_cube_time)
    
    # Apply time selection
    if time_indices is not None:
        indices_to_use = time_indices
    elif time_selection_limits is not None:
        indices_to_use = [i for i, t in enumerate(new_cube_time) 
                         if time_selection_limits[0] <= t <= time_selection_limits[1]]
    else:
        indices_to_use = list(range(len(new_cube_time)))
    
    if verbose:
        print(f"Using {len(indices_to_use)} time points out of {len(new_cube_time)}")
    
    # Filter data
    final_cube = new_cube[:, indices_to_use, :, :]
    final_times = new_cube_time[indices_to_use]
    
    # Apply log transform if requested
    if log_transform:
        final_cube = np.maximum(1e-6, final_cube)
        final_cube = np.real(np.log(final_cube))
    
    # Convert to torch tensors if requested
    if return_torch:
        if cuda_flag and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
            
        final_cube = torch.from_numpy(final_cube).type(dtype).detach()
        final_times = torch.from_numpy(final_times).type(dtype).detach()
    
    # Prepare metadata
    metadata = {
        'sample_id': sample_id,
        'experiment_name': experiment_name,
        'experiment_noise_name': experiment_noise_name,
        'laser_pulse_delay': laser_pulse_delay,
        'magnification': magnification,
        'binning': binning,
        'available_gate_delays': available_gate_delays,
        'gate_widths': gate_widths,
        'mean_pl_over_image': mean_pl_over_the_image,
        'time_selection_limits': time_selection_limits,
        'indices_used': indices_to_use,
        'log_transformed': log_transform,
        'original_shape': new_cube.shape if new_cube is not None else None,
        'final_shape': final_cube.shape if hasattr(final_cube, 'shape') else final_cube.shape
    }
    
    return final_cube, final_times, metadata

# Example usage function
def load_sample_dataset(data_directory, sample_id, **kwargs):
    """
    Simplified wrapper for loading a dataset with common defaults.
    
    Example:
    --------
    data_cube, times, metadata = load_sample_dataset(
        "path/to/data", 
        sample_id=3,
        experiment_name="2023_02_10_INRIA",
        experiment_noise_name="2023_02_10_INRIA_noise",
        time_indices=range(4, 11)  # Use specific time indices
    )
    """
    return load_photoluminescence_dataset(data_directory, sample_id, **kwargs)
