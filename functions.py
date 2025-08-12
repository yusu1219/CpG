import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import Inference as inf
import Bioinformatics as bio
import os
import pandas as pd
import warnings
import numpy as np
from typing import List
warnings.filterwarnings("ignore")

def can_merge(a: List[int], b: List[int]) -> bool:
    '''Check if two samples can be merged (non-overlapping observations).
    
    Args:
        a: First methylation state vector
        b: Second methylation state vector
        
    Returns:
        True if samples can be merged (no overlapping observations), False otherwise
    '''
    for ai, bi in zip(a, b):
        if ai != -1 and bi != -1: 
            return False
    return True

def merge_samples(a: List[int], b: List[int]) -> List[int]:
    '''Merge two methylation state vectors.
    
    Args:
        a: First methylation state vector
        b: Second methylation state vector
        
    Returns:
        Merged methylation state vector combining non-overlapping observations
    '''
    return [a[i] if a[i] != -1 else b[i] for i in range(len(a))]

def compress_xobs(xobs: List[List[int]]) -> List[List[int]]:
    '''Compress xobs sample set by merging non-overlapping observations.
    
    Args:
        xobs: List of methylation state vectors (observation matrix)
        
    Returns:
        Compressed observation matrix with merged samples where possible
    '''
    compressed = []
    for sample in xobs:
        merged = False
        for i, target in enumerate(compressed):
            if can_merge(target, sample):
                compressed[i] = merge_samples(target, sample)
                merged = True
                break
        if not merged:
            compressed.append(sample)
    return compressed

def visualize_xobs(xobs: List[List[int]], title: str):
    '''Visualize the xobs matrix.
    
    Args:
        xobs: List of methylation state vectors to visualize
        title: Title for the visualization plot
    '''
    num_samples = len(xobs)
    length = len(xobs[0]) if num_samples > 0 else 0

    # Set height for each sample and total figure height
    sample_height = 0.01  # Height per sample in inches
    fig_height = max(4, num_samples * sample_height)

    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Create colormap: white for -1, pink for 0, blue for 1
    cmap = ListedColormap(['white', '#ff9cb0', '#8cf7ff'])

    # Prepare data (map -1,0,1 to 0,1,2)
    data = np.array(xobs)
    data = np.where(data == -1, 0, data + 1)

    # Display image (maintain equal physical height for each sample)
    cax = ax.imshow(data, cmap=cmap, aspect='auto', interpolation='none')

    # Set y-axis ticks at sample centers
    ax.set_yticks([])

    # Set axis labels
    ax.set_xlabel('CpG-Sites')
    ax.set_title(title)

    # Add legend
    legend_elements = [
        patches.Patch(facecolor='#8cf7ff', label='Methylated (1)'),
        patches.Patch(facecolor='#ff9cb0', label='Unmethylated (0)'),
        patches.Patch(facecolor='white', label='Missing (-1)')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def _cal_cpel(cpg_sites_file, data, chr, start, end, step=500, compress=False, vis=False):
    '''Calculate CpG island parameters for a single file.
    
    Args:
        cpg_sites_file: Path to CpG sites file
        data: Methylation data DataFrame
        chr: Chromosome
        start: Start position
        end: End position
        step: Window size for segmentation (default: 500)
        compress: Whether to compress the observation matrix (default: False)
        vis: Whether to visualize the matrix (default: False)
        
    Returns:
        Tuple of (CpG counts per segment, estimated theta parameters)
    '''
    cpg_n_list = []  # CpG counts per segment
    xobs = []        # Observation matrix
    new_xobs = []    # Transformed observation matrix for calculations
    
    # Calculate CpG counts per segment
    for current_start in range(start, end + 1, step):
        current_end = min(current_start + step - 1, end)
        cpg_in_range = bio.get_cpg_in_range(chr, current_start, current_end, cpg_sites_file)
        cpg_n_list.append(len(cpg_in_range))
    
    # Extract reads for the entire region
    result_df = bio.extract_reads_in_range(chr, start, end, data, cpg_in_range)
    cpg_in_range = bio.get_cpg_in_range(chr, start, end, cpg_sites_file)
    
    # Build observation matrix
    xobs = bio.build_xobs(result_df, cpg_in_range)
    
    # Create transformed observation matrix (-1→0, 0→-1, 1→1)
    for read in xobs:
        new_read = []
        for val in read:
            if val == -1:
                new_read.append(0)    # Missing → 0
            elif val == 0:
                new_read.append(-1)   # Unmethylated → -1
            elif val == 1:
                new_read.append(1)    # Methylated → 1
            else:
                new_read.append(val)  # Other values unchanged
        new_xobs.append(new_read)

    # Optional compression and visualization (using original xobs)
    if compress:
        xobs = compress_xobs(xobs)
    
    # Estimate theta parameters using transformed matrix
    theta = inf.est_theta_sa(cpg_n_list, new_xobs)

    if vis and compress:
        print("compressed_xobs:")
        visualize_xobs(xobs, f"Compressed xobs Matrix (Samples: {len(xobs)})")

    return cpg_n_list, theta

def process_files(folder_path, chr, start, end, step, compress=False, vis=False):
    '''Process all files in a folder to calculate theta values.
    
    Args:
        folder_path: Path to folder containing methylation files
        chr: Chromosome
        start: Start position
        end: End position
        step: Window size for segmentation
        compress: Whether to compress observation matrices (default: False)
        vis: Whether to visualize matrices (default: False)
        
    Returns:
        Tuple of (CpG counts per segment, list of theta values for each file)
    '''
    theta_list = []
    n_list = None

    # Get all .mhap.gz files in folder
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.mhap.gz')]
    print(file_list)

    for file in file_list:
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path, sep='\t', header=None, dtype={3: 'str'})
        current_n, current_theta = _cal_cpel(
            cpg_sites_file, data, chr=chr, start=start, end=end, 
            step=step, compress=compress, vis=vis
        )

        # Verify consistent CpG counts across files
        if n_list is None:
            n_list = current_n
        else:
            assert n_list == current_n, "Inconsistent n_list across files"
        print(file, current_theta)
        theta_list.append(current_theta)

    return n_list, theta_list