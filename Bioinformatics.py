import tabix
import pandas as pd
import Inference as inf
import HypothesisTesting as ht

def get_cpg_in_range(chr, start, end, cpg_sites_file):
    '''Get all CpG sites within a specified genomic range.
    
    Args:
        chr: Chromosome name
        start: Start position of the range
        end: End position of the range
        cpg_sites_file: Path to tabix-indexed CpG sites file
        
    Returns:
        List of CpG positions within the specified range
    '''
    tbx = tabix.open(cpg_sites_file)
    
    results = tbx.query(str(chr), start, end)
    cpg_positions = []
    for result in results:
        cpgsite = int(result[1])
        if cpgsite >= start and cpgsite <= end:
            cpg_positions.append(cpgsite)
    
    return cpg_positions

def extract_reads_in_range(chr, start, end, df, cpg_in_range):
    '''Extract reads overlapping with the target genomic range and adjust their methylation states.
    
    Args:
        chr: Chromosome name
        start: Start position of the range
        end: End position of the range
        df: DataFrame containing methylation data
        cpg_in_range: List of CpG positions in the target range
        
    Returns:
        DataFrame containing filtered and adjusted reads
    '''
    # Filter reads that overlap with the target range
    mask = (df[0] == chr) & (df[1] < end) & (df[2] > start)
    filtered_df = df.loc[mask].copy()

    # Adjust start positions for reads starting before the target range
    start_clip = (filtered_df[1] < start)
    filtered_df.loc[start_clip, 1] = start
    
    # Adjust end positions for reads ending after the target range
    end_clip = (filtered_df[2] > end)
    filtered_df.loc[end_clip, 2] = end
    
    # Process reads that were clipped at the start
    for idx in filtered_df[start_clip].index:
        end_pos = filtered_df.loc[idx, 2]
        try:
            end_idx = cpg_in_range.index(end_pos)
            k1 = end_idx + 1
            if len(filtered_df.loc[idx, 3]) >= k1:
                filtered_df.loc[idx, 3] = filtered_df.loc[idx, 3][-k1:]
            else:
                filtered_df.loc[idx, 3] = ""  
            filtered_df.loc[idx, 1] = cpg_in_range[0]
        except ValueError:
            pass
    
    # Process reads that were clipped at the end
    for idx in filtered_df[end_clip].index:
        start_pos = filtered_df.loc[idx, 1]
        try:
            start_idx = cpg_in_range.index(start_pos)
            k2 = len(cpg_in_range) - start_idx
            if len(filtered_df.loc[idx, 3]) >= k2:
                filtered_df.loc[idx, 3] = filtered_df.loc[idx, 3][:k2]
            else:
                filtered_df.loc[idx, 3] = "" 
            filtered_df.loc[idx, 2] = cpg_in_range[-1]
        except ValueError:
            pass
    
    return filtered_df

def construct_x(start, end, methylation_status, cpg_in_range, strand):
    '''Construct methylation state vector for a single read.
    
    Args:
        start: Start position of the read
        end: End position of the read
        methylation_status: String representing methylation states (e.g., "101")
        cpg_in_range: List of CpG positions in the target range
        strand: Strand direction ("+" or "-")
        
    Returns:
        List representing methylation states (-1: missing, 0: unmethylated, 1: methylated)
    '''
    t = len(cpg_in_range)
    l = len(methylation_status)
    x = [-1] * t
    
    start_idx = cpg_in_range.index(start)
    end_idx = cpg_in_range.index(end)
    
    for i, m in enumerate(methylation_status):
        if start_idx + i >= t:
            break  
        if m == '0':
            x[start_idx + i] = 0  # Unmethylated
        elif m == '1':
            x[start_idx + i] = 1  # Methylated
        elif m == '.':
            x[start_idx + i] = -1  # Missing data

    return x

def build_xobs(result_df, cpg_in_range):
    '''Build observation matrix by expanding repeated reads.
    
    Args:
        result_df: DataFrame containing processed reads
        cpg_in_range: List of CpG positions in the target range
        
    Returns:
        List of methylation state vectors (observation matrix)
    '''
    xobs = []
    
    for _, row in result_df.iterrows():
        start = row[1]  # Start position
        end = row[2]  # End position
        methylation_status = str(row[3])  # Methylation states
        repeat_count = row[4]  # Read count
        strand = str(row[5])  # Strand direction
                
        # Construct methylation vector
        x = construct_x(start, end, methylation_status, cpg_in_range, strand)
        
        # Expand based on read count
        for _ in range(repeat_count):
            xobs.append(x)
    
    return xobs