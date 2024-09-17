from os.path import join
import sqlite3
from timeit import default_timer as timer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, threshold_minimum

from ..misc import well_string_to_nums, print_time
from ..db_df_processing import db_functions
from ..visualization import single_cell_vis as sc_vis
from ..db_df_processing.df_functions import split_dataframe
from .qc_config import QCconfig
from ..config import Config 
from ..drug_scoring.drug_scoring import reformat_concentrations

########## IMAGE RAW INTENSTIY BASED QC #######################
###############################################################
###############################################################


def get_qc_image_df(conn, fluorophores_of_interest, cfg):
    intensity_columns = ['Image_Intensity_MeanIntensity_{}'.format(fluo) for fluo in fluorophores_of_interest]
    metadata_columns = ['Image_Metadata_Well', 'Image_Metadata_column', 'Image_Metadata_field', 'Image_Metadata_row']
    other_qc_columns = ['Image_Count_nuclei']
    target_columns = intensity_columns + metadata_columns + other_qc_columns
    query_string = db_functions.make_query_string(target_columns, cfg.db_im_table_name)
    df = pd.read_sql(query_string, conn)
    return df

def get_image_df_outlier_wells(df, fluorophores_of_interest, qc_plot_savename=None, plot_qc=True, be_relaxed=False, ignore_cellnumbers=False, ignore_fluos=[]):
    intensity_columns = ['Image_Intensity_MeanIntensity_{}'.format(fluo) for fluo in fluorophores_of_interest]
    metadata_columns = ['Image_Metadata_Well', 'Image_Metadata_column', 'Image_Metadata_field', 'Image_Metadata_row']
    other_qc_columns = ['Image_Count_nuclei']
    wells = list(set(df['Image_Metadata_Well'].values))
    plot_qc_columns = intensity_columns + other_qc_columns
    ignore_columns = []
    if ignore_cellnumbers:
        ignore_columns += [f for f in plot_qc_columns if ('nuclei' in f) or ('DNA' in f) or ('DAPI' in f)]
    if len(ignore_fluos) > 0:
        for fluo in ignore_fluos:
            ignore_columns += [f for f in plot_qc_columns if (fluo in f)]
    
    all_qc_columns = [f for f in plot_qc_columns if not (f in ignore_columns)]
    exclusion_matrix = np.zeros((16, 24))
    wells_to_exclude = []
    matrices_to_show = []
    for qc_col in plot_qc_columns:
        values_to_check = []
        matrix = np.zeros((16, 24))
        for well in wells:
            if 'Count' in qc_col:
                value = np.sum(df[df['Image_Metadata_Well'] == well][qc_col].to_numpy())
            else:
                value = np.nanmean(df[df['Image_Metadata_Well'] == well][qc_col].to_numpy())
            values_to_check.append(value)
            row, col = well_string_to_nums(well)
            matrix[row-1,col-1] = value
        if qc_col in all_qc_columns:
            med = np.median(values_to_check)
            std = np.std(values_to_check)
            if be_relaxed:
                left = med - (4*std)
                right = med + (4*std)
            else:
                left = med - (3*std)
                right = med + (3*std)
            wells_to_exclude += [wells[i] for i in range(len(wells)) if (values_to_check[i] < left) or (values_to_check[i] > right)]
        matrices_to_show.append(matrix)
    for well in wells_to_exclude:
        row, col = well_string_to_nums(well)
        exclusion_matrix[row-1,col-1] = 1
    if plot_qc:
        ###### make plot
        num_fig_rows = len(plot_qc_columns)+1
        fig, ax = plt.subplots(nrows=num_fig_rows, figsize=(8,6*num_fig_rows))
        for i in range(len(plot_qc_columns)):
            sh = ax[i].imshow(matrices_to_show[i], cmap='coolwarm')
            ax[i].set_title(plot_qc_columns[i])
            plt.colorbar(sh, ax=ax[i])
        ax[-1].imshow(exclusion_matrix)
        if qc_plot_savename is None:
            fig.show()
        else:
            fig.savefig(qc_plot_savename)
    return wells_to_exclude

def do_image_qc(conn, fluorophores_of_interest, qc_savename, cfg, plot_qc=True, be_relaxed=False, ignore_cellnumbers=False):
    df = get_qc_image_df(conn, fluorophores_of_interest, cfg)
    outlier_wells = get_image_df_outlier_wells(df, fluorophores_of_interest, qc_plot_savename=qc_savename, plot_qc=plot_qc, be_relaxed=be_relaxed, ignore_cellnumbers=ignore_cellnumbers)
    return outlier_wells

######  END OF IMAGE RAW INTENSTIY BASED QC ###################
###############################################################
###############################################################


######  BEGINNING OF MORPHOLOGY BASED QC ######################
###############################################################
###############################################################

def exclude_poor_wells(well_array, qc_vector, well_exclusion_threshold=0.5):
    out_vec = np.zeros(qc_vector.shape)
    unique_wells = list(set(well_array))
    for well in unique_wells:
        well_mask = well_array == well
        frac_good = np.sum(qc_vector[well_mask]) / np.sum(well_mask)
        if frac_good > well_exclusion_threshold:
            out_vec[well_mask] = qc_vector[well_mask]
        else:
            print('Excluding well {} because of {} good enough segmented cells for threshold {}'.format(well, frac_good, well_exclusion_threshold))
    return out_vec.flatten().astype(bool)

def make_morphology_qc_vector(input_df, qc_cfg):
    qc_vector = np.ones(input_df.shape[0]).astype(bool)
    if not (qc_cfg.max_size_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.size_col].to_numpy() < qc_cfg.max_size_threshold)
    if not (qc_cfg.min_size_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.size_col].to_numpy() > qc_cfg.min_size_threshold)
    if not (qc_cfg.min_dapi_intensity_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.dapi_intensity_col].to_numpy() > qc_cfg.min_dapi_intensity_threshold)
    if not (qc_cfg.min_solidity_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.solidity_col].to_numpy() > qc_cfg.min_solidity_threshold)
    if not (qc_cfg.max_eccentricity_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.eccentricity_col].to_numpy() < qc_cfg.max_eccentricity_threshold)
    if not (qc_cfg.min_entropy_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.texture_entropy_col].to_numpy() > qc_cfg.min_entropy_threshold)
    if not (qc_cfg.min_contrast_threshold is None):
        qc_vector = qc_vector & (input_df[qc_cfg.texture_contrast_col].to_numpy() > qc_cfg.min_contrast_threshold)
    if not (qc_cfg.texture_sum_min_value is None):
        qc_vector = qc_vector & (input_df[qc_cfg.texture_sum_col].to_numpy() > qc_cfg.texture_sum_min_value)
    if not (qc_cfg.info_meas2_exclude_value is None):
        qc_vector = qc_vector & (input_df[qc_cfg.texture_info_meas2_col].to_numpy() != qc_cfg.info_meas2_exclude_value)
    if not (qc_cfg.correlation_exclude_values is None):
        qc_vector = qc_vector & ~np.isin(input_df[qc_cfg.correlation_col].to_numpy(), qc_cfg.correlation_exclude_values)
    return qc_vector


def do_morphology_qc(input_df, vis_folder, target_folder, well_exclusion_threshold=0.5,\
                      show_cells=False, platename='', exclude_wells_morph=True, cfg=None, id_df_add_columns=None, split_dfs=True):
    if cfg is None:
        cfg = QCconfig(None)
    elif not isinstance(cfg, QCconfig):
        print('Copying config settings to qc config')
        if isinstance(cfg, Config):
            cfg = QCconfig(cfg.cfg_folder, cfg_prefix=cfg.cfg_prefix, copy_cfg=cfg)
        else:
            cfg = QCconfig(cfg.cfg_folder, cfg_prefix=cfg.cfg_prefix)
    #### dropping nans
    print('DF shape before dropping nans is {}'.format(input_df.shape))
    input_df = input_df.dropna(axis=0, thresh=int(input_df.shape[1] * cfg.nan_fraction_threshold))
    input_df = input_df.dropna(axis=1, thresh=int(input_df.shape[0] * cfg.nan_fraction_threshold))
    print('DF shape after dropping nans is {}'.format(input_df.shape))
    #### partitioning df
    #all_columns = [f for f in input_df.columns]
    #if (id_df_add_columns is None):
    #    id_df_add_columns = []
    ## ID DF add columns are probably not needed actually
    #id_df_add_columns += [f for f in cfg.get_model_feature_columns() if not (f in id_df_add_columns)]
    print('Starting cell qc')
    #### performing qc and visualizing, qc params are stored in QCconfig object
    qc_vector = make_morphology_qc_vector(input_df, cfg)
    ### show cells if required
    if show_cells:
        vsr = sc_vis.Visualizer(input_df, vis_folder, x_id='nuclei_AreaShape_Center_X', y_id='nuclei_AreaShape_Center_Y')
        sc_vis.visualize_cells_in_grid(qc_vector, visualizer_here=vsr, max_grid_size=10, savename=join(target_folder, '{}_ok_cells.png'.format(platename)))
        sc_vis.visualize_cells_in_grid(~qc_vector, visualizer_here=vsr, max_grid_size=10, savename=join(target_folder, '{}_bad_cells.png'.format(platename)))
    if exclude_wells_morph:
        print('QC: Excluding wells by morphology qc fraction', flush=True)
        qc_vector = exclude_poor_wells(input_df['Metadata_Well'].to_numpy(), qc_vector, well_exclusion_threshold=well_exclusion_threshold)
    else:
        print('QC: No well exclusion due to morphology', flush=True)
    if split_dfs:
        print('Making split dataframe. The following columns will be additionally added to the id df')
        print(id_df_add_columns)
        feature_df, id_df = split_dataframe(input_df[qc_vector], cfg, id_df_add_columns=id_df_add_columns)
        #feature_df = feature_df[qc_vector]
        #id_df = input_df[qc_vector]
        print('DF shape after morphology qc is {}'.format(feature_df.shape), flush=True)
        #id_keep_columns = [f for f in id_columns if f in input_df.columns]
        #id_df = id_df[id_keep_columns]
        return feature_df, id_df
    else:
        out_df = input_df[qc_vector]
        return out_df


def plot_cell_feature_distributions(df, target_columns, qc_plot_savename=None, prefix=''):
    #### get median values and sums for each column of interest
    median_matrices_to_show = []
    sum_matrices_to_show = []
    wells = list(set(df['Metadata_Well'].values))
    for qc_col in target_columns:
        median_matrix = np.zeros((16, 24))
        sum_matrix = np.zeros((16, 24))
        for well in wells:
            well_values = df[df['Metadata_Well'] == well][qc_col].to_numpy()
            sum_val = np.sum(well_values)
            median_val = np.median(well_values)
            row, col = well_string_to_nums(well)
            sum_matrix[row-1,col-1] = sum_val
            median_matrix[row-1,col-1] = median_val
        median_matrices_to_show.append(median_matrix)
        sum_matrices_to_show.append(sum_matrix)
    ### make matrix of cellnumbers and median DAPI intensity per well
    cellnum_matrix = np.zeros((16,24))
    #dapi_matrix = np.zeros((16,24))
    for well in wells:
        well_df = df[df['Metadata_Well'] == well]
        row, col = well_string_to_nums(well)
        cellnum_matrix[row-1,col-1] = well_df.shape[0]
        #dapi_matrix[row-1,col-1] = np.sum(well_df['nuclei_Intensity_MeanIntensity_DNA'].to_numpy())
    ###### make plot
    num_fig_rows = len(target_columns)+1
    fig, ax = plt.subplots(nrows=num_fig_rows, ncols=2, figsize=(16,6*num_fig_rows))
    for i in range(len(target_columns)):
        ### median plot
        sh1 = ax[i,0].imshow(median_matrices_to_show[i], cmap='coolwarm')
        ax[i,0].set_title(prefix + ' Median value ' + target_columns[i])
        plt.colorbar(sh1, ax=ax[i,0])
        ### sum plot
        sh1 = ax[i,1].imshow(sum_matrices_to_show[i], cmap='coolwarm')
        ax[i,1].set_title(prefix + ' Sum ' + target_columns[i])
        plt.colorbar(sh1, ax=ax[i,1])
    sh3 = ax[-1,0].imshow(cellnum_matrix)
    ax[-1,0].set_title(prefix + ' Number of cells')
    plt.colorbar(sh3, ax=ax[-1,0])
    if qc_plot_savename is None:
        fig.show()
    else:
        fig.savefig(qc_plot_savename)
    return

######  BEGINNING OF OTHER QC FUNCTIONS ######################
###############################################################
###############################################################

def filter_implausible_cellnumbers(in_df, n_stds=2, subsetting_column=None, target_column='NumberOfCells', verbose=False):
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    if subsetting_column is None:
        subsetting_column = 'Plate'
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    keep_mask = np.zeros(in_df.shape[0]).astype(bool)
    for subset in subsets:
        subset_mask = (in_df[subsetting_column] == subset).to_numpy()
        dmso_wells = np.unique(in_df[dmso_mask & subset_mask]['Metadata_Well'].to_numpy())
        treated_wells = np.unique(in_df[~dmso_mask & subset_mask]['Metadata_Well'].to_numpy())
        n_treated_wells = treated_wells.shape[0]
        if isinstance(target_column, str):
            dmso_cellnumbers = np.array([in_df[(in_df['Metadata_Well'] == well) & dmso_mask][target_column].values[0] for well in dmso_wells])
            treated_cellnumbers = np.array([in_df[(in_df['Metadata_Well'] == well) & ~dmso_mask][target_column].values[0] for well in treated_wells])
            dmso_mean = np.mean(dmso_cellnumbers)
            dmso_std = np.std(dmso_cellnumbers)
            keep_wells_treated = [treated_wells[i] for i in range(n_treated_wells) if treated_cellnumbers[i] <= (dmso_mean + (n_stds * dmso_std))]
            keep_wells_mask = (in_df['Metadata_Well'].isin(keep_wells_treated + list(dmso_wells))).to_numpy()
            keep_mask = keep_mask | (subset_mask & keep_wells_mask)
        else:
            keep_wells_mask_all = np.ones(in_df.shape[0]).astype(bool)
            for t_col in target_column:
                dmso_cellnumbers = np.array([in_df[(in_df['Metadata_Well'] == well) & dmso_mask][t_col].values[0] for well in dmso_wells])
                treated_cellnumbers = np.array([in_df[(in_df['Metadata_Well'] == well) & ~dmso_mask][t_col].values[0] for well in treated_wells])
                dmso_mean = np.mean(dmso_cellnumbers)
                dmso_std = np.std(dmso_cellnumbers)
                keep_wells_treated = [treated_wells[i] for i in range(n_treated_wells) if treated_cellnumbers[i] <= (dmso_mean + (n_stds * dmso_std))]
                keep_wells_mask = (in_df['Metadata_Well'].isin(keep_wells_treated + list(dmso_wells))).to_numpy()
                keep_wells_mask_all = keep_wells_mask_all & keep_wells_mask
            keep_mask = keep_mask | (subset_mask & keep_wells_mask_all)
    if verbose:
        print('Keeping {} of {} wells'.format(int(np.sum(keep_mask)), in_df.shape[0]), flush=True)
    return in_df[keep_mask.flatten()]

def add_normalized_cellnumber_column(in_df, target_column, subsetting_column=None):
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    if subsetting_column is None:
        subsetting_column = 'Plate'
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    normed_vals = np.zeros(in_df.shape[0])
    for subset in subsets:
        subset_mask = (in_df[subsetting_column] == subset).to_numpy()
        mean_val = in_df[(in_df['Drug'] == 'DMSO') & subset_mask][target_column].mean()
        normed_vals[subset_mask] = in_df[target_column][subset_mask] / mean_val
    return in_df.assign(**{target_column + '_normalized': normed_vals})
        

def get_mad_filtered_wells(treatment_df, normalized_column, frac_diff=0.3, prev_frac_diff=0.3, prev_frac_diff_single=None):
    """
    get wells were conc point i+1 is not more than prev_frac_diff higher than conc point i
    and where the median absolute deviation is not more than frac_diff

    """
    keep_wells = []
    sorted_df = treatment_df.sort_values('Concentration')
    prev_val = 1.
    unique_concentrations = sorted_df['Concentration'].unique()
    for c_index, concentration in enumerate(unique_concentrations):
        conc_df = sorted_df[sorted_df['Concentration'] == concentration]
        median_val = conc_df[normalized_column].mean()
        if  ((prev_frac_diff is None) or (median_val <= (prev_val + prev_frac_diff))) and (prev_frac_diff_single is None):
            median_abs_dev = np.median(np.abs(conc_df[normalized_column] - median_val))
            if (frac_diff is None) or (median_abs_dev <= frac_diff):
                keep_wells += conc_df['Metadata_Well'].to_list()
                prev_val = median_val
        elif not (prev_frac_diff_single is None):
            vals_above = conc_df[normalized_column].to_numpy() <= (prev_val + prev_frac_diff_single)
            median_abs_dev = np.median(np.abs(conc_df[normalized_column] - median_val))
            if np.sum(vals_above) == 1:
                keep_wells += conc_df[vals_above]['Metadata_Well'].to_list()
                prev_val = np.mean(conc_df[vals_above][normalized_column].to_numpy())
            elif (np.sum(vals_above) > 1) and (frac_diff is None) or (median_abs_dev <= frac_diff):
                keep_wells += conc_df[vals_above]['Metadata_Well'].to_list()
                prev_val = median_val            
    return keep_wells



def filter_bad_replicates(in_df, frac_diff=0.3, prev_frac_diff=0.3, prev_frac_diff_single=None, subsetting_column=None, target_columns='NumberOfCells', verbose=True, min_n_replicates=1):
    if in_df['Concentration'].to_numpy().dtype == np.object_:
        in_df = reformat_concentrations(in_df, adjust_for_combos=True)
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    if subsetting_column is None:
        subsetting_column = 'Plate'
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    keep_mask = np.zeros((in_df.shape[0], len(target_columns))).astype(bool)
    for i in range(len(target_columns)):
        keep_mask[:,i] = dmso_mask
    local_df = reformat_concentrations(in_df, adjust_for_combos=True)
    for i, t_col in enumerate(target_columns):
        local_df = add_normalized_cellnumber_column(local_df, t_col, subsetting_column)
        for subset in subsets:
            subset_mask = (local_df[subsetting_column] == subset).to_numpy()
            treatments = local_df[subset_mask & ~dmso_mask]['Drug'].unique()
            keep_wells_per_subset = []
            for treatment in treatments:
                keep_wells_here = get_mad_filtered_wells(local_df[subset_mask & (local_df['Drug'] == treatment)], t_col + '_normalized', frac_diff=frac_diff, prev_frac_diff=prev_frac_diff, prev_frac_diff_single=prev_frac_diff_single)
                if len(keep_wells_here) >= min_n_replicates:
                    keep_wells_per_subset += keep_wells_here
            keep_mask[:,i] = keep_mask[:,i] | (local_df['Metadata_Well'].isin(keep_wells_per_subset).to_numpy() & subset_mask) | dmso_mask
            if verbose:
                print('Got {} wells to keep for subset {}'.format(len(keep_wells_per_subset), subset))
                print(np.sum(keep_mask[:,i]))
    full_keep_mask = keep_mask[:,0]
    if len(target_columns) >= 1:
        for i in range(len(target_columns)):
            full_keep_mask = full_keep_mask & keep_mask[:,i]
    if verbose:
        print('Bad replicate filtering: Keeping {} of {} wells'.format(int(np.sum(full_keep_mask)), in_df.shape[0]), flush=True)
    return in_df[full_keep_mask.flatten()]

def filter_drugs_by_avg_error(in_df, frac_diff=0.3, subsetting_column=None, target_columns='NumberOfCells', verbose=False):
    if in_df['Concentration'].to_numpy().dtype == np.object_:
        in_df = reformat_concentrations(in_df, adjust_for_combos=True)
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    if subsetting_column is None:
        subsetting_column = 'Plate'
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    keep_mask = np.zeros((in_df.shape[0], len(target_columns))).astype(bool)
    for i in range(len(target_columns)):
        keep_mask[:,i] = dmso_mask
    local_df = reformat_concentrations(in_df, adjust_for_combos=True)
    
    for i, t_col in enumerate(target_columns):
        local_df = add_normalized_cellnumber_column(local_df, t_col, subsetting_column)
        normalized_column = t_col + '_normalized'
        for subset in subsets:
            subset_mask = (local_df[subsetting_column] == subset).to_numpy()
            treatments = local_df[subset_mask & ~dmso_mask]['Drug'].unique()
            keep_treatments = []
            for treatment in treatments:
                treatment_df = local_df[subset_mask & (local_df['Drug'] == treatment)]
                sorted_df = treatment_df.sort_values('Concentration')
                prev_val = 1.
                diffs_per_conc = []
                unique_concentrations = sorted_df['Concentration'].unique()
                for c_index, concentration in enumerate(unique_concentrations):
                    conc_df = sorted_df[sorted_df['Concentration'] == concentration]
                    median_val = conc_df[normalized_column].mean()
                    median_abs_dev = np.median(np.abs(conc_df[normalized_column] - median_val))
                    diffs_per_conc.append(median_abs_dev)
                if np.mean(diffs_per_conc) <= frac_diff:
                    keep_treatments.append(treatment)
            keep_mask[:,i] = keep_mask[:,i] | (local_df['Drug'].isin(keep_treatments).to_numpy() & subset_mask) | dmso_mask
            if verbose:
                print('Got {} drugs to keep for subset {}'.format(len(keep_treatments), subset))
                print(np.sum(keep_mask[:,i]))
    full_keep_mask = keep_mask[:,0]
    if len(target_columns) >= 1:
        for i in range(len(target_columns)):
            full_keep_mask = full_keep_mask & keep_mask[:,i]
    if verbose:
        print('Bad replicate filtering: Keeping {} of {} wells'.format(int(np.sum(full_keep_mask)), in_df.shape[0]), flush=True)
    return in_df[full_keep_mask.flatten()]
    
    
def cap_population_numbers(in_df, target_columns, subsetting_column=None, verbose=False):
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    treated_mask = ~dmso_mask
    if subsetting_column is None:
        subsetting_column = 'Plate'
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    ## subset based on which level to normalize on, usually plate
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    cellnumbers_per_col = in_df[target_columns].to_numpy().copy()
    n_corrected_wells = 0
    single_col = len(target_columns) == 1
    for subset in subsets:
        subset_mask = (in_df[subsetting_column] == subset).to_numpy()
        for i, t_col in enumerate(target_columns):
            ## get dmso cellnumbers to calculate mean
            #dmso_cellnumbers = np.array([in_df[(in_df['Metadata_Well'] == well) & dmso_mask][t_col].values[0] for well in dmso_wells])
            if single_col:
                dmso_avg = np.mean(cellnumbers_per_col[dmso_mask & subset_mask])
                cellnumbers_per_col = cellnumbers_per_col.flatten()
                cellnumbers_here = cellnumbers_per_col
            else:    
                dmso_avg = np.mean(cellnumbers_per_col[dmso_mask & subset_mask, i])
                cellnumbers_here = cellnumbers_per_col[:,i]
            if verbose:
                print('## Capping: BEFORE CORRECTION')
                print('Average DMSO cellnumber for {} in Plate {} is {}'.format(t_col, subset, dmso_avg))
                print('Average treated cellnumber for {} in Plate {} is {}'.format(t_col, subset, np.nanmean(in_df[t_col].to_numpy()[subset_mask & ~dmso_mask])))
                print('Max treated cellnumber for {} in Plate {} is {}'.format(t_col, subset, np.nanmax(in_df[t_col].to_numpy()[treated_mask])))
            filter_mask = subset_mask & treated_mask & (cellnumbers_here >= dmso_avg)
            if single_col:
                cellnumbers_per_col[filter_mask] = dmso_avg ## cap cellnumbers with mean
            else:
                cellnumbers_per_col[filter_mask, i] = dmso_avg ## cap cellnumbers with mean
            n_corrected_wells_here = np.sum(subset_mask & treated_mask & (in_df[t_col].to_numpy() >= dmso_avg)) ##  count corrected wells for printing
            n_corrected_wells += n_corrected_wells_here
            if verbose:
                print('Capping: Correcting {} wells for {} in Plate {}'.format(n_corrected_wells_here, t_col, subset), flush=True)
    ass_dict = dict()
    for i, t_col in enumerate(target_columns):
        if verbose:
            print('## Capping: AFTER CORRECTION')
            for subset in subsets:
                subset_mask = (in_df[subsetting_column] == subset).to_numpy()
                dmso_avg = np.nanmedian(cellnumbers_per_col[dmso_mask & subset_mask, i])
                print('Average DMSO cellnumber for {} in Plate {} is {}'.format(t_col, subset, dmso_avg))
                print('Average treated cellnumber for {} in Plate {} is {}'.format(t_col, subset, np.nanmean(cellnumbers_per_col[subset_mask & ~dmso_mask, i])))
                print('Max treated cellnumber for {} in Plate {} is {}'.format(t_col, subset, np.nanmax(cellnumbers_per_col[subset_mask & ~dmso_mask, i])))
        if single_col:
            ass_dict[t_col] = cellnumbers_per_col
        else:
            ass_dict[t_col] = cellnumbers_per_col[:,i]
    if verbose:
        print('Capping: Corrected {} of {} wells'.format(int(n_corrected_wells/len(target_columns)), in_df.shape[0]), flush=True)
    return in_df.assign(**ass_dict)

def filter_bad_conditions(in_df, subsetting_column=None, target_column='NumberOfCells', mad_threshold=0.5, verbose=False):
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    if subsetting_column is None:
        subsetting_column = 'Plate'
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    keep_mask = np.zeros(in_df.shape[0]).astype(bool)
    n_conditions_kept = 0
    for subset in subsets:
        subset_mask = (in_df[subsetting_column] == subset).to_numpy()
        dmso_wells = np.unique(in_df[dmso_mask & subset_mask]['Metadata_Well'].to_numpy())
        dmso_cellnumbers = np.array([in_df[(in_df['Metadata_Well'] == well) & dmso_mask][target_column].values[0] for well in dmso_wells])
        norm_dmso_cellnumbers = dmso_cellnumbers / np.mean(dmso_cellnumbers)
        mad = np.median(np.abs(norm_dmso_cellnumbers - np.median(norm_dmso_cellnumbers)))
        if mad <= mad_threshold:
            keep_mask = keep_mask | subset_mask
            n_conditions_kept += 1
    if np.sum(keep_mask) == 0:
        print('Full dataframe was bad. Returning None')
        return None
    else:
        if verbose:
            print('Keeping {} out of {} conditions'.format(n_conditions_kept, subsets.shape[0]), flush=True)
    return in_df[keep_mask]


def filter_by_min_n_wells(in_table, target_column, min_n):
    unique_elems = in_table[target_column].unique()
    keep_elems = []
    for elem in unique_elems:
        unique_wells = in_table[in_table[target_column] == elem]['Metadata_Well'].unique()
        if unique_wells.shape[0] >= min_n:
            keep_elems.append(elem)
    return in_table[in_table[target_column].isin(keep_elems)]

def scale_viability_table(in_df, target_columns, subsetting_column=None):
    dmso_mask = (in_df['Drug'] == 'DMSO').to_numpy()
    treated_mask = ~dmso_mask
    if subsetting_column is None:
        subsetting_column = 'Plate'
    if isinstance(target_columns, str):
        target_columns = [target_columns]
    ## subset based on which level to normalize on, usually plate
    subsets = np.unique(in_df[subsetting_column].to_numpy())
    cellnumbers_per_col = in_df[target_columns].to_numpy().copy()
    single_col = len(target_columns) == 1
    for subset in subsets:
        subset_mask = (in_df[subsetting_column] == subset).to_numpy()
        for i, t_col in enumerate(target_columns):
            ## get dmso cellnumbers to calculate mean
            treated_mask = ~dmso_mask
            if single_col:
                dmso_avg = np.mean(cellnumbers_per_col[dmso_mask & subset_mask])
                min_val = np.min(cellnumbers_per_col[subset_mask])
                max_val = np.max(cellnumbers_per_col[subset_mask])
                cellnumbers_per_col = cellnumbers_per_col.flatten()
            else:    
                dmso_avg = np.mean(cellnumbers_per_col[dmso_mask & subset_mask, i])
                min_val = np.min(cellnumbers_per_col[subset_mask, i])
                max_val = np.max(cellnumbers_per_col[subset_mask, i])
            filter_mask = subset_mask & treated_mask
            ### normalize cellnumbers such that min is zero and max stays the same
            if single_col:
                cellnums_to_normalize = cellnumbers_per_col[filter_mask]
                cellnumbers_per_col[filter_mask] =  (cellnums_to_normalize - min_val) * (max_val/(max_val - min_val))
            else:
                cellnums_to_normalize = cellnumbers_per_col[filter_mask, i]
                cellnumbers_per_col[filter_mask, i] = (cellnums_to_normalize - min_val) * (max_val/(max_val - min_val))
    ass_dict = dict()
    for i, t_col in enumerate(target_columns):
        if single_col:
            ass_dict[t_col] = cellnumbers_per_col
        else:
            ass_dict[t_col] = cellnumbers_per_col[:,i]
    return in_df.assign(**ass_dict)
