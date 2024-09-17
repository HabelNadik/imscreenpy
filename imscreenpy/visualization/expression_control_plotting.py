import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .single_cell_vis import Visualizer, get_single_cell_images
from .qc_plotting import show_randomly_sampled_images


def create_barchart_data(id_df, celltypes):
    """
    Make dataframe with the following columns:
    num cells - for each celltype and total per well
    Treated - yes or no, depending on whether it's a negative control well or not
    Well - add the well id
    """
    all_cellnums = []
    df_celltypes = []
    treated_or_not = []
    df_wells = []
    wells = list(set(id_df['Metadata_Well'].values))
    wells.sort()
    for well in wells:
        well_mask = (id_df['Metadata_Well'] == well).to_numpy()
        if id_df[well_mask]['Drug'].values[0] == 'DMSO':
            treatment = 'No'
        else:
            treatment = 'Yes'
        ### append values for all cells to list
        all_cellnums.append(np.sum(well_mask))
        df_celltypes.append('All cells')
        df_wells.append(well)
        treated_or_not.append(treatment)
        for celltype in celltypes:
            ### append celltype specific values to list
            celltype_num = np.sum(id_df[well_mask][celltype])
            all_cellnums.append(celltype_num)
            df_celltypes.append(celltype)
            df_wells.append(well)
            treated_or_not.append(treatment)
    df_dict = {'Metadata_Well' : df_wells, 'Cellnumber' : all_cellnums, 'Treated' : treated_or_not, 'Celltype' : df_celltypes}
    df = pd.DataFrame(df_dict)
    return df

def get_intensity_borders_for_depiction(intensity_vector, mask):
    negative_population = intensity_vector[~mask]
    positive_population = intensity_vector[mask]
    if np.sum(mask) == 0:
        positive_population = negative_population
    elif np.sum(~mask) == 0:
        negative_population = positive_population
    ## get lower, middle, upper population positive
    positive_upper_quartile = np.quantile(positive_population, 0.75)
    positive_lower_quartile = np.quantile(positive_population, 0.25)
    positive_upper_borders = [positive_upper_quartile, np.nanmax(positive_population)]
    positive_lower_borders = [np.nanmin(positive_population), positive_lower_quartile]
    positive_middle_borders = [positive_lower_quartile, positive_upper_quartile]
    # same for negative
    negative_upper_quartile = np.quantile(negative_population, 0.75)
    negative_lower_quartile = np.quantile(negative_population, 0.25)
    negative_upper_borders = [negative_upper_quartile, np.nanmax(negative_population)]
    negative_lower_borders = [np.nanmin(negative_population), negative_lower_quartile]
    negative_middle_borders = [negative_lower_quartile, negative_upper_quartile]
    ## all borders
    positive_borders = [positive_lower_borders, positive_middle_borders, positive_upper_borders]
    negative_borders = [negative_lower_borders, negative_middle_borders, negative_upper_borders]
    return positive_borders, negative_borders

def get_boolean_for_intensities(full_intensity_vector, num_cells, left_border, right_border):
    bool_vec = (full_intensity_vector > left_border) & (full_intensity_vector < right_border)
    print('Getting boolean for intensities. Found {} entries within bool of shape {}'.format(np.sum(bool_vec), full_intensity_vector.shape))
    out = np.zeros(bool_vec.shape)
    if np.sum(bool_vec) > 0:
        positions = np.argwhere(bool_vec.flatten()).flatten()
        samples = np.random.choice(positions, size=num_cells, replace=False)
        out[samples] = 1
    return out

def make_expression_classification_figure(df, intensity_columns, celltypes,\
     vis_folder, threshold_mean_models=None, savename=None):
    if not (vis_folder is None):
        vsr = Visualizer(df, vis_folder,  x_id='nuclei_AreaShape_Center_X', y_id='nuclei_AreaShape_Center_Y')
    else:
        vsr = None
    masks = [df[cell_type].to_numpy().astype(bool) for cell_type in celltypes]
    bordersets = [get_intensity_borders_for_depiction(df[int_col].to_numpy(), mask) for int_col, mask in zip(intensity_columns, masks)]
    fluorophores = [int_col.split('_')[-1] for int_col in intensity_columns]
    ## collect cell images for depiction
    positive_cells_per_color = []
    negative_cells_per_color = []
    all_intensities = []
    for i, borders in enumerate(bordersets):
        pos_borders, neg_borders = borders
        intensities = df[intensity_columns[i]].to_numpy()
        log_intensities = np.log10(intensities + np.finfo(np.float32).eps).reshape(-1,)
        cells_positive = []
        cells_negative = []
        print('Making expression classification figure for marker {}. Looking at intensity vector of shape {}'.format(celltypes[i], log_intensities.shape))
        print('Log Min {} Max {} Mean {} Median {}'.format(np.nanmin(log_intensities), np.nanmax(log_intensities), np.nanmean(log_intensities), np.nanmedian(log_intensities)))
        for borders in pos_borders:
            print('Getting positive cells for border {}'.format(borders))
            bool_vec = get_boolean_for_intensities(intensities, 2, borders[0], borders[1])
            if not vsr is None:
                pos_cells = get_single_cell_images(bool_vec, 2, visualizer_here=vsr)
                cells_positive += pos_cells
        for borders in neg_borders:
            print('Getting negative cells for border {}'.format(borders))
            bool_vec = get_boolean_for_intensities(intensities, 2, borders[0], borders[1])
            if not vsr is None:
                neg_cells = get_single_cell_images(bool_vec, 2, visualizer_here=vsr)
                cells_negative += neg_cells
        positive_cells_per_color.append(cells_positive)
        negative_cells_per_color.append(cells_negative)
        all_intensities.append(intensities)
    ## collect barchart data
    cellnumber_df = create_barchart_data(df, celltypes)
    fig = plt.figure(figsize=(21, 21), constrained_layout=True)
    gs = fig.add_gridspec((2*len(celltypes)) + 5,11)  
    ### plot histograms and thresholds
    for i, intensities in enumerate(all_intensities):
        hist_axis = fig.add_subplot(gs[2*i:(2*i)+2,:5]) ### histograms go to the left side on the top
        untreated_mask = (df['Drug'] == 'DMSO').to_numpy().astype(bool)
        untreated_positive = untreated_mask & masks[i]
        treated_positive = ~untreated_mask & masks[i]
        intensities_treated = intensities[~untreated_mask]
        intensities_untreated = intensities[untreated_mask]
        _, bins_pos, patches_pos = hist_axis.hist(intensities_treated, bins=500, log=True, color='blue', alpha=0.3)
        for k, patch in enumerate(patches_pos):
            sub_vector = treated_positive[(intensities > bins_pos[k]) & (intensities < bins_pos[k+1])]
            ## color as positive if more than 50% of cells in bin are positive, treated cells have different shades of purple-ish
            if np.sum(sub_vector) > (sub_vector.shape[0] * 0.5): 
                patch.set_facecolor('#ff80b3') ### corresponds to a light purple
                patch.set_alpha(0.3)
            else:
                patch.set_facecolor('#800033') ### corresponds to a dark-ish purple
                patch.set_alpha(0.3)
        _, bins_neg, patches_neg = hist_axis.hist(intensities_untreated, bins=500, log=True, alpha=0.3)
        for k, patch in enumerate(patches_neg):
            sub_vector = untreated_positive[(intensities > bins_neg[k]) & (intensities < bins_neg[k+1])]
            ## color as positive if more than 50% of cells in bin are positive
            if np.sum(sub_vector) > (sub_vector.shape[0] * 0.5):
                patch.set_facecolor('#80e5ff') ### a light blue
                patch.set_alpha(0.3)
            else:
                patch.set_facecolor('#006680') ### a darker blue
                patch.set_alpha(0.3)
                
        if not (threshold_mean_models is None):
            threshold = threshold_mean_models[i].get_threshold()
            hist_axis.axvline(threshold, color='red')
            for mean in threshold_mean_models[i].means_:
                hist_axis.axvline(mean, color='purple')
            hist_axis.set_title(celltypes[i] + ' - {}: Threshold:{}. Fraction positive cells in control {}.'.format(fluorophores[i], round(threshold, 5), round(np.sum(untreated_positive) / np.sum(untreated_mask), 5)))
        else:
            hist_axis.set_title(celltypes[i] + ' - {}: Fraction positive cells in control {}.'.format(fluorophores[i], round(np.sum(untreated_positive) / np.sum(untreated_mask), 5)))
        hist_axis.set_xlabel(celltypes[i] + ' - {} (Log10)'.format(fluorophores[i]))
        if not vsr is None:
            ### plot representative cells
            pos_cells_here = positive_cells_per_color[i]
            neg_cells_here = negative_cells_per_color[i]
            for k in range(len(pos_cells_here)):
                pos_cell_axis = fig.add_subplot(gs[2*i,5+k])
                neg_cell_axis = fig.add_subplot(gs[(2*i)+1,5+k])
                title_pos = 'Predicted positive \n'
                title_neg = 'Predicted negative \n'
                if k == 0:
                    title_pos += 'lower quartile'
                    title_neg += 'lower quartile'
                elif k == 2:
                    title_pos += 'middle quartile'
                    title_neg += 'middle quartile'
                elif k == 4:
                    title_pos += 'upper quartile'
                    title_neg += 'upper quartile'
                if (k % 2) == 0:
                    pos_cell_axis.set_title(title_pos)
                    neg_cell_axis.set_title(title_neg)
        
                pos_cell_axis.set_xticks([])
                pos_cell_axis.set_yticks([])
                
                neg_cell_axis.set_xticks([])
                neg_cell_axis.set_yticks([])

                pos_cell_axis.imshow(pos_cells_here[k])
                neg_cell_axis.imshow(neg_cells_here[k])
    print('Done with filling gridspec with histograms and patches. Last index was {} number of rows is {}. Trying to add barcharts at {}'.format(i, (2*len(celltypes)) + 5, 2*len(fluorophores)))
    ### now plot the bars for: cellnumbers in control, cellnumbers treated, each fluorophore and the double negative population for control and treated
    barchart_axis = fig.add_subplot(gs[2*len(fluorophores):,:])
    sns.barplot(x='Treated', y='Cellnumber', hue='Celltype', data=cellnumber_df, ax=barchart_axis)
    sns.swarmplot(x='Treated', y='Cellnumber', hue='Celltype', dodge=True, color='black', data=cellnumber_df, ax=barchart_axis)
    plt.setp(barchart_axis.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)
    plt.close('all')


def make_classification_control_figure(df, cp_out_top_folder, savename, cfg):
    
    keys = list(cfg.class_assignment_dict.keys())
    keys.sort()
    celltypes_by_indices = [cfg.class_assignment_dict[key] for key in keys]
    celltype_color_dict = dict()
    for celltype in celltypes_by_indices:
        if celltype == 'ADR':
            celltype_color_dict[celltype] = 'yellow'
        elif celltype == 'MES':
            celltype_color_dict[celltype] = 'purple'
        elif celltype == 'INTER':
            celltype_color_dict[celltype] = 'blue'
        elif celltype == 'DEAD':
            celltype_color_dict[celltype] = 'black'
    show_randomly_sampled_images(df, cp_out_top_folder, celltypes_by_indices, celltype_color_dict, savename=savename,\
         num_images_to_sample=4, min_size=80, n_dmso_images_to_sample=2)