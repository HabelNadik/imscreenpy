from os.path import join, isfile, isdir
from os import listdir
import sys
import sqlite3

import pandas as pd
import numpy as np
import sklearn
import scipy
from scipy import stats
import skimage
from skimage.io import imread
from skimage.util import img_as_float, img_as_ubyte
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage import exposure
import matplotlib.pyplot as plt
from string import ascii_uppercase

#import single_cell_vis as sc_vis
from .single_cell_vis import Visualizer
from ..misc import nums_to_well_string

def make_query_string(columns, table_name, row_col_field_tuple=None):
    column_string = ''
    for col in columns:
        column_string += col + ', '
    column_string = column_string[:len(column_string)-2]
    query_string = "SELECT {} FROM {};".format(column_string, table_name)
    if not (row_col_field_tuple is None):
        row = row_col_field_tuple[0]
        col = row_col_field_tuple[1]
        field = row_col_field_tuple[2]
        condition_string = ' WHERE ((Metadata_row = {}) AND (Metadata_column = {}) AND (Metadata_field = {}));'.format(row, col, field)
        query_string = query_string[:len(query_string)-1] + condition_string
    return query_string

id_columns = ['ObjectNumber', 'nuclei_Number_Object_Number', 'ImageNumber', 'unique_image_id',\
              'Metadata_Well', 'Metadata_row', 'Metadata_column', 'Metadata_field', 'nuclei_Intensity_MeanIntensity_GFP',\
              'nuclei_Intensity_MedianIntensity_GFP', 'nuclei_Intensity_StdIntensity_GFP', 'nuclei_AreaShape_Center_X', 'nuclei_AreaShape_Center_Y']

other_columns = ['nuclei_Intensity_MedianIntensity_APC', 'nuclei_Intensity_MedianIntensity_PE']

id_columns += [f + 'IllumCorrected' for f in id_columns if ('GFP' in f)]

qc_columns = ['nuclei_Intensity_MeanIntensity_DNA', 'nuclei_Texture_SumEntropy_DNA_3_00_256', 'nuclei_Texture_Contrast_DNA_3_00_256',\
              'nuclei_Texture_Correlation_DNA_3_00_256', 'nuclei_Texture_Correlation_DNA_3_00_256', 'nuclei_Texture_SumAverage_DNA_3_00_256',\
               'nuclei_Texture_InfoMeas2_DNA_3_00_256']


def make_name(row, col, field, channel, template='r{}c{}f{}p01-ch{}sk1fk1fl1.tiff'):
    if len(str(row)) > 1:
        row = str(row)
    else:
        row = '0' + str(row)
    if len(str(col)) > 1:
        col = str(col)
    else:
        col = '0' + str(col)
    if len(str(field)) > 1:
        field = str(field)
    else:
        field = '0' + str(field)
    return template.format(row, col, field, channel)

def get_image_make_color(platename, row, col, field, use_channels=None, annotation=None, image_folder=None):
    if image_folder is None:
        image_folder = annotation[annotation['PlateID'] == platename]['RawImageDataPath'].values[0]
    print('Getting image from folder ' + image_folder)
    green = [0, 217, 0]
    yellow = [252, 222, 0]
    red = [252, 41, 0]
    blue = [45, 113, 255]
    out_image = np.zeros((1080, 1080, 3), dtype=np.float64)
    channel_colors = [red, yellow, green, blue]
    fluorophores = ['APC', 'PE', 'GFP', 'DAPI']
    for i in range(len(fluorophores)):
        fluo = fluorophores[i]
        if (use_channels is None) or (fluo in use_channels):
            fluo_id = annotation[annotation['PlateID'] == platename][fluo + '_Channel_ID'].values[0]
            if  fluo_id in [1,2,3,4]:
                filename = make_name(row, col, field, int(fluo_id))
                im = img_as_float(imread(join(image_folder, filename)))
                im_min, im_max = np.percentile(im, (1., 99.8))
                im = rescale_intensity(im, in_range=(im_min, im_max))
                im = exposure.adjust_sigmoid(im, cutoff=0.6, gain=8)
                print(im.dtype)
                for k in range(3):
                    out_image[:,:,k] += (im * (float(channel_colors[i][k]) / 255.))#.astype(np.uint8)
                print('Reading image with shape {} min {} and max {}'.format(im.shape, np.amax(im), np.amin(im)))
    #out_image = exposure.adjust_sigmoid(np.clip(out_image, 0., 1.), cutoff=0.5, gain=10)
    print('After contrast adjustment and before clipping min is {} max is {}'.format(np.amin(out_image), np.amax(out_image)))
    out_image = np.clip(out_image, 0., 1.)#.astype(np.uint8)
    out_image = img_as_ubyte(out_image)
    return out_image

def has_cells(platename, row, col, field, table):
    if (row is None) and (col is None) and (field is None):
        subtable = table
    elif (platename is None) or (not ('Plate' in table.columns)):
        if ('Metadata_row' in table.columns):
            subtable = table[(table['Metadata_row'] == row) & (table['Metadata_column'] == col) & (table['Metadata_field'] == field)]
        else:# assume that table is just one field if metadata is not present
            subtable = table
    else:
        subtable = table[(table['Plate'] == platename) & (table['Metadata_row'] == row) & (table['Metadata_column'] == col) & (table['Metadata_field'] == field)]
    print('Found {} cells found for row {} col {} field {}.'.format(subtable.shape[0], row, col, field))
    if subtable.shape[0] > 0:
        print('Stickings with this field')
    else:
        print('Trying other fields')
    return subtable.shape[0] > 0

def get_xy_with_values(platename, row, col, field, table, value_column):
    scatter_x_negative = []
    scatter_y_negative = []
    scatter_x_positive = []
    scatter_y_positive = []
    if (row is None) and (col is None) and (field is None):
        subtable = table
    elif (platename is None) or (not ('Plate' in table.columns)):
        if ('Metadata_row' in table.columns):
            subtable = table[(table['Metadata_row'] == row) & (table['Metadata_column'] == col) & (table['Metadata_field'] == field)]
        else:# assume that table is just one field if metadata is not present
            subtable = table
    else:
        subtable = table[(table['Plate'] == platename) & (table['Metadata_row'] == row) & (table['Metadata_column'] == col) & (table['Metadata_field'] == field)]
    print('Getting subtable with shape {}'.format(subtable.shape))
    print('Columns in table are')
    print([f for f in table.columns])
    print('Columns in subtable are')
    print([f for f in subtable.columns])
    scatter_x, scatter_y, scatter_values = get_xy_with_values_from_image(subtable, None, return_value_column=value_column)
    return scatter_x, scatter_y, scatter_values


def get_xy_with_values_from_image(table, image_location_or_id, return_value_column=None, obj_prefix='nuclei_', custom_image_column=None):
    if (image_location_or_id is None):
        im_table = table
    elif not (custom_image_column is None):
        im_table = table[table[custom_image_column] == image_location_or_id]
    elif isinstance(image_location_or_id, int):
        im_table = table[table['ImageNumber'] == image_location_or_id]
    elif isinstance(image_location_or_id, str):
        im_table = table[table['Metadata_FileLocation'] == image_location_or_id]
    #scatter_x = im_table['nuclei_AreaShape_Center_X'].to_numpy()
    #scatter_y = 1080. - im_table['nuclei_AreaShape_Center_Y'].to_numpy()
    scatter_x = im_table[obj_prefix + 'Location_Center_X'].to_numpy()
    scatter_y = 1080. - im_table[obj_prefix + 'Location_Center_Y'].to_numpy()
    if not (return_value_column is None):
        scatter_values = im_table[return_value_column].to_numpy()
        return scatter_x, scatter_y, scatter_values
    else:
        return scatter_x, scatter_y

def norm_values(values, new_min=5, new_max=120):
    out_values = (values - np.amin(values)) * ((new_max - new_min) / (np.amax(values) - np.amin(values))) + new_min
    return out_values

def threshold_and_separate(x, y, intensities, threshold):
    x_below = []
    x_above = []
    intensities_below = []
    y_below = []
    y_above = []
    intensities_above = []
    for x_val, y_val, intensity in list(zip(x, y, intensities)):
        if intensity <= threshold:
            x_below.append(x_val)
            y_below.append(y_val)
            intensities_below.append(intensity)
        else:
            x_above.append(x_val)
            y_above.append(y_val)
            intensities_above.append(intensity)
    return x_below, y_below, intensities_below, x_above, y_above, intensities_above


def show_image_and_numerical_property(platename, row, col, field, table, target_column, savename=None, single_image_table=False):
    if not single_image_table:
        if not has_cells(platename, row, col, field, table):
            print('No cells found for row {} col {} field {}. Trying other fields.'.format(row, col, field))
            if ('Metadata_field' in table.columns):
                unique_fields = np.unique(table['Metadata_field'].to_numpy())
                field = unique_fields[0]
            print('Changed field to {}'.format(field))
        image = get_image_make_color(platename, row, col, field, table)
        scatter_y, scatter_x, scatter_values = get_xy_with_values(None, row, col, field, table, target_column)
        scatter_values = norm_values(scatter_values)
        fig, ax = plt.subplots(figsize=(30,15), ncols=2)
        ax[0].imshow(image)
        ax[1].scatter(scatter_x, scatter_y, s=scatter_values)
        ax[1].set_xlim(0, 1080)
        ax[1].set_ylim(0, 1080)
    else:
        scatter_y, scatter_x, scatter_values = get_xy_with_values(None, None, None, None, table, target_column)
        scatter_values = norm_values(scatter_values)
        fig, ax = plt.subplots(figsize=(15,15))
        ax.scatter(scatter_x, scatter_y, s=scatter_values)
        ax.set_xlim(0, 1080)
        ax.set_ylim(0, 1080)
    #plt.legend(ax=ax[1])
    if not (savename is None):
        fig.savefig(savename)
    else:
        fig.show()
    return

def subset_image_table_by_coords(image_table, x_min=0, x_max=1080, y_min=0, y_max=1080):
    x_mask = (image_table['nuclei_Location_Center_X'] >= x_min).to_numpy() & (image_table['nuclei_Location_Center_X'] < x_max).to_numpy()
    ## y coords need to be converted to matrix style
    #y_mask = (image_table['nuclei_Location_Center_Y'] >= y_min).to_numpy() & (image_table['nuclei_Location_Center_Y'] < y_max).to_numpy()
    y_max_new = y_max#1080 - y_min
    y_min_new = y_min#1080 - y_max
    y_mask = (image_table['nuclei_Location_Center_Y'] >= y_min_new).to_numpy() & (image_table['nuclei_Location_Center_Y'] < y_max_new).to_numpy()
    #y_mask = (image_table['nuclei_Location_Center_Y'] >= (1080 - y_min)).to_numpy() & (image_table['nuclei_Location_Center_Y'] < (1080 - y_max)).to_numpy()
    out = image_table[x_mask & y_mask]
    return out

def show_image_and_multi_scatterplot(image_or_path, image_table, celltypes=None, sizes=None, min_size=50, max_size=350,\
                               x_min=0, x_max=1080, y_min=0, y_max=1080, savename=None, celltype_color_dict=None):
    image_table = subset_image_table_by_coords(image_table, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    fig, ax = plt.subplots(figsize=(30,10), ncols=3)
    non_viable_bool = ~image_table['Viable'].to_numpy().astype(bool)
    non_viable_table = image_table[non_viable_bool]
    x_dead, y_dead = get_xy_with_values_from_image(non_viable_table, None, obj_prefix='nuclei_')
    dead_sizes = np.ones(non_viable_table.shape[0]) * min_size    
    if celltypes is None:
        x, y = get_xy_with_values_from_image(image_table, None, obj_prefix='nuclei_')
        sizes = np.ones(x.shape[0]) * min_size
    else:
        x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype = get_coords_colors_sizes_for_image_table(image_table, celltypes, celltype_color_dict, dot_size=min_size)
   
    if isinstance(image_or_path, str):
        im = skimage.io.imread(image_or_path)
    else:
        im = image_or_path
    ax[0].imshow(im[y_min:y_max,x_min:x_max])
    for x, y, color, c_sizes in zip(x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype):
            ax[1].scatter(x, y, color=color, edgecolor='black', sizes=c_sizes)
    for x, y, color, c_sizes in zip(x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype):
            ax[2].scatter(x, y, color=color, edgecolor='black', sizes=c_sizes)
            
    ax[2].scatter(x_dead, y_dead, sizes=dead_sizes, marker='X', color='black')
    y_scat_min = 1080 - y_max
    y_scat_max = 1080 - y_min
    ax[1].set_xlim(left=x_min, right=x_max)
    #ax[1].set_ylim(bottom=y_min, top=y_max)
    ax[1].set_ylim(bottom=y_scat_min, top=y_scat_max)
    ax[2].set_xlim(left=x_min, right=x_max)
    #ax[2].set_ylim(bottom=y_min, top=y_max)
    ax[2].set_ylim(bottom=y_scat_min, top=y_scat_max)
    clean_ax(ax[0])
    clean_ax(ax[1])
    clean_ax(ax[2])
    fig.tight_layout()
    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)
    return


def get_coords_colors_sizes_for_image_table(image_table, celltypes, celltype_color_dict, dot_size=None):
    rest_vec = np.ones(image_table.shape[0]).astype(bool)
    x_per_celltype = []
    y_per_celltype = []
    celltype_colors = []
    sizes_per_celltype = []
    for celltype in celltypes:
        ## grab celltypes for all cells
        bool_vec = image_table[celltype].to_numpy().astype(bool)
        celltype_table = image_table[bool_vec]
        x_c, y_c = get_xy_with_values_from_image(celltype_table, None, obj_prefix='nuclei_')
        x_per_celltype.append(x_c)
        y_per_celltype.append(y_c)
        celltype_colors.append(celltype_color_dict[celltype])
        rest_vec = rest_vec & ~bool_vec
        if not (dot_size is None):
            sizes_per_celltype.append(np.ones(celltype_table.shape[0]) * dot_size)
        ## grab celltypes for viable  cells
        x_c_via, y_c_via = get_xy_with_values_from_image(celltype_table, None, obj_prefix='nuclei_')
    ### grab non-classified cells    
    rest_table = image_table[rest_vec]
    if rest_table.shape[0] > 0:
        x_rest, y_rest = get_xy_with_values_from_image(rest_table, None, obj_prefix='nuclei_')
        x_per_celltype.append(x_rest)
        y_per_celltype.append(y_rest)
        celltype_colors.append('white')
        if not (dot_size is None):
            sizes_per_celltype.append(np.ones(rest_table.shape[0]) * dot_size)
    
    return x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype
    



def make_multi_channel_plot(df, platename, \
    annotation, row=6, col=6, field=2, raw_image_folder_replacement_strings=None,\
         seg_image_folder_replacement_strings=None, savename=None):
    ### make sure dataframe only contains the plate of interest
    if ('Plate' in df.columns):
        df = df[df['Plate'] == platename]
    #all_fluorophores = ['APC', 'GFP', 'PE', 'DAPI']
    all_fluorophores = ['APC', 'GFP', 'PE']
    membrane_markers_to_use = []
    fluorophores_for_membranes = []
    use_channels_per_image = []
    channel_ids = [1,2,3,4]
    marker_intensity_columns = []
    intensity_column_nametemplate = 'nuclei_Intensity_MedianIntensity_{}'
    ### get required information from annotation
    plate_row = annotation[annotation['PlateID'] == platename]
    seg_image_folder = plate_row['SegmentationOutputPathUncorrected'].values[0]
    for fluo in all_fluorophores:
        fluo_column = fluo + '_Channel_ID'
        plate_row_fluo_value = plate_row[fluo_column].values[0]
        plate_row_marker_value = plate_row[fluo].values[0]
        #if plate_row_fluo_value in channel_ids:
        if plate_row_marker_value.strip() != 'None':
            fluorophores_for_membranes.append(fluo_column)
            use_channels_per_image.append([fluo, 'DAPI'])
            membrane_markers_to_use.append(plate_row_marker_value)
            marker_intensity_columns.append(intensity_column_nametemplate.format(fluo))
    ### collect raw images and data for scatterplots
    single_channel_plus_dapi_images = []
    single_marker_coordinates_and_sizes = []
    if (raw_image_folder_replacement_strings is None):
        raw_image_folder = None
    else:
        raw_image_folder_raw = plate_row['RawImageDataPath'].values[0]
        orig = raw_image_folder_replacement_strings[0]
        target = raw_image_folder_replacement_strings[1]
        raw_image_folder = raw_image_folder_raw.replace(orig, target)
    if (seg_image_folder_replacement_strings is None):
        seg_image_folder = plate_row['SegmentationOutputPathUncorrected'].values[0]
    else:
        seg_image_folder_raw = plate_row['SegmentationOutputPathUncorrected'].values[0]
        orig = seg_image_folder_replacement_strings[0]
        target = seg_image_folder_replacement_strings[1]
        seg_image_folder = seg_image_folder_raw.replace(orig, target)
    for i in range(len(use_channels_per_image)):
        if not has_cells(platename, row, col, field, df):
            print('No cells found for row {} col {} field {}. Trying other fields.'.format(row, col, field))
            if ('Metadata_field' in df.columns):
                unique_fields = np.unique(df[(df['Metadata_row'] == row) & (df['Metadata_column'] == col)]['Metadata_field'].to_numpy())
                if len(list(unique_fields)) == 0:
                    print('NO CELLS FOUND IN WELL!! ABORTING!!!!!')
                    return
                field = unique_fields[0]
            print('Changed field to {}'.format(field))
        if not has_cells(platename, row, col, field, df):
            all_channel_image = get_image_make_color(platename, row, col, field, annotation=annotation, image_folder=raw_image_folder)
            if not (seg_image_folder is None):
                vsr = Visualizer(df, seg_image_folder)
                segmentation_image = vsr.get_field_image(row, col, field)
            else:
                segmentation_image = np.zeros((10,10,3))
            fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
            ax[0].imshow(all_channel_image)
            ax[0].set_title('Full chanel image')
            ax[1].imshow(segmentation_image)
            ax[1].set_title('Segmentation')
            if savename is None:
                fig.show()
            else:
                fig.savefig(savename)
            return
        channel_image = get_image_make_color(platename, row, col, field,\
             use_channels=use_channels_per_image[i], annotation=annotation, image_folder=raw_image_folder)
        x, y, intensity_values = get_xy_with_values(platename, row, col, field, df,  marker_intensity_columns[i])
        """
        channel_image = get_image_make_color(platename, row, col, field,\
             use_channels=use_channels_per_image[i], annotation=annotation, image_folder=raw_image_folder)
        x, y, intensity_values = get_xy_with_values(platename, row, col, field,\
             df, marker_intensity_columns[i])
        """
        intensity_values = norm_values(intensity_values)
        single_channel_plus_dapi_images.append(channel_image)
        single_marker_coordinates_and_sizes.append([x,y,intensity_values])
    all_channel_image = get_image_make_color(platename, row, col, field, annotation=annotation, image_folder=raw_image_folder)
    if not (seg_image_folder is None):
        vsr = Visualizer(df, seg_image_folder)
        segmentation_image = vsr.get_field_image(row, col, field)
    else:
        segmentation_image = np.zeros((10,10,3))
    num_rows_to_show = len(single_channel_plus_dapi_images) + 1
    fig, ax = plt.subplots(nrows=num_rows_to_show, ncols=2, figsize=(20, 10*num_rows_to_show))
    for i in range(len(single_channel_plus_dapi_images)):
        ax[i,0].imshow(single_channel_plus_dapi_images[i])
        ax[i,0].set_title('DAPI + ' + use_channels_per_image[i][0] + ' - ' + membrane_markers_to_use[i])
        x, y, sizes = single_marker_coordinates_and_sizes[i]
        ax[i,1].scatter(x,y,s=sizes)
        ax[i,1].set_title('nuclei. size scaled relative to {}'.format(marker_intensity_columns[i]))
    ax[-1,0].imshow(all_channel_image)
    ax[-1,0].set_title('Full chanel image')
    ax[-1,1].imshow(segmentation_image)
    ax[-1,1].set_title('Segmentation')
    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)
    return

def make_single_image_classification_plot(df, platename, \
    annotation, threshold, target_fluorophore, row=6, col=6, field=2, raw_image_folder_replacement_strings=None,\
         seg_image_folder_replacement_strings=None, savename=None):
    ### make sure dataframe only contains the plate of interest
    if ('Plate' in df.columns):
        df = df[df['Plate'] == platename]
    #all_fluorophores = ['APC', 'GFP', 'PE', 'DAPI']
    all_fluorophores = ['APC', 'GFP', 'PE']
    membrane_markers_to_use = []
    fluorophores_for_membranes = []
    use_channels_per_image = []
    channel_ids = [1,2,3,4]
    marker_intensity_columns = []
    if ('_Corrected' in target_fluorophore):
        target_fluorophore = target_fluorophore.split('_')[0]
        intensity_column_nametemplate = 'nuclei_Intensity_MeanIntensity_{}_Corrected'
    else:
        intensity_column_nametemplate = 'nuclei_Intensity_MeanIntensity_{}'
    ### get required information from annotation
    plate_row = annotation[annotation['PlateID'] == platename]
    seg_image_folder = plate_row['SegmentationOutputPathUncorrected'].values[0]
    if (raw_image_folder_replacement_strings is None):
        raw_image_folder = None
    else:
        raw_image_folder_raw = plate_row['RawImageDataPath'].values[0]
        orig = raw_image_folder_replacement_strings[0]
        target = raw_image_folder_replacement_strings[1]
        raw_image_folder = raw_image_folder_raw.replace(orig, target)
    if (seg_image_folder_replacement_strings is None):
        seg_image_folder = plate_row['SegmentationOutputPathUncorrected'].values[0]
    else:
        seg_image_folder_raw = plate_row['SegmentationOutputPathUncorrected'].values[0]
        orig = seg_image_folder_replacement_strings[0]
        target = seg_image_folder_replacement_strings[1]
        seg_image_folder = seg_image_folder_raw.replace(orig, target)
    fluo_column = target_fluorophore + '_Channel_ID'
    use_channels = [target_fluorophore, 'DAPI']
    marker_intensity_column = intensity_column_nametemplate.format(target_fluorophore)
    channel_image = get_image_make_color(platename, row, col, field,\
             use_channels=use_channels, annotation=annotation, image_folder=raw_image_folder)
    x, y, intensity_values = get_xy_with_values(platename, row, col, field, df,  marker_intensity_column)
    x_below, y_below, intensities_below, x_above, y_above, intensities_above = threshold_and_separate(x, y, intensity_values, threshold)
    sizes_below = norm_values(intensities_below, new_min=5, new_max=60)
    sizes_above = norm_values(intensities_above, new_min=61, new_max=120)

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    ax[0].imshow(channel_image)
    ax[0].set_title('DAPI + ' + target_fluorophore)
    ax[1].scatter(x_below,y_below,s=sizes_below)
    if target_fluorophore == 'GFP':
        ax[1].scatter(x_above, y_above, s=sizes_above, marker='x', color='green')
    else:
        ax[1].scatter(x_above, y_above, s=sizes_above, marker='x')
    ax[1].set_title('nuclei. size scaled relative to {}'.format(marker_intensity_column))

    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)
    return

def clean_ax(in_ax):
    in_ax.set_xticklabels([])
    in_ax.set_yticklabels([])
    in_ax.set_xticks([])
    in_ax.set_yticks([])
    return

def get_cell_coordinates_colors_sizes(image_table, min_size, celltypes, celltype_color_dict, distinguish_viable=False):
    if distinguish_viable:
        non_viable_bool = ~image_table['Viable'].to_numpy().astype(bool)
        non_viable_table = image_table[non_viable_bool]
        x_dead, y_dead = get_xy_with_values_from_image(non_viable_table, None, obj_prefix='nuclei_')
        dead_sizes = np.ones(non_viable_table.shape[0]) * min_size
        image_table = image_table[~non_viable_bool]
    if celltypes is None:
        x, y = get_xy_with_values_from_image(image_table, None, obj_prefix='nuclei_')
        sizes = np.ones(x.shape[0]) * min_size
    
    else:
        rest_vec = np.ones(image_table.shape[0]).astype(bool)
        x_per_celltype = []
        y_per_celltype = []
        celltype_colors = []
        sizes_per_celltype = []
        for i, celltype in enumerate(celltypes):
            #bool_vec = image_table['AAEclass'].to_numpy() == i
            bool_vec = image_table[celltype].to_numpy().astype(bool)
            celltype_table = image_table[bool_vec]
            x_c, y_c = get_xy_with_values_from_image(celltype_table, None, obj_prefix='nuclei_')
            x_per_celltype.append(x_c)
            y_per_celltype.append(y_c)
            celltype_colors.append(celltype_color_dict[celltype])
            rest_vec = rest_vec & ~bool_vec
            sizes_per_celltype.append(np.ones(celltype_table.shape[0]) * min_size)
        rest_table = image_table[rest_vec]
        x_rest, y_rest = get_xy_with_values_from_image(rest_table, None, obj_prefix='nuclei_')
        x_per_celltype.append(x_rest)
        y_per_celltype.append(y_rest)
        sizes_per_celltype.append(np.ones(rest_table.shape[0]) * min_size)
        celltype_colors.append('white')
    return x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype

def show_randomly_sampled_images(df, cp_out_top_folder, celltypes, celltype_color_dict, savename=None, num_images_to_sample=5,\
                                  min_size=80, distinguish_viable=False, x_min=0, x_max=1080, y_min=0, y_max=1080, n_dmso_images_to_sample=0):
    """
    Show randomly samples images from the dataframe that were segmented by cellprofiler
    """
    vsr = Visualizer(df, cp_out_top_folder)
    im_ids = df['unique_image_id'].to_numpy()
    unique_im_ids = np.unique(im_ids)
    ## if we have less requested images than available images, just use all images
    if (unique_im_ids.shape[0] <= num_images_to_sample) or (unique_im_ids.shape[0] <= n_dmso_images_to_sample):
        sampled_ids = list(unique_im_ids)
        num_images_to_sample = len(sampled_ids)
    elif (n_dmso_images_to_sample > 0) and (n_dmso_images_to_sample <= num_images_to_sample): # sample treated and untreated if requested
        dmso_im_ids = im_ids[(df['Drug'] == 'DMSO').to_numpy()]
        treated_im_ids = im_ids[(df['Drug'] != 'DMSO').to_numpy()]
        sampled_ids_dmso = np.random.choice(dmso_im_ids, size=n_dmso_images_to_sample, replace=False)
        if (num_images_to_sample - n_dmso_images_to_sample) > 0:
            sampled_ids_treated = np.random.choice(treated_im_ids, size=num_images_to_sample - n_dmso_images_to_sample, replace=False)
        else:
            sampled_ids_treated = []
        sampled_ids = list(sampled_ids_dmso) + list(sampled_ids_treated)
    else:
        sampled_ids = np.random.choice(im_ids, size=num_images_to_sample, replace=False)
    rows = [df[df['unique_image_id'] == im_id]['Metadata_row'].values[0] for im_id in sampled_ids]
    cols = [df[df['unique_image_id'] == im_id]['Metadata_column'].values[0] for im_id in sampled_ids]
    fields = [df[df['unique_image_id'] == im_id]['Metadata_field'].values[0] for im_id in sampled_ids]
    ## x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype
    nested_x = []
    nested_y = []
    nested_colors = []
    nested_sizes = []
    images = []
    for i, image_id in enumerate(sampled_ids):
        image = vsr.get_field_image(rows[i], cols[i], fields[i])
        image_table = df[df['unique_image_id'] == image_id]
        x, y, colors, sizes = get_cell_coordinates_colors_sizes(image_table, min_size, celltypes, celltype_color_dict, distinguish_viable=distinguish_viable)
        nested_x.append(x)
        nested_y.append(y)
        nested_colors.append(colors)
        nested_sizes.append(sizes)
        images.append(image)
    fig, ax = plt.subplots(nrows=num_images_to_sample, ncols=2, figsize=(20,10*num_images_to_sample))
    for i in range(num_images_to_sample):
        x_per_celltype = nested_x[i]
        y_per_celltype = nested_y[i]
        celltype_colors = nested_colors[i]
        sizes_per_celltype = nested_sizes[i]
        im = images[i]
        im_row = df[df['unique_image_id'] == sampled_ids[i]]
        treatment = im_row['Drug'].values[0]
        ax_im_title = 'Well {} Field {} Cellprofiler output image'.format(nums_to_well_string(rows[i], cols[i]), fields[i])
        ax_im_title += '\n' + treatment
        if treatment != 'DMSO':
            ax_im_title += ' {} uM'.format(round(im_row['Concentration'].values[0], 5))
        ax_scatter_title = 'Well {} Field {} cell classifications'.format(nums_to_well_string(rows[i], cols[i]), fields[i])
        ax[i,0].imshow(im[y_min:y_max,x_min:x_max])
        if celltypes is None:
            ax[i,1].scatter(x, y, color='white', edgecolor='black', sizes=sizes)
        else:
            for x, y, color, c_sizes in zip(x_per_celltype, y_per_celltype, celltype_colors, sizes_per_celltype):
                ax[i,1].scatter(x, y, color=color, edgecolor='black', sizes=c_sizes)
        ax[i,1].set_xlim(left=x_min, right=x_max)
        ax[i,1].set_ylim(bottom=y_min, top=y_max)
        clean_ax(ax[i,0])
        clean_ax(ax[i,1])
        ax[i,0].set_title(ax_im_title)
        ax[i,1].set_title(ax_scatter_title)
    fig.tight_layout()
    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)
    return

def add_ticks_to_plate_image(in_ax, n_rows=16, n_cols=24):
    col_ticks = list(range(n_cols))
    col_ticklabels = [str(i+1) for i in col_ticks]
    row_ticks = list(range(n_rows))
    row_ticklabels = [ascii_uppercase[i] for i in row_ticks]
    in_ax.set_xticks(col_ticks)
    in_ax.set_xticklabels(col_ticklabels)
    in_ax.set_yticks(row_ticks)
    in_ax.set_yticklabels(row_ticks)
    return in_ax

def plot_cellnumbes_on_ax(in_df, ax, n_rows=16, n_cols=24):
    use_mat = np.zeros((n_rows,n_cols))
    for r in range(n_rows+1):
        for c in range(1, n_cols+1):
            well_mask = (in_df['Metadata_row'] == r).to_numpy() & (in_df['Metadata_column'] == c).to_numpy()
            use_mat[r-1,c-1] = np.sum(well_mask)
    sh = ax.imshow(use_mat, cmap='coolwarm')
    ax = add_ticks_to_plate_image(ax, n_rows=n_rows, n_cols=n_cols)
    return ax, sh
    