from timeit import default_timer as timer

import pandas as pd
import numpy as np

from ..misc import print_time, make_unique_row_col_field_ids
from ..config import Config

def update_id_df_with_columns(original_id_df, new_df, columns_to_add=[]):
    ### reduce new df with contained images
    print('Input DF shapes are {} and {}'.format(original_id_df.shape, new_df.shape))
    original_id_df = original_id_df.sort_values(['unique_image_id', 'ObjectNumber'], kind='stable')
    new_df = new_df.sort_values(['unique_image_id', 'ObjectNumber'], kind='stable')
    print('Making sure that types match for image id columns')
    im_ids_orig_converted = np.array([int(im_id) for im_id in original_id_df['unique_image_id'].values])
    im_ids_new_converted = np.array([int(im_id) for im_id in new_df['unique_image_id'].values])
    start_assignment = timer()
    orig_im_ids = np.unique(im_ids_orig_converted)
    new_im_ids = np.unique(im_ids_new_converted)
    print('Checking if new image ids are in old ones. New mask has {} ids and old mask has {}. First one in old is {}'.format(len(list(new_im_ids)), len(list(orig_im_ids)), orig_im_ids[0]))
    print('First im id in new df is {}'.format(new_im_ids[0]))
    assignment_mask = np.zeros(new_df.shape[0]).astype(bool)
    print('Double checking before for loop. New df shape is {}'.format(new_df.shape))
    test_index = 0
    for im_id in orig_im_ids:
        im_mask_new = im_ids_new_converted == im_id
        im_mask_orig = im_ids_orig_converted == im_id
        orig_im_object_numbers = list(original_id_df[im_mask_orig]['ObjectNumber'].values)
        obj_num_mask = (new_df['ObjectNumber'].isin(orig_im_object_numbers)).to_numpy().astype(bool)
        if obj_num_mask.shape[0] == 0:
            print('object mask has length zero at index {}!!! shape is {}'.format(test_index, obj_num_mask.shape), flush=True)
        position_mask_new_to_old = im_mask_new & obj_num_mask
        assignment_mask = assignment_mask | position_mask_new_to_old
        test_index += 1
    print('Assignment mask has {} true entries'.format(np.sum(assignment_mask)))
    new_df = new_df[assignment_mask]
    print('Attempting to match new df with shape {} and old df with shape {}'.format(new_df.shape, original_id_df.shape), flush=True)
    print('Double checcking if everything is matching .....')
    new_obj_numbers = new_df['ObjectNumber'].to_numpy()
    old_obj_numbers = original_id_df['ObjectNumber'].to_numpy()
    if (new_obj_numbers == old_obj_numbers).all():
        print('Everything correct!! Continuing.', flush=True)
        ass_dict = dict()
        for col in columns_to_add:
            ass_dict[col] = new_df[col].values
        original_id_df = original_id_df.assign(**ass_dict)
        end_assignment = timer()
        print('Done with assignment of bounding boxes!')
        print_time(start_assignment, end_assignment)
        return original_id_df
    else:
        print('Mismatch found!! ABORTING!!', flush=True)
        quit()

def split_dataframe(input_df, cfg, id_df_add_columns=None):
    all_columns = [f for f in input_df.columns]
    if cfg is None:
        cfg = Config(None)
    id_columns = cfg.get_id_columns()
    if not (id_df_add_columns is None):
        id_columns += id_df_add_columns
    #forbidden_columns = [f for f in all_columns if (('Metadata' in f) or (f in id_columns) or ('Location' in f) or ('_Center_' in f) or ('GFP' in f) or ('Children' in f) or ('BoundingBox' in f) or ('cells_dist' in f) or ('APC' in f) or ('PE' in f))]
    use_columns = [f for f in all_columns if not (f in id_columns)]
    feature_df = input_df[use_columns]
    id_keep_columns = [f for f in id_columns if f in input_df.columns]
    id_df = input_df[id_keep_columns]
    print('Split dataframe. ID DF columns are')
    print([f for f in id_df.columns])
    return feature_df, id_df

def remove_cell_related_columns(input_df):
    cols_to_keep = [f for f in input_df.columns if (not ('cells_dist' in f))]
    return input_df[cols_to_keep]

def full_table_to_well_viability_table(full_table, markers):
    #### determine array size
    plates = list(set(full_table['Plate'].values))
    unique_well_plate_list = []
    for plate in plates:
        plate_table = full_table[full_table['Plate'] == plate]
        wells = list(set(plate_table['Metadata_Well'].values))
        wells.sort()
        for well in wells:
            unique_well_plate_list.append((plate, well))
    num_unique_wells = len(unique_well_plate_list)
    viable_cells_all = np.zeros((num_unique_wells))
    viable_cells_per_celltype = np.zeros((num_unique_wells, len(markers)))
    wells = []
    rows = []
    columns = []
    compounds = []
    doses = []
    plates = []
    for i in range(num_unique_wells):
        plate = unique_well_plate_list[i][0]
        well = unique_well_plate_list[i][1]
        well_table = full_table[(full_table['Metadata_Well'] == well) & (full_table['Plate'] == plate)]
        via_well_table = well_table[well_table['Viable'].values.astype(bool)]
        viable_cells_all[i] = via_well_table.shape[0]
        for k in range(len(markers)):
            viable_cells_per_celltype[i,k] = np.sum(via_well_table[markers[k]])
        rows.append(well_table['Metadata_row'].values[0])
        columns.append(well_table['Metadata_column'].values[0])
        plates.append(plate)
        wells.append(well)
        compounds.append(well_table['Drug'].values[0])
        doses.append(well_table['Concentration'].values[0])
    df_dict = {'Plate' : plates, 'Metadata_Well' : wells, 'Drug' : compounds,  'Concentration' : doses, \
        'Metadata_row' : rows, 'Metadata_column' : columns, 'NumberOfCells' : viable_cells_all}
    for k in range(len(markers)):
        marker = markers[k]
        df_dict['NumberOfCells_{}'.format(marker)] =  viable_cells_per_celltype[:,k]
    df = pd.DataFrame(df_dict)
    return df


def add_normalized_number_to_viability_table(df, column_to_normalize, normalization_target_drug):
    if not ('Condition' in df.columns):
        norm_values = df[df['Drug'] == normalization_target_drug][column_to_normalize].to_numpy()
        all_values = df[column_to_normalize].to_numpy()
        median_val = np.median(norm_values)
        normed_values = all_values / median_val
    else:
        normed_values = np.zeros(df.shape[0])
        all_values = df[column_to_normalize].to_numpy()
        all_conditions = list(set(df['Condition'].values))
        ctrl_mask = (df['Drug'] == normalization_target_drug).to_numpy()
        for condition in all_conditions:
            condition_mask = (df['Condition'] == condition).to_numpy()
            ctrl_vals = all_values[condition_mask & ctrl_mask]
            median_val = np.median(ctrl_vals)
            condition_normed_vals = all_values[condition_mask] / median_val
            normed_values[condition_mask] = condition_normed_vals
    return df.assign(NormalizedValue=normed_values)

def create_match_vectors(im_ids, object_numbers):
    num_digits = len(str(np.amax(im_ids)))
    ids = object_numbers * (10**num_digits)
    out = ids + im_ids
    return out

def match_df(input_df, df_to_match_to):
    df_to_match_to = df_to_match_to.sort_values(by=['unique_image_id', 'ObjectNumber'])
    input_df = input_df.sort_values(by=['unique_image_id', 'ObjectNumber'])
    ass_vec = np.zeros(input_df.shape[0]) # and create a vector for the assignment of the new df to the old one
    ### create unique ids to match with each other
    match_vector_target_df = create_match_vectors(input_df['unique_image_id'].to_numpy().astype(int), input_df['ObjectNumber'].to_numpy())
    match_vector_anchor_df = create_match_vectors(df_to_match_to['unique_image_id'].to_numpy().astype(int), df_to_match_to['ObjectNumber'].to_numpy())
    print('Created match vectors')
    print('First entries for input df and pred df are')
    print(match_vector_target_df[:20])
    print(match_vector_anchor_df[:20])
    ### if the match between both is perfect, we just return the new dataframe and are done
    if match_vector_target_df.shape[0] == match_vector_anchor_df.shape[0]:
        if np.equal(match_vector_target_df, match_vector_anchor_df).all():
            print('Found perfect match of id vectors')
            return input_df, df_to_match_to
    ### if the match is not perfect we match on a per image basis
    ### The order for both dataframes will be the exact same, so we just need to know which objects we should skip
    num_images_processed = 0
    start_batch = timer()
    unique_im_ids = list(set(df_to_match_to['unique_image_id'].values))
    unique_im_ids.sort()
    input_ids = input_df['unique_image_id'].to_numpy().astype(int)
    anchor_ids = df_to_match_to['unique_image_id'].to_numpy().astype(int)
    total_num_matching_ids_found = 0
    for im_id in unique_im_ids:
        ## get the subtable for the image of interest
        im_pos_target = (input_ids == im_id)
        im_pos_anchor = (anchor_ids == im_id)
        im_match_vector_target = match_vector_target_df[im_pos_target]
        im_match_vector_anchor = match_vector_anchor_df[im_pos_anchor]
        ## only do something if the image exists in both tables
        if (im_match_vector_target.shape[0] > 0) and (im_match_vector_anchor.shape[0] > 0):
            if num_images_processed == 0:
                start_if = timer()
            ## if the ids are the exact same, we just do the match for the corresponding image
            if (im_match_vector_target.shape[0] == im_match_vector_anchor.shape[0]) and (im_match_vector_target == im_match_vector_anchor).all():
                ass_vec[im_pos_target] = np.ones(np.sum(im_pos_anchor))
                total_num_matching_ids_found += im_match_vector_anchor.shape[0]
            else:
                if num_images_processed == 0:
                    end_if = timer()
                    #print('Got into if clause')
                    #print_time(start_if, end_if)
                # we iterate over the positions in the matching mask from the target table
                match_array = im_pos_target.reshape(-1,)
                truth_arr_per_image = []
                local_match_positions = np.argwhere(match_array).flatten()
                for match_position, subtable_index in zip(local_match_positions, range(im_match_vector_target.shape[0])): ## go through the parent (nuclei) ids from the new table
                    target_id = im_match_vector_target[subtable_index] ## get the id of the object in the target df
                    ## check if it exists in the anchor df
                    anchor_id_mask = im_match_vector_anchor == target_id
                    if np.sum(anchor_id_mask) == 0: ## it's not there
                        #print('NO MATCH FOUND!!!')
                        ass_vec[match_position] = 0 # so we put a zero 
                        truth_arr_per_image.append(0)
                    else:
                        truth_arr_per_image.append(1) ## otherwise we put a 1
                        total_num_matching_ids_found += 1
                if num_images_processed == 0:
                    end_match = timer()
                    print('Computed matchings')
                    print_time(end_if, end_match)
                #print('Putting viability array of length {} into {} positions afte processing anchor vector of shape {} and target vector of shape {}'.format(len(truth_arr_per_image), np.sum(match_array), im_match_vector_anchor.shape, im_match_vector_target.shape), flush=True)
                ass_vec[match_array] = truth_arr_per_image
        if (num_images_processed % 2000) == 0:
            print('Processed {} images'.format(num_images_processed+1))
            print('Found {} matching ids so far'.format(total_num_matching_ids_found))
            end_batch = timer()
            print_time(start_batch, end_batch)
            start_batch = timer()
        num_images_processed += 1
    print('Done with processing datafreames with id df shape {} and target df shape {}'.format(df_to_match_to.shape, input_df.shape))
    print('Found a total of {} matchings'.format(total_num_matching_ids_found))
    ### after this whole ordeal we should have an assignment vector that allows us to match our target dataframe with the id df
    ## so let's see if this is the case
    reduced_match_vector_pred = match_vector_target_df[ass_vec.astype(bool)]
    if reduced_match_vector_pred.shape[0] != match_vector_anchor_df.shape[0]:
        print('MISMATCH OF VECTOR ENTRIES! NOT GOOD AT ALL!!!!!')
        print('Anchor vector has shape {} and match vector has shape {}'.format(match_vector_anchor_df.shape, reduced_match_vector_pred.shape))
        print('Fraction of matching entries after ignoring order is {}'.format(len([f for f in reduced_match_vector_pred if (f in match_vector_anchor_df)]) / match_vector_anchor_df.shape[0]))
        print('Returning None')
        return None
    if np.equal(reduced_match_vector_pred, match_vector_anchor_df).all():
        print('All good. Returning new datafrmae')
        return input_df[ass_vec.astype(bool)], df_to_match_to
    else:
        print('MISMATCH BETWEEN VECTORS! NOT GOOD AT ALL!!!!!')
        print('Fraction of matching entries with order is {}'.format(np.sum(reduced_match_vector_pred == match_vector_anchor_df) / match_vector_anchor_df.shape[0]))
        print('Fraction of matching entries after ignoring order is {}'.format(len([f for f in reduced_match_vector_pred if (f in match_vector_anchor_df)]) / match_vector_anchor_df.shape[0]))
        print('Returning None')
        return None
    

def add_unique_image_id_column(in_df, cfg=None):
    if cfg is None:
        cfg = Config(None)
    rows = in_df[cfg.row_cpc]
    cols = in_df[cfg.col_cpc]
    fields = in_df[cfg.field_cpc]
    unique_ids = make_unique_row_col_field_ids(rows, cols, fields)
    return in_df.assign(unique_image_id=unique_ids)