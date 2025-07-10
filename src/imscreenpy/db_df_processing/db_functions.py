import argparse
from timeit import default_timer as timer
from os.path import isfile, isdir, join, dirname, abspath
from os import listdir
import sys

import h5py
import sqlite3

sys.path.append('..')

import numpy as np
import pandas as pd

from ..misc import get_feature_list, print_time, make_unique_row_col_field_ids
from .df_functions import match_df

row_cpc = "Metadata_row"
col_cpc = "Metadata_column"
field_cpc = "Metadata_field"
well_cpc = 'Metadata_Well'
un_id_cpc = 'unique_image_id'


def make_query_string(columns, table_name, exclusion_wells=None, exclusion_well_column=None):
    column_string = ''
    processed_columns = []
    for col in columns:
        if not (col in processed_columns):
            column_string += col + ', '
        processed_columns.append(col)
    column_string = column_string[:len(column_string)-2]
    if (exclusion_wells is None) or (len(exclusion_wells) == 0):
        query_string = "SELECT {} FROM {};".format(column_string, table_name)
    else:
        ##### concatenate exclusion wells to generate list
        exclusion_well_string = ""
        for well in exclusion_wells:
            exclusion_well_string += "\'" + well + "\', "
        exclusion_well_string = exclusion_well_string[:len(exclusion_well_string)-2]
        query_string = "SELECT {} FROM {} WHERE NOT ({} IN ({}));".format(column_string, table_name, exclusion_well_column, exclusion_well_string)
    return query_string


def get_all_possible_columns(conn, target_table):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM {};".format(target_table))
    des = cursor.description
    columns = [f[0] for f in des]
    return columns

def exclude_columns(columns, exclusion_column_patterns):
    out_cols = []
    for col in columns:
        include = True
        for ex_pattern in exclusion_column_patterns:
            if len(ex_pattern) == 1:
                if ex_pattern[0] in col:
                    include = False
            elif len(ex_pattern) == 2:
                if (ex_pattern[0] in col) and (ex_pattern[1] in col):
                    include = False
        if include:
            out_cols.append(col)
    print('Column exclusion: Reduced from {} to {}'.format(len(columns), len(out_cols)), flush=True)
    return out_cols

def full_df_from_conn(conn, table_name, exclusion_wells=None, column_subset=None, exclusion_well_column='Metadata_Well', exclusion_column_patterns=[['index']]):
    if column_subset is None:
        all_cols = get_all_possible_columns(conn, table_name)
        use_cols = exclude_columns(all_cols, exclusion_column_patterns)
    else:
        use_cols = column_subset
    query_string = make_query_string(use_cols, table_name, exclusion_wells=exclusion_wells, exclusion_well_column=exclusion_well_column)    
    df = pd.read_sql_query(query_string, conn)
    return df


def double_check_columns(cursor, columns_to_use, table_name, exclude_metadata_cols=False):
    if exclude_metadata_cols:
        columns_to_use = [f for f in columns_to_use if (not ('Metadata' in f)) and (not (f == 'unique_image_id'))]
    res = cursor.execute('SELECT * FROM {}'.format(table_name))
    column_names_here = [f[0] for f in res.description]
    found_columns = [f for f in column_names_here if f in (columns_to_use)]
    columns_not_found = [f for f in columns_to_use if not (f in column_names_here)]
    columns_not_found = list(set(columns_not_found))
    if len(columns_not_found) > 0:
        columns_not_found.sort()
        print('#######################################################')
        print('WARNING!!!! THE FOLLOWING COLUMNS WERE NOT FOUND IN {}'.format(table_name))
        print(columns_not_found)
        print('#######################################################')
    return found_columns



    

def source_db_to_target(source_conn_path, target_conn, processed_ids, cfg=None, reduce_table_width=True, extract_relationships=False, pre_defined_columns=None):
    """
    Extract a dataframe from source_conn_path and append its contents to the database at target_conn
    """
    
    start_table_load = timer()
    test_conn = sqlite3.connect(source_conn_path)
    cur = test_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = []
    rows = cur.fetchall()
    for row in rows:
        table_names.append(row[0])
    if cfg is None:
        cell_table_name = 'Per_Object'
        im_table_name = 'Per_Image'
    else:
        cell_table_name = cfg.db_cell_table_name
        im_table_name = cfg.db_im_table_name
    obj_table = [f for f in table_names if (cell_table_name in f)][0]
    im_table = [f for f in table_names if (im_table_name in f)][0]
    test_conn.close()
    if extract_relationships:
        rel_table = [f for f in table_names if ('Per_Relationships' in f)][0]
        obj_df, im_df, rel_df = get_dfs_from_db(source_conn_path, cfg=cfg, reduce_table_width=reduce_table_width, extract_relationships=extract_relationships, pre_defined_columns=pre_defined_columns)
    else:
        obj_df, im_df = get_dfs_from_db(source_conn_path, cfg=cfg, reduce_table_width=reduce_table_width, extract_relationships=extract_relationships, pre_defined_columns=pre_defined_columns)
    end_df_assignment = timer()
    #### reduce object and image dataframes if images were processed previously
    proc_id_bool_series_im_df = ~im_df[un_id_cpc].isin(processed_ids)
    num_prev_unused_ids = np.sum(proc_id_bool_series_im_df.values)
    if num_prev_unused_ids < im_df.shape[0]:
        proc_id_bool_series_obj_df = ~obj_df[un_id_cpc].isin(processed_ids)
        print('----------------------')
        print('Found already used ids in current df. Starting reduction')
        print('Reducing by: {} from original shape of {}'.format(im_df.shape[0] - num_prev_unused_ids, im_df.shape[0]))
        obj_df = obj_df[proc_id_bool_series_obj_df]
        im_df = im_df[proc_id_bool_series_im_df]
        print('The following unique ids have been found in the dataframe before')
        print([f for f in np.unique(im_df[un_id_cpc]) if (f in processed_ids)])
        print('Object dataframe now has shape {}'.format(obj_df.shape))
        print('-----------------------')
    unique_ids_unique = [f for f in np.unique(im_df[un_id_cpc].values) if not (f in processed_ids)]
    out_processed_ids = unique_ids_unique
    end_id_processing = timer()
    print('Checked for double occurences and cleaned ids. {} ids processed so far after adding {} ids'.format(len(processed_ids) + len(out_processed_ids), len(unique_ids_unique)))
    print_time(end_df_assignment, end_id_processing)
    obj_df.to_sql(obj_table, target_conn, if_exists='append')
    im_df.to_sql(im_table, target_conn, if_exists='append')
    if extract_relationships:
        rel_df.to_sql(rel_table, target_conn, if_exists='append')
        print('Appended relationship dataframe with the following columns')
        print([f for f in rel_df.columns])
    end_total = timer()
    print('Done')
    print_time(start_table_load, end_total)
    return out_processed_ids
    

def get_dfs_from_db(source_conn_path, cfg=None, reduce_table_width=True, extract_relationships=False, pre_defined_columns=None):
    """
    Read an object df, an image df and optionally a relationship df from a database at source_conn_path
    """
    ## establish connection and get table names
    start_table_load = timer()
    first_conn = sqlite3.connect(source_conn_path)
    cur = first_conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = []
    rows = cur.fetchall()
    ## get tables and prepare processing
    if cfg is None:
        cell_table_name = 'Per_Object'
        im_table_name = 'Per_Image'
    else:
        cell_table_name = cfg.db_cell_table_name
        im_table_name = cfg.db_im_table_name
    all_query_template = "SELECT * FROM {};"
    for row in rows:
        table_names.append(row[0])
    obj_table = [f for f in table_names if (cell_table_name in f)][0]
    im_table = [f for f in table_names if (im_table_name in f)][0]
    ## build query string based on kwargs
    if not (pre_defined_columns is None): ## only query selected columns
        pre_defined_columns = double_check_columns(cur, pre_defined_columns, cell_table_name, exclude_metadata_cols=True)
        if not ('ImageNumber' in pre_defined_columns):
            pre_defined_columns.append('ImageNumber')
            print('Adding ImageNumber to pre-defined columns, because it is required for mapping')
        obj_query = make_query_string(pre_defined_columns, obj_table)
    elif reduce_table_width: ## exclude columns containing certain keywords
        print('Reducing table width by excluding columns containing the following keywords')
        taboo_list = ['Moment', 'IntensityEdge', 'CenterMassIntensity', 'MADIntensity', 'Intensity_MassDisplacement', 'Location_MaxIntensity', 'StdIntensity', 'InertiaTensor']
        print(taboo_list)
        colnames = get_all_possible_columns(first_conn, cell_table_name)
        reduced_colnames = []
        for colname in colnames:
            keep = True
            for tcol in taboo_list:
                if tcol in colname:
                    keep = False
            if keep:
                reduced_colnames.append(colname)
        obj_query = make_query_string(reduced_colnames, obj_table)
        print('Excluded {} columns. Old count was {}. New count is {}'.format(len(colnames) - len(reduced_colnames), len(colnames), len(reduced_colnames)))
    else: ## query all columns
        obj_query = all_query_template.format(obj_table)
    obj_df = pd.read_sql_query(obj_query, first_conn)
    
    if extract_relationships:
        rel_table = [f for f in table_names if ('Per_Relationships' in f)][0]
        rel_query = all_query_template.format(rel_table)
        rel_df = pd.read_sql_query(rel_query, first_conn)
        rel_image_numbers = rel_df['image_number1'].values
        rows_rel = np.zeros(rel_image_numbers.shape, dtype=np.uint16)
        cols_rel = np.zeros(rel_image_numbers.shape, dtype=np.uint16)
        fields_rel = np.zeros(rel_image_numbers.shape, dtype=np.uint16)
        wells_rel = np.chararray(rel_image_numbers.shape, itemsize=3, unicode=True)
        
    #### get row col field ids by image number to add them to the object dataframe / cell table
    image_numbers = obj_df['ImageNumber'].values
    rows = np.zeros(image_numbers.shape, dtype=np.uint16)
    cols = np.zeros(image_numbers.shape, dtype=np.uint16)
    fields = np.zeros(image_numbers.shape, dtype=np.uint16)
    wells = np.chararray(image_numbers.shape, itemsize=3, unicode=True)
    im_df = pd.read_sql_query("SELECT * FROM {}".format(im_table), first_conn)
    end_table_load = timer()
    print('Loaded object table: {} and image table: {}'.format(obj_table, im_table))
    print_time(start_table_load, end_table_load)
    im_df_im_nums = im_df['ImageNumber'].values
    in_rows = im_df['Image_' + row_cpc].values
    in_cols = im_df['Image_' + col_cpc].values
    in_fields = im_df['Image_' + field_cpc].values
    in_wells = im_df['Image_' + well_cpc].values
    #unique_im_ids = np.zeros(image_numbers.shape, dtype=np.object)
    for im_num in np.unique(image_numbers):
        in_mask = image_numbers == im_num
        im_df_mask = im_df_im_nums == im_num
        rows[in_mask] = in_rows[im_df_mask][0]
        cols[in_mask] = in_cols[im_df_mask][0]
        fields[in_mask] = in_fields[im_df_mask][0]
        wells[in_mask] = in_wells[im_df_mask][0]
        if extract_relationships:
            in_rel_mask = rel_image_numbers == im_num
            rows_rel[in_rel_mask] = in_rows[im_df_mask][0]
            cols_rel[in_rel_mask] = in_cols[im_df_mask][0]
            fields_rel[in_rel_mask] = in_fields[im_df_mask][0]
            wells_rel[in_rel_mask] = in_wells[im_df_mask][0]
    print('Creating tables with unique rows: {}, cols:{}, wells {}'.format(np.unique(rows), np.unique(cols), np.unique(wells)))
    ### create unique image ids for easy mapping later - allows to identify duplicates
    unique_obj_df_ids = make_unique_row_col_field_ids(rows, cols, fields) ### for object df
    unique_im_df_ids = make_unique_row_col_field_ids(in_rows, in_cols, in_fields) ### and for the image df
    if extract_relationships:
        unique_rel_df_ids = make_unique_row_col_field_ids(rows_rel, cols_rel, fields_rel)
        rel_df_ass_dict = {row_cpc : rows_rel, col_cpc : cols_rel, field_cpc : fields_rel, un_id_cpc : unique_rel_df_ids, well_cpc : wells_rel}
        rel_df = rel_df.assign(**rel_df_ass_dict)
    ### assign all metadata to the output dataframe and return
    print('Made unique image identities, assigning to dataframe')
    print('Image df: {} ids in total {} are unique. first one is {}'.format(unique_im_df_ids.shape[0], len(list(set(unique_im_df_ids))), unique_im_df_ids[0]))
    end_unique_id_creation = timer()
    print_time(end_table_load, end_unique_id_creation)
    obj_df_ass_dict = {row_cpc : rows, col_cpc : cols, field_cpc : fields, un_id_cpc : unique_obj_df_ids, well_cpc : wells}
    obj_df = obj_df.assign(**obj_df_ass_dict)
    im_df_ass_dict = {un_id_cpc : unique_im_df_ids}
    im_df = im_df.assign(**im_df_ass_dict)
    print('Done with dataframe assignment')
    end_df_assignment = timer()
    print_time(end_unique_id_creation, end_df_assignment)
    first_conn.close()
    if extract_relationships:
        return obj_df, im_df, rel_df
    else:
        return obj_df, im_df

    
def get_paths_to_databases(sup_folder):
    out_paths = []
    subfolders = [join(sup_folder, f) for f in listdir(sup_folder) if isdir(join(sup_folder, f))]
    for sf in subfolders:
        db_files = [f for f in listdir(sf) if isfile(join(sf, f)) and (f.endswith('.db'))]
        if len(db_files) > 0:
            for dbf in db_files:
                out_paths.append(join(sf, dbf))
    return out_paths

    
def combine_databases(folder, cfg=None, output_db='combined_db.db', reduce_table_width=True, extract_relationships=False, pre_defined_columns=None, training_data=False, output_folder=None):
    ### create database
    conn = None
    try:
        conn = sqlite3.connect(join(folder, output_db))
    except sqlite3.OperationalError as e:
        print(e)
        print('Database could not be created because of wrong path. ABORTING!')
        print('Path was:')
        print(join(folder, output_db))
        quit()
    finally:
        if conn:
            conn.close()
    if isfile(join(folder, output_db)):
        db_paths = get_paths_to_databases(folder)
        print('Starting database creation from a total of {} sub databases'.format(len(db_paths)))
        if output_folder is None:
            output_folder = folder
        target_conn = sqlite3.connect(join(output_folder, output_db))
        ### copy contents from first database to output database
        processed_ids = []
        ### proceed with the rest
        for i, db_path in enumerate(db_paths):
            print('reading path {} for index {}'.format(db_path, i))
            if training_data: ## only this case after refactoring of im prep pipeline
                this_dir = dirname(abspath(__file__))
                feature_filename = join(this_dir, 'reduced_feature_list.txt')
                features = get_feature_list(filepath=feature_filename)
                id_columns = ['ObjectNumber', 'nuclei_Number_Object_Number', 'ImageNumber']
                qc_candidate_columns = ['nuclei_AreaShape_Area', 'nuclei_Intensity_MedianIntensity_DNA', 'nuclei_Intensity_MeanIntensity_DNA',\
                 'nuclei_Texture_SumEntropy_DNA_3_00_256', 'nuclei_Texture_Contrast_DNA_3_00_256', 'nuclei_Texture_Correlation_DNA_3_00_256',\
                 'nuclei_Texture_SumAverage_DNA_3_00_256', 'nuclei_Texture_InfoMeas2_DNA_3_00_256']
                qc_columns = [f for f in qc_candidate_columns if (not (f in features))]
                pre_defined_columns = features + id_columns + qc_columns
            elif not (cfg is None):
                if pre_defined_columns is None:
                    pre_defined_columns = []
                pre_defined_columns = pre_defined_columns + cfg.get_all_set_columns()
            elif not (pre_defined_columns is None):
                id_columns = ['ObjectNumber', 'nuclei_Number_Object_Number', 'ImageNumber']
                qc_candidate_columns = ['nuclei_AreaShape_Area', 'nuclei_Intensity_MedianIntensity_DNA', 'nuclei_Intensity_MeanIntensity_DNA',\
                 'nuclei_Texture_SumEntropy_DNA_3_00_256', 'nuclei_Texture_Contrast_DNA_3_00_256', 'nuclei_Texture_Correlation_DNA_3_00_256',\
                 'nuclei_Texture_SumAverage_DNA_3_00_256', 'nuclei_Texture_InfoMeas2_DNA_3_00_256']
                qc_columns = [f for f in qc_candidate_columns if (not (f in pre_defined_columns))]
                pre_defined_columns = pre_defined_columns + id_columns + qc_columns
            processed_ids += source_db_to_target(db_path, target_conn, processed_ids, cfg=cfg, reduce_table_width=reduce_table_width, extract_relationships=extract_relationships, pre_defined_columns=pre_defined_columns)
            print('Done with path')
            #print('Processed ids are')
            #print(processed_ids)
        target_conn.close()
        print('Finalized database processing')
        print('Processed a total of {} unique images'.format(len(processed_ids)))
    else:
        print('NO DB OBJECT CREATED! ABORTING!!')
        return

def create_match_vectors(im_ids, object_numbers):
    num_digits = len(str(np.amax(im_ids)))
    ids = object_numbers * (10**num_digits)
    out = ids + im_ids
    return out

def get_matched_feature_df_with_target_columns(id_df, target_columns, db_path_or_conn, cfg_or_table_name=None, unite_with_id_df=False):
    """
    This function was formerly know as update_id_df_with_columns
    
    """
    ### get dataframe with columns of interest
    id_columns = ['ObjectNumber', 'nuclei_Number_Object_Number', 'ImageNumber', 'unique_image_id', 'Metadata_Well', 'Metadata_row', 'Metadata_column']
    all_columns = target_columns + id_columns
    if isinstance(db_path_or_conn, str) and isfile(db_path_or_conn):
        conn = sqlite3.connect(db_path_or_conn)
    else:
        conn = db_path_or_conn
    if (cfg_or_table_name is None):
        table_name = 'Per_Object'
    elif isinstance(cfg_or_table_name, str):
        table_name = cfg_or_table_name
    else:
        table_name = cfg_or_table_name.db_cell_table_name
    new_df = full_df_from_conn(conn, table_name, column_subset=all_columns)
    print('Got new dataframe with shape {} from database'.format(new_df.shape))
    if not (un_id_cpc in new_df.columns):
        im_id_values =  make_unique_row_col_field_ids(new_df['Metadata_row'].values, new_df['Metadata_column'].values, new_df['Metadata_field'].values)
        new_df = new_df.assign(**{un_id_cpc : im_id_values})
    if not (un_id_cpc in id_df.columns):
        im_id_values =  make_unique_row_col_field_ids(id_df['Metadata_row'].values, id_df['Metadata_column'].values, id_df['Metadata_field'].values)
        id_df = id_df.assign(**{un_id_cpc : im_id_values})
    if isinstance(new_df[un_id_cpc].values[0], str):
        new_df = new_df.assign(**{un_id_cpc : new_df[un_id_cpc].to_numpy().astype(np.uint64)})
    if isinstance(id_df[un_id_cpc].values[0], str):
        id_df = id_df.assign(**{un_id_cpc : id_df[un_id_cpc].to_numpy().astype(np.uint64)})
    ### match image ids to make sure that the matched dataframe sizes are correct
    print('pre matching. first few unique image ids in new df are: {}'.format(new_df[un_id_cpc].unique()[:5]))
    print('pre matching. first few unique image ids in id df are: {}'.format(id_df[un_id_cpc].unique()[:5]))
    unique_target_images = id_df[un_id_cpc].unique()
    new_df = new_df[new_df[un_id_cpc].isin(unique_target_images)]
    print('Matching dataframes with shapes {} for id df and {} for new df'.format(id_df.shape, new_df.shape))
    matched_new_df, _ = match_df(new_df, id_df)
    if unite_with_id_df:
        #add_columns = [f for f in target_columns if (not (f in id_df.columns))]
        #add_df = matched_new_df[add_columns]
        add_dict = dict([(f, matched_new_df[f].to_numpy()) for f in target_columns if (not (f in id_df.columns))])
        out = id_df.assign(**add_dict)#pd.concat([id_df, add_df], axis=1)
        return out
    else:

        return matched_new_df
    

def get_image_df(conn_or_path, fluorophores_of_interest, cfg=None, other_qc_columns=['Image_Count_nuclei'], \
                  intensity_column_prefix='Image_Intensity_MeanIntensity_', metadata_well_column='Image_Metadata_Well',\
                    metadata_column_column='Image_Metadata_column',\
                        metadata_field_column='Image_Metadata_field', metadata_row_column='Image_Metadata_row', db_im_table_name='Per_Image'):
    """
    Get a dataframe containing image-level data for the specified fluorophores of interest

    Parameters
    ----------
    conn_or_path : str or sqlite3.Connection
        Either a path to a sqlite database or a sqlite3.Connection object


    fluorophores_of_interest : list of str
        The fluorophores for which to extract data
    
    cfg : Config (optional)
        An imscreenpy.config.Config object containing metadata columns and other information. If None, columns will be read
        from function arguments

    other_qc_columns : list of str (optional)
        Other columns to include in the output dataframe
    
    intensity_column_prefix : str (optional)
        The prefix of the intensity columns in the database
    
    metadata_well_column : str (optional)
        The name of the well column in the database

    metadata_column_column : str (optional)
        The name of the column column in the database
    
    metadata_field_column : str (optional)
        The name of the field column in the database

    metadata_row_column : str (optional)
        The name of the row column in the database

    db_im_table_name : str (optional)
        The name of the table in the database containing image data
    
    """

    if isinstance(conn_or_path, str):
        conn = sqlite3.connect(conn_or_path)
    else:
        conn = conn_or_path
    
    #intensity_columns = ['Image_Intensity_MeanIntensity_{}'.format(fluo) for fluo in fluorophores_of_interest]
    intensity_columns = [intensity_column_prefix + '{}'.format(fluo) for fluo in fluorophores_of_interest]
    if not (cfg is None):
        metadata_columns = cfg.im_metadata_columns
        im_table_name = cfg.db_im_table_name
    else:
        metadata_columns = [f for f in [metadata_column_column, metadata_field_column, metadata_row_column, metadata_well_column] if not (f is None)]
        im_table_name = db_im_table_name
    target_columns = intensity_columns + metadata_columns
    if not other_qc_columns is None:
        target_columns += other_qc_columns
    query_string = make_query_string(target_columns, im_table_name)
    df = pd.read_sql(query_string, conn)
    if isinstance(conn_or_path, str):
        conn.close()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge databases from CellProfiler into single big database')
    parser.add_argument('input_folder', help='path to images folder', type=str)
    parser.add_argument('--db_out_name', help='Name to give to the output database', type=str, dest='db_out_name', default='combined_db.db')
    parser.add_argument('--keep_all_columns', help='Do not reduce the dataframe width. Might lead to SQL errors', action='store_true', dest='keep_all_columns', default=False)
    parser.add_argument('--get_relationships', help='Also add the Per_Relationships table to the target database', dest='get_relationships', action='store_true', default=False)
    parser.add_argument('--training', help='Whether to create training data for the multi-bayes model', action='store_true', dest='training', default=False)
    parser.add_argument('--output_folder', dest='output_folder', default=None, help='Output folder to write database to')


    args = parser.parse_args()
    in_folder = args.input_folder
    db_name = args.db_out_name
    do_reduction = not args.keep_all_columns
    do_relationships = args.get_relationships
    training = args.training
    out_folder = args.output_folder

    if training:
        print('Creating databases for bot')

    combine_databases(in_folder, output_db=db_name, extract_relationships=do_relationships, \
        output_folder=out_folder, reduce_table_width=do_reduction, training_data=training)
