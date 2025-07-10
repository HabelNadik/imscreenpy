import argparse
from os.path import join, isfile, abspath, dirname
import sqlite3
import pandas as pd
import numpy as np

## local sub-modules
from imscreenpy.db_df_processing import db_functions, df_functions
from imscreenpy.qc import qc_functions
from imscreenpy.celltype_classification import celltype_detection_manager as celltype_manager
from imscreenpy.viability_classification import viability_manager
from imscreenpy import misc
from imscreenpy import hit_reporting
from imscreenpy import config


def print_fancy_message(message):
    full_length = len(message) + 22 
    borderline = '#' * full_length
    middle = '#' * 10
    middle += ' ' + message + ' '
    middle += '#' * 10
    full_message = borderline + '\n' + middle + '\n' + borderline
    print(full_message, flush=True)
    return

############################################################
############################################################
############# FUNCTIONS FOR ANNOTATION PROCESSING



def get_segmentation_folder(plate_id, annotation, corrected_segmentation=False, cfg=None):
    if cfg is None:
        cfg = config.Config(None)
    sub_table = annotation[annotation[cfg.an_plate_id_col] == plate_id]
    if not corrected_segmentation:
        return sub_table[cfg.an_seg_path_col].values[0]
    else:
        return sub_table[cfg.an_corr_seg_path_col].values[0]

def get_plates(patient_id, annotation, cfg=None, use_experiment_col=False):
    if cfg is None:
        cfg = config.Config(None)
    if use_experiment_col:
        return [int(f) for f in annotation[annotation[cfg.an_experiment_id_col] == patient_id][cfg.an_plate_id_col].values]
    else:
        return [int(f) for f in annotation[annotation[cfg.an_patient_id_col] == patient_id][cfg.an_plate_id_col].values]


def get_target(patient_id, annotation, cfg=None):
    if cfg is None:
        cfg = config.Config(None)
    return annotation[annotation[cfg.an_patient_id_col] == patient_id][cfg.an_target_col].values[0]


def make_target_column(cell_table, marker_columns, target_annotation):
    """
    Generate a target column for an input dataframe with single-cells per row

    Arguments:
    cell_table: pandas dataframe - table with single cells as rows
    marker_columns: list of str - list of marker columns to use for target generation
    target_annotation: str - string that describes the target column to generate. Can be a single marker, a combination of markers, or a special string like 'tripleneg'
    
    """
    target_bool = np.ones((cell_table.shape[0])).astype(bool)
    if 'tripleneg' in target_annotation.lower():
        for col in marker_columns:
            target_bool = target_bool & ~cell_table[col].values.astype(bool)
    elif ('pos' in target_annotation) or ('neg' in target_annotation):
        pos_markers = []
        neg_markers = []
        position_marker = 0
        for i in range(len(target_annotation)):
            if target_annotation[i:i+3] == 'pos':
                marker = target_annotation[position_marker:i]
                pos_markers.append(marker)
                position_marker = i+3
            elif target_annotation[i:i+3] == 'neg':
                marker = target_annotation[position_marker:i]
                neg_markers.append(marker)
                position_marker = i+3
        for marker in pos_markers:
            target_bool = target_bool.astype(bool) & cell_table[marker].values.astype(bool)
        for marker in neg_markers:
            target_bool = target_bool.astype(bool) & ~cell_table[marker].values.astype(bool)
    else:
        return cell_table, target_annotation
    ass_dict = {target_annotation : target_bool}
    cell_table = cell_table.assign(**ass_dict)
    return cell_table, target_annotation

def get_transferlist(patient_id, annotation, annotation_folder, cfg=None, from_plate_id=False):
    if cfg is None:
        cfg = config.Config(None)
    if from_plate_id:
        plate_annotation = annotation[annotation[cfg.an_plate_id_col] == patient_id]
        transferlist_filename = plate_annotation[cfg.an_transferlist_col].values[0]
    else:
        patient_annotation = annotation[annotation[cfg.an_patient_id_col] == patient_id]
        transferlist_filename = patient_annotation[cfg.an_transferlist_col].values[0]
    if not transferlist_filename.endswith('.xls'):
        if transferlist_filename.startswith('2023'):
            transferlist = pd.read_csv(join(annotation_folder, transferlist_filename), encoding='utf-16', delimiter='\t')
        else:
            transferlist = pd.read_csv(join(annotation_folder, transferlist_filename), encoding='latin-1', delimiter='\t')
    else:
        transfer_path = join(annotation_folder, transferlist_filename)
        transfer_path_new = transfer_path.replace('.xls', '.csv')
        platename = patient_annotation[cfg.an_plate_id_col].values[0]
        transferlist = misc.excel_annotation_to_transferlist(join(annotation_folder, transferlist_filename), transfer_path_new, target_sheet=1, concentration_sheet=2, dest_plate_name=platename)
    print('Loaded transferlist of shape {} from path {}'.format(transferlist.shape, join(annotation_folder, transferlist_filename)))
    print('Columns are {}'.format(list(transferlist.columns)))
    return transferlist

############################################################
############################################################
######### SOME TABLE AND DATABASE PROCESSING ###############

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
    all_cells_all = np.zeros(num_unique_wells)
    wells = []
    rows = []
    columns = []
    compounds = []
    doses = []
    plates = []
    if 'Viable' in full_table.columns:
        via_table_mode = 2
        print('Using column name Viable to get viable cell populations per well')
    elif 'DEAD' in full_table.columns:
        via_table_mode = 1
        print('Using column name DEAD to get viable cell populations per well')
    else:
        via_table_mode = 0
        print('No viability column present. assuming all cells are viable')
    for i in range(num_unique_wells):
        plate = unique_well_plate_list[i][0]
        well = unique_well_plate_list[i][1]
        well_table = full_table[(full_table['Metadata_Well'] == well) & (full_table['Plate'] == plate)]
        if via_table_mode == 2:
            via_well_table = well_table[well_table['Viable'].values.astype(bool)]
        elif via_table_mode == 1:
            via_well_table	= well_table[~well_table['DEAD'].values.astype(bool)]
        else:
            via_well_table = well_table
        viable_cells_all[i] = via_well_table.shape[0]
        all_cells_all[i] = well_table.shape[0]
        for k in range(len(markers)):
            viable_cells_per_celltype[i,k] = np.sum(via_well_table[markers[k]])
        rows.append(well_table['Metadata_row'].values[0])
        columns.append(well_table['Metadata_column'].values[0])
        plates.append(plate)
        wells.append(well)
        compounds.append(well_table['Drug'].values[0])
        doses.append(well_table['Concentration'].values[0])
    df_dict = {'Plate' : plates, 'Metadata_Well' : wells, 'Drug' : compounds,  'Concentration' : doses, \
        'Metadata_row' : rows, 'Metadata_column' : columns, 'NumberOfCells' : viable_cells_all, 'NumberOfAllCell' : all_cells_all}
    for k in range(len(markers)):
        marker = markers[k]
        df_dict['NumberOfCells_{}'.format(marker)] =  viable_cells_per_celltype[:,k]
    df = pd.DataFrame(df_dict)
    return df


############################################################
############################################################
######### APPLICATION LOGIC TO GET HIT REPORT


def preprocess_cp_db(plate, target_connection, target_fluorophores, output_folder, segmentation_folder,\
                      cfg, reduction_columns=None, exclude_wells_morph=True, relaxed_image_qc=False, no_qc_exclusion=False):
    """
    Create two clean dataframes from the data, one with features and one with meta information
    arguments:
    target_connection: sqlite3 connection object - contains the relevant cellprofiler data
    target_fluorophores: all fluorophores measured in the target experiment
    output_folder: output folder
    """
    im_qc_plot_filepath = join(output_folder, '{}_image_qc.png'.format(plate))
    im_outlier_well_df = qc_functions.do_image_qc(target_connection, target_fluorophores + ['DNA'], im_qc_plot_filepath, cfg, be_relaxed=relaxed_image_qc,\
                                                   ignore_cellnumbers=relaxed_image_qc)
    im_outlier_wells = list(im_outlier_well_df[cfg.im_well_cpc].unique())
    if no_qc_exclusion:
        im_outlier_wells = []
        exclude_wells_morph = False
    print_fancy_message('Done with imaging qc for plate {}'.format(plate))
    ## we avoid: cell markers on the nucleus (nuclei + GFP|PE|APC), cell location parameters, and cell moments
    exclusion_column_patterns=[['index'], ['nuclei', 'APC'], ['nuclei', 'PE'], ['nuclei', 'GFP'], ['cells_dist', 'Location'],\
         ['cells_dist', 'Moment'], ['cells_dist', 'Tensor'], ['cells_dist', 'Displacement'], ['cells_dist', 'Quartile'], ['cells_dist', 'Integrated']]
    full_dataframe = db_functions.full_df_from_conn(target_connection, cfg.db_cell_table_name, exclusion_wells=im_outlier_wells, exclusion_column_patterns=exclusion_column_patterns, column_subset=reduction_columns)
    print_fancy_message('Loaded full dataframe {} with shape {}'.format(plate, full_dataframe.shape))
    id_df_model_add_columns = cfg.get_expression_model_columns() + cfg.get_viability_model_columns()
    id_df_model_add_columns = list(set(id_df_model_add_columns))
    if no_qc_exclusion:
        feature_df, id_df = df_functions.split_dataframe(full_dataframe, cfg, id_df_add_columns=id_df_model_add_columns)
        return feature_df, id_df
    else:
        feature_df, id_df = qc_functions.do_morphology_qc(full_dataframe, segmentation_folder, output_folder, exclude_wells_morph=exclude_wells_morph, cfg=cfg, id_df_add_columns=id_df_model_add_columns)
        return feature_df, id_df, im_outlier_well_df


def run_preprocessing_qc_workflow(plate_id, target_connection, output_folder, segmentation_folder, cfg, target_fluorophores,\
                                   checkpoint, transferlist, exclude_wells_morph=False, relaxed_image_qc=False, is_last=False, return_qc_df=False, no_well_exclusion=False):
    feature_df_pre_checkpoint_path = join(output_folder, '{}_features_preprocessing.csv'.format(plate_id))
    qc_checkpoint_path = join(output_folder, '{}_image_qc_excluded_wells.csv'.format(plate_id))
    if is_last:
        id_df_pre_checkpoint_path = join(output_folder, '{}_id_df.csv'.format(plate_id))    
    else:
        id_df_pre_checkpoint_path = join(output_folder, '{}_id_df_preprocessing.csv'.format(plate_id))
    if checkpoint and isfile(id_df_pre_checkpoint_path) and isfile(feature_df_pre_checkpoint_path):
        feature_df = pd.read_csv(feature_df_pre_checkpoint_path)
        id_df = pd.read_csv(id_df_pre_checkpoint_path)
        print_fancy_message('Checkpoint Loading: Loading pre feature and id df')
        return feature_df, id_df
    else:
        target_column_list = cfg.get_all_set_columns()#get_cell_target_columns(target_fluorophores, cfg)
        feature_df, id_df, well_qc_df = preprocess_cp_db(int(plate_id), target_connection, target_fluorophores, output_folder,\
                segmentation_folder, cfg=cfg, reduction_columns=target_column_list, exclude_wells_morph=exclude_wells_morph, relaxed_image_qc=relaxed_image_qc, no_qc_exclusion=no_well_exclusion)
        id_df = misc.add_treatments(id_df, plate_id, transferlist)[0]
        print_fancy_message('Set up feature and info dataframes')
        if checkpoint:
            feature_df.to_csv(feature_df_pre_checkpoint_path)
            id_df.to_csv(id_df_pre_checkpoint_path)
            well_qc_df.to_csv(qc_checkpoint_path, index=False)
            print_fancy_message('Checkpointing: Saving pre feature and id df')
        if return_qc_df:
            return feature_df, id_df, well_qc_df
        return feature_df, id_df
    

def run_celltype_processing_workflow(feature_df, id_df, target_fluorophores, celltypes, segmentation_folder, output_folder, plate_id, cfg, expression_model_name, annotation, checkpoint, is_last=False):
    feature_df_celltypes_checkpoint_path = join(output_folder, '{}_features_celltypes.csv'.format(plate_id))
    if is_last:
        id_df_celltypes_checkpoint_path = join(output_folder, '{}_id_df.csv'.format(plate_id))
    else:
        id_df_celltypes_checkpoint_path = join(output_folder, '{}_id_df_celltypes.csv'.format(plate_id))
    if checkpoint and isfile(feature_df_celltypes_checkpoint_path) and isfile(id_df_celltypes_checkpoint_path):
        feature_df = pd.read_csv(feature_df_celltypes_checkpoint_path)
        id_df = pd.read_csv(id_df_celltypes_checkpoint_path)
        print_fancy_message('Checkpoint Loading: Loading feature and id df with celltypes')
    else:
        id_df = celltype_manager.add_celltype_prediction(feature_df, id_df, target_fluorophores, celltypes, segmentation_folder, output_folder, plate_id, cfg, expression_model_name, annotation)
        feature_df = df_functions.remove_cell_related_columns(feature_df)
        if checkpoint:
            feature_df.to_csv(feature_df_celltypes_checkpoint_path)
            id_df.to_csv(id_df_celltypes_checkpoint_path)
            print_fancy_message('Checkpointing: Saving reduced feature df and id df with celltypes')
    return feature_df, id_df


def pre_hit_workflow(target_connection, annotation, output_folder,\
     transferlist, plate_id, cfg, checkpoints=False, exclude_wells_morph=True, relaxed_image_qc=False, no_well_exclusion=False):
    ## getting required info from config
    expression_model_name = cfg.get_expression_model()
    viability_model_name = cfg.get_viability_model()
    target_fluorophores, celltypes = misc.get_fluorophores_and_celltypes(plate_id, annotation, cfg)
    segmentation_folder = get_segmentation_folder(plate_id, annotation, cfg=cfg)
    ### setting up checkpoint paths, depending on which models are used
    ## preprocessing dfs are always saved when checkpointing is enabled
    feature_df_pre_checkpoint_path = join(output_folder, '{}_features_preprocessing.csv'.format(plate_id))
    id_df_pre_checkpoint_path = join(output_folder, '{}_id_df_preprocessing.csv'.format(plate_id))
    ## last id df is also always saved
    last_df_checkpoint_path = join(output_folder, '{}_id_df.csv'.format(plate_id))
    #### set paths for checkpointing based on whether models are selected
    ## viability prediction is optional
    if (not (cfg.get_viability_model() is None)) and (not (cfg.get_viability_model().lower() == 'none')):
        id_df_viabiltiy_checkpoint_path = join(output_folder, '{}_id_df.csv'.format(plate_id))
        last_step = 'viability'
        id_df_viabiltiy_checkpoint_path = last_df_checkpoint_path
    else:
        last_step = 'celltypes'
        id_df_viabiltiy_checkpoint_path = None
    ## celltype prediction is optional
    if (not (cfg.get_expression_model() is None)) and (not (cfg.get_expression_model().lower() == 'none')):
        feature_df_celltypes_checkpoint_path = join(output_folder, '{}_features_celltypes.csv'.format(plate_id))
        id_df_celltypes_checkpoint_path = join(output_folder, '{}_id_df_celltypes.csv'.format(plate_id))
        if last_step == 'celltypes':
            id_df_celltypes_checkpoint_path = last_df_checkpoint_path
    else:
        if last_step == 'celltypes':
            last_step = 'preprocessing'
            feature_df_celltypes_checkpoint_path = ''
            id_df_celltypes_checkpoint_path = ''
        else:
            last_step = 'viability'
    #### load dataframes from checkpoints if they exist
    ## if final df is present, we just load and return it, this one can be preprocessing, celltypes or viabilities, as set above
    if isfile(last_df_checkpoint_path) and checkpoints:
        print_fancy_message('Final id df for plate {} was already here. Loading it'.format(plate_id))
        id_df = pd.read_csv(last_df_checkpoint_path)
        return id_df
    else: ## otherwise, we check if we have the earlier celltype checkpoint
        if checkpoints and isfile(feature_df_celltypes_checkpoint_path) and isfile(id_df_celltypes_checkpoint_path): ## load celltype dataframes if they are there
            print_fancy_message('Plate {} Checkpoint Loading: Loading feature and id df with celltypes'.format(plate_id))
            feature_df, id_df = run_celltype_processing_workflow(None, None, target_fluorophores, celltypes, segmentation_folder, output_folder, plate_id, cfg, expression_model_name, annotation, checkpoints, is_last=False)
        else: ## otherwise, start with preprocessing df
            feature_df, id_df = run_preprocessing_qc_workflow(plate_id, target_connection, output_folder, segmentation_folder, cfg, target_fluorophores,\
                                                    checkpoints, transferlist, exclude_wells_morph=exclude_wells_morph, relaxed_image_qc=relaxed_image_qc, is_last= last_step == 'preprocessing', no_well_exclusion=no_well_exclusion)
            if last_step == 'preprocessing': # return it if this is the last step
                return id_df
            else:
                feature_df, id_df = run_celltype_processing_workflow(feature_df, id_df, target_fluorophores, celltypes, segmentation_folder, output_folder, plate_id, cfg, expression_model_name, annotation, checkpoints, is_last=False)
    ### if the celltype dataframe is the last one, we return it now
    if last_step == 'celltypes':
        return id_df
    else: ## otherwise we generate the viability dataframe
        id_df = viability_manager.add_viability_prediction(feature_df, id_df, target_connection, viability_model_name, annotation, cfg, plate_id, output_folder)
        id_df.to_csv(id_df_viabiltiy_checkpoint_path)
        return id_df

def full_table_from_folders(database_paths, plates, transferlist, annotation, output_folder, cfg, checkpoints=False, exclude_wells_morph=True, relaxed_image_qc=False, no_well_exclusion=False, intensity_column_suffix=''):
    id_dfs = []
    for i, db_path in enumerate(database_paths):
        fluos, markers = misc.get_fluorophores_and_celltypes(plates[i], annotation, cfg)
        cfg.set_intensity_columns(fluos, intensity_column_suffix=intensity_column_suffix)
        if ('LiveOrDye' in markers):
            print('Found viability dye in markers. Adding additional feature columns')
            print('Previous number of columns is {}'.format(len(cfg.get_model_feature_columns())))
            cfg.add_feature_columns(join(cfg.cfg_folder, 'nuclei_viability_features.txt'))
            print('New number of columns is {}'.format(len(cfg.get_model_feature_columns())))
        if not isfile(db_path):
            seg_folder = get_segmentation_folder(plates[i], annotation, cfg=cfg)
            db_functions.combine_databases(seg_folder, cfg=cfg, output_db='benchmark_db.db', reduce_table_width=False)
        conn = sqlite3.connect(db_path)
        id_df = pre_hit_workflow(conn, annotation, output_folder, transferlist, plates[i], cfg, checkpoints=checkpoints, exclude_wells_morph=exclude_wells_morph, relaxed_image_qc=relaxed_image_qc, no_well_exclusion=no_well_exclusion)
        id_df = id_df.assign(object_index=np.arange(id_df.shape[0]))
        id_df = id_df.set_index('object_index')
        #id_df = id_df.reindex(index=np.arange(id_df.shape[0]))
        id_dfs.append(id_df)
    if len(id_dfs) == 1:
        out = id_dfs[0]
    else:
        common_columns = set(list(id_dfs[0].columns)).intersection(*[set(list(df.columns)) for df in id_dfs[1:]])
        out = pd.concat([f[list(common_columns)] for f in id_dfs])
    return out

def write_report_plots(well_viability_table, output_folder, viability_tablename, plates_prefix, target):
    report_savename_template = '{}_{}_inhibition_report.png'
    pos_ctrl_savename_template = '{}_{}_pos_ctrl.png'
    pos_ctrl_savepath = join(output_folder, pos_ctrl_savename_template.format(plates_prefix, target))
    report_savename = report_savename_template.format(plates_prefix, target)
    report_savepath = join(output_folder, report_savename)
    ## generate report
    inh_report = hit_reporting.LDTable_InhibitionReport(well_viability_table)
    inh_report.pos_ctrl_mini_plot(target, savename=pos_ctrl_savepath)
    inh_report.pharmacoscopy_style_plot(target, report_savename.split('.')[0], savetitle=report_savepath, add_scatter=False, plot_all_bars=True)
    return

def run_main(patient_id, annotation, transferlist, db_filename, output_folder, cfg, checkpoints=False, exclude_wells_morph=True, relaxed_image_qc=False, use_plate_id=False, no_well_exclusion=False):
    ### generate full table of viable cells
    if use_plate_id:
        plates = [patient_id]
    else:
        plates = get_plates(patient_id, annotation, cfg=cfg)
    print('Processing the following plates for patient id {}'.format(patient_id))
    print(plates)
    segmentation_folders = [get_segmentation_folder(plate, annotation, cfg=cfg) for plate in plates]
    print('Generating paths from the following segmentation folders')
    print(segmentation_folders)
    db_paths = [join(seg_folder, db_filename) for seg_folder in segmentation_folders]
    
    full_table = full_table_from_folders(db_paths, plates, transferlist, annotation, output_folder, cfg, checkpoints=checkpoints, exclude_wells_morph=exclude_wells_morph, relaxed_image_qc=relaxed_image_qc, no_well_exclusion=no_well_exclusion)
    ## store table
    fst_name = plates[0]
    if len(plates) > 1:
        snd_name = plates[1]
        plates_prefix = str(fst_name) + '-' + str(snd_name)[len(str(snd_name))-2:]
    else:
        plates_prefix = str(fst_name)
    full_tablename = plates_prefix + '_full.csv'
    viability_tablename = plates_prefix + '_viabilities.csv'
    ## get variables for storage and hit reporting
    target = get_target(patient_id, annotation, cfg=cfg)
    if (cfg.class_assignment_dict is None):
        _, celltypes = misc.get_fluorophores_and_celltypes(plates[0], annotation, cfg)
    else:
        celltypes = list(cfg.class_assignment_dict.values())
    if (target.lower() == 'none') or (target is None):
        target = celltypes
    if not (target in full_table.columns):
        full_table, _ = make_target_column(full_table, celltypes, target)
    full_table.to_csv(join(output_folder, full_tablename))
    if isinstance(target, str):
        viability_table_populations = celltypes
        if not (target in viability_table_populations):
            viability_table_populations.append(target)
    well_viability_table = full_table_to_well_viability_table(full_table, viability_table_populations)
    well_viability_table.to_csv(join(output_folder, viability_tablename))
    if isinstance(target, str):
        write_report_plots(well_viability_table, output_folder, viability_tablename, plates_prefix, target)
    else:
        for population in target:
            write_report_plots(well_viability_table, output_folder, viability_tablename, plates_prefix, population)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ExperimentID', help='internal id of experiment: may be a patient, sample or other entity', type=str)
    parser.add_argument('output_folder', help='Folder to write results to', type=str, nargs='?')
    parser.add_argument('--annotation_folder', help='folder with image files to process', dest='annotation_folder', type=str)
    parser.add_argument('--expression_model', help='Model for predicting cell types or marker expression', dest='expression_model', type=str)
    parser.add_argument('--viability_model', help='Viability model to use', dest='viability_model', type=str)
    parser.add_argument('--checkpoints', help='Whether to store data from certain checkpoitns and load it again to continue', action='store_true', default=True, dest='checkpoints')
    parser.add_argument('--skip_reports', help='whether to skip the creation of the full inhibition report', action='store_true', default=False, dest='skip_reports')
    parser.add_argument('--no_morphology_well_exclusion', help='whether to apply a threshold of cells that pass qc to exclude wells', action='store_true', default=False, dest='no_morphology_well_exclusion')
    parser.add_argument('--relaxed_image_qc', help='do image qc with 4 std.devs instead of 3', dest='relaxed_image_qc', action='store_true', default=False)
    parser.add_argument('--config_prefix', dest='config_prefix', help='the prefix of the config files for this project', type=str, default='aml_')
    parser.add_argument('--write_latent', dest='write_latent', action='store_true', default=False, help='whether to also write latent representations of model outputs when using deep learning models')
    parser.add_argument('--single_plate_id', dest='single_plate_id', action='store_true', default=False, help='whether to use the plate id as the experiment id and thus only process one plate. This is useful for debugging and testing')

    ### additional important variables
    annotation_filename = 'annotations.xls'
    full_db_name = 'benchmark_db.db'


    args = parser.parse_args()
    exp_id = args.ExperimentID
    annotation_folder = args.annotation_folder
    out_folder = args.output_folder
    expression_model = args.expression_model
    viability_model = args.viability_model
    morph_ex = not args.no_morphology_well_exclusion
    relax_image_qc = args.relaxed_image_qc
    if relax_image_qc:
        im_qc_string = 'relaxed'
    else:
        im_qc_string = 'strict'
    print('Image QC is set to {}'.format(im_qc_string))
    print('Morphology based well exclusion is set to {}'.format(morph_ex))    
    
    print('expression model is {}'.format(expression_model), flush=True)
    if isinstance(out_folder, list) and (len(out_folder) > 0):
        out_folder = out_folder[0]

    annotation_path = join(join(annotation_folder, 'sample_annotations'), annotation_filename)
    print('Setting up config')
    cfg = config.Config(None, cfg_prefix=args.config_prefix)
    cfg.set_expression_model(expression_model)
    cfg.set_viability_model(viability_model)
    cfg.set_predict_latent(args.write_latent)
    #cfg.set_intensity_columns()
    print('Set up config. Using the following feature columns')
    print(cfg.get_model_feature_columns())
    annotation = pd.read_excel(annotation_path, sheet_name=0)
    drug_an_folder = join(annotation_folder, 'drug_annotations')
    transferlist = get_transferlist(exp_id, annotation, join(drug_an_folder, 'transferlists'), cfg=cfg, from_plate_id=args.single_plate_id)
    print('Loaded transferlist of shape {}'.format(transferlist.shape))
    
    
    run_main(exp_id, annotation, transferlist, full_db_name, out_folder, cfg, checkpoints=args.checkpoints , exclude_wells_morph=morph_ex, relaxed_image_qc=relax_image_qc, use_plate_id=args.single_plate_id)
