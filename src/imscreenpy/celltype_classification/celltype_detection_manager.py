import numpy as np
import pandas as pd
from os.path import join

from ..visualization import expression_control_plotting
from .celltype_detection_models import AML_AAE_celltype_model, NBAAEmodel


def correct_cd3_over_estimation(id_df, colums_to_correct_with=['CD117','CD34', 'CD33', 'CD13', 'CD11a', 'CD10']):
    columns_to_correct = [f for f in id_df.columns if f in colums_to_correct_with]
    corr_arr = id_df[columns_to_correct].to_numpy()
    corr_mask = np.sum(corr_arr, axis=1) > 0
    cd3_corrected = id_df['CD3'].to_numpy()
    cd3_corrected[corr_mask] = 0
    return id_df.assign(CD3=cd3_corrected)

def add_celltype_prediction(feature_df, id_df, fluorophores, celltypes,\
     segmentation_folder, output_folder, plate, cfg, model_name, annotation):
    print('Adding celltype prediction with model {}'.format(model_name), flush=True)
    if 'expression_model_class_name' in cfg.__dict__.keys():
        class_name = cfg.expression_model_class_name
        if class_name == 'NBAAEmodel':
            use_model = NBAAEmodel(annotation, model_name, cfg)
            id_df = use_model.predict_fluorophores(plate, output_folder, id_df)
            savename = join(output_folder, '{}_classification_control.png'.format(plate))
            expression_control_plotting.make_classification_control_figure(id_df, segmentation_folder, savename, cfg)
        elif model_name.lower().startswith('patchpy'):
            model_name = model_name.split('patchpy')[1]
            channel_id_column = 'DAPI_Channel_ID'
            print('Running patchpy prediction', flush=True)
            ### fix dataframe by adding missing columns if necessary
            missing_columns = [f for f in cfg.get_aae_columns() if not (f in id_df.columns)]
            use_model = AML_AAE_celltype_model(annotation, cfg=cfg)
            id_df = use_model.predict_fluorophores(plate, output_folder, id_df, fluorophores, celltypes, model_name)
            if 'CD3' in id_df.columns:
                id_df = correct_cd3_over_estimation(id_df)
            savename = join(output_folder, '{}_expression_classification_control.png'.format(plate))
            intensity_columns = cfg.get_intensity_columns()
            if len(intensity_columns) > len(fluorophores):
                intensity_use_cols = []
                if len([f for f in intensity_columns if f.startswith('cells')]) > 1:
                    for fluo in fluorophores:
                        intensity_use_cols.append([f for f in intensity_columns if f.startswith('cells') and fluo in f][0])
                intensity_columns = intensity_use_cols
            intensity_column_assignment_dict = dict()
            for int_col in intensity_columns:
                intensity_column_assignment_dict[int_col] = feature_df[int_col].to_numpy()
            id_df = id_df.assign(**intensity_column_assignment_dict)
            expression_control_plotting.make_expression_classification_figure(id_df, intensity_columns, celltypes, segmentation_folder, savename=savename, threshold_mean_models=None)
        else:
            print('Did not find model class {} in cfg, using default AAE model'.format(class_name))
            
    elif model_name.lower().startswith('nb') or model_name.startswith('clb_') or model_name.startswith('sknsh'):
        print('Running NB prediction with model {}'.format(model_name), flush=True)
        #if model_name == 'AAE':
        #    model_version = 0
        #else:
        #    if ('_v' in model_name):
        #        _, version_string = model_name.split('_v')
        #    elif ('_' in model_name):
        #        _, version_string = model_name.split('_')
        #    model_version = int(version_string)
        use_model = NBAAEmodel(annotation, model_name, cfg)
        id_df = use_model.predict_fluorophores(plate, output_folder, id_df)
        savename = join(output_folder, '{}_classification_control.png'.format(plate))
        expression_control_plotting.make_classification_control_figure(id_df, segmentation_folder, savename, cfg)
    
    elif model_name.lower().startswith('patchpy'):
        model_name = model_name.split('patchpy')[1]
        channel_id_column = 'DAPI_Channel_ID'
        print('Running patchpy prediction', flush=True)
        ### fix dataframe by adding missing columns if necessary
        missing_columns = [f for f in cfg.get_aae_columns() if not (f in id_df.columns)]
        use_model = AML_AAE_celltype_model(annotation, cfg=cfg)
        id_df = use_model.predict_fluorophores(plate, output_folder, id_df, fluorophores, celltypes, model_name)
        if 'CD3' in id_df.columns:
            id_df = correct_cd3_over_estimation(id_df)
        savename = join(output_folder, '{}_expression_classification_control.png'.format(plate))
        intensity_columns = cfg.get_intensity_columns()
        if len(intensity_columns) > len(fluorophores):
            intensity_use_cols = []
            if len([f for f in intensity_columns if f.startswith('cells')]) > 1:
                for fluo in fluorophores:
                    intensity_use_cols.append([f for f in intensity_columns if f.startswith('cells') and fluo in f][0])
            intensity_columns = intensity_use_cols
        intensity_column_assignment_dict = dict()
        for int_col in intensity_columns:
            intensity_column_assignment_dict[int_col] = feature_df[int_col].to_numpy()
        id_df = id_df.assign(**intensity_column_assignment_dict)
        expression_control_plotting.make_expression_classification_figure(id_df, intensity_columns, celltypes, segmentation_folder, savename=savename, threshold_mean_models=None)
    elif ('AAE' in model_name):
        print('Running regular AAE prediction', flush=True)
        if model_name == 'AAE':
            model_version = 0
        else:
            if ('_v' in model_name):
                _, version_string = model_name.split('_v')
            elif ('_' in model_name):
                _, version_string = model_name.split('_')
            model_version = int(version_string)
        use_model = AML_AAE_celltype_model(annotation, cfg=cfg)
        id_df = use_model.predict_fluorophores(plate, output_folder, id_df, fluorophores, celltypes, model_version)
        savename = join(output_folder, '{}_expression_classification_control.png'.format(plate))
        intensity_columns = cfg.get_intensity_columns()
        intensity_column_assignment_dict = dict()
        for int_col in intensity_columns:
            intensity_column_assignment_dict[int_col] = feature_df[int_col].to_numpy()
        id_df = id_df.assign(**intensity_column_assignment_dict)
        expression_control_plotting.make_expression_classification_figure(id_df, intensity_columns, celltypes, segmentation_folder, savename=savename, threshold_mean_models=None)
    else:
        print('Invalid model name specified: {}'.format(model_name))
        return None
    return id_df