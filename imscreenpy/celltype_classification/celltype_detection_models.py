import os

import pandas as pd
import numpy as np

from ..cell_prediction_models.prediction_model import ClusterPredictionModel

def get_intensity_columns(fluorophores, obj_type='cells_dist', measurement_type='Median'):
    out_cols = ['{}_Intensity_{}Intensity_{}'.format(obj_type, measurement_type, f) for f in fluorophores]
    return out_cols


def get_columns_for_model(fluorophores, model_name):
    if model_name == 'bayesian':
        return get_intensity_columns(fluorophores, obj_type='cells_dist', measurement_type='Median')
    else:
        return get_intensity_columns(fluorophores, obj_type='cells_dist', measurement_type='Median')

class AAEmarkerModel(ClusterPredictionModel):

    def __init__(self, target_fluorophore, dapi_channel_id, required_columns, annotation, batch_filename_template, version_number, channel_id_column, slurm_script_path, partition_size=100000, cfg=None, name='celltype_model'):
        super().__init__(required_columns, annotation, slurm_script_path, batch_filename_template, version_number, channel_id_column, partition_size=partition_size, cfg=cfg, name=name)
        self.target_fluo = target_fluorophore
        self.dapi_channel_id = int(dapi_channel_id)
        print('Set up AAE marker model with version {} for {} and channel id column {}'.format(self.version_number, target_fluorophore, channel_id_column), flush=True)
        if isinstance(self.version_number, str):
            self.channel_id_column = [channel_id_column, 'DAPI_Channel_ID']
            print('Adjusted channel id column to {}'.format(self.channel_id_column), flush=True)
        return

    def build_slurm_command(self, last_batch_index, batch_index, plate, raw_image_data_path, channel_id, id_df_path, prediction_filepath):
        command_jobname = "--job-name={}_{}_{}".format(self.job_name_prefix, batch_index, plate)
        if isinstance(self.version_number, str):
            if isinstance(self.channel_id_column, str):
                target_channel = self.channel_id_column.split('_')[0]
            else:
                channel_list = [f.split('_')[0] for f in self.channel_id_column]
                target_channel = '#'.join(channel_list)
            print('Celltype prediction. Building command with target channel: {}'.format(target_channel), flush=True)
            command_exports = "--export=images_path={},target_channels={},df_path={},outpath={},start_index={},end_index={},model_identifier={}".format(raw_image_data_path, target_channel, id_df_path, prediction_filepath, last_batch_index, batch_index, self.version_number)
        else:
            command_exports = "--export=images_path={},target_channel_id={},df_path={},outpath={},start_index={},end_index={},dapi_channel_id={},model_version={}".format(raw_image_data_path, int(channel_id), id_df_path, prediction_filepath, last_batch_index, batch_index, self.dapi_channel_id, int(self.version_number))
        if self.predict_latent:
            command = "sbatch {} {} {}".format(command_jobname, command_exports, self.latent_script_path)
        else:    
            command = "sbatch {} {} {}".format(command_jobname, command_exports, self.slurm_script_path)
        return command


    def make_output_filename(self, plate, last_batch_index, batch_index, latent=False):
        if latent:
            return self.latent_filename_template.format(plate, self.target_fluo, last_batch_index, batch_index)
        else:
            return self.filename_template.format(plate, self.target_fluo, last_batch_index, batch_index)


class AML_AAE_celltype_model:


    def __init__(self, annotation, cfg=None):
        self.annotation = annotation
        self.cfg = cfg
        self.partition_size = 100000
        return

    def predict_fluorophores(self, plate, target_folder, id_df, target_fluorophores, target_markers, version_number):
        plate_df = self.annotation[self.annotation[self.cfg.an_plate_col] == plate]
        ### get model specific data
        batch_filename_template = self.cfg.celltype_filename_template
        slurm_script_path = self.cfg.celltype_script_path
        dapi_channel_id = plate_df['DAPI_Channel_ID'].values[0]
        required_columns = self.cfg.get_aae_columns()
        assignment_dict = dict()
        for i, fluo in enumerate(target_fluorophores):
            print('Running prediction model for fluorophore {} and marker {}'.format(fluo, target_markers[i]))
            channel_id_column = '{}_Channel_ID'.format(fluo)
            marker_model = AAEmarkerModel(fluo, dapi_channel_id, required_columns, self.annotation, batch_filename_template,\
                                           version_number, channel_id_column, slurm_script_path, partition_size=self.partition_size, cfg=self.cfg, name='{}_celltype_model'.format(fluo))
            pred_array = marker_model.predict(plate, target_folder, id_df)
            assignment_dict[target_markers[i]] = pred_array
        return id_df.assign(**assignment_dict)


class NBAAEmodel(ClusterPredictionModel):

    def __init__(self, annotation, model_name, cfg, partition_size=100000, name='nb_model', target_channel_string='Alexa647#Cy3#DAPI'):
        ### get model specific data
        batch_filename_template = cfg.celltype_filename_template
        slurm_script_path = cfg.celltype_script_path
        required_columns = cfg.get_aae_columns()
        self.target_channel_string = target_channel_string
        super().__init__(required_columns, annotation, slurm_script_path, batch_filename_template, model_name, 'DAPI_Channel_ID', partition_size=partition_size, cfg=cfg, name=name)
        return

    def build_slurm_command(self, last_batch_index, batch_index, plate, raw_image_data_path, channel_id, id_df_path, prediction_filepath):
        command_jobname = "--job-name=NB_AAE_{}_{}".format(batch_index, plate)
        command_exports = "--export=images_path={},target_channels={},df_path={},outpath={},start_index={},end_index={},model_identifier={}".format(raw_image_data_path, self.target_channel_string, id_df_path, prediction_filepath, last_batch_index, batch_index, self.version_number)
        #command_exports = "--export=df_path={},platename={},outpath={},start_index={},end_index={},model_identifier={}".format(\
        #            id_df_path, plate, prediction_filepath, last_batch_index, batch_index, self.version_number)
        if self.predict_latent:
            command = "sbatch {} {} {}".format(command_jobname, command_exports, self.latent_script_path)
        else:    
            command = "sbatch {} {} {}".format(command_jobname, command_exports, self.slurm_script_path)
        return command


    def make_output_filename(self, plate, last_batch_index, batch_index, latent=False):
        if latent:
            return self.latent_filename_template.format(plate, self.version_number, last_batch_index, batch_index)
        else:
            return self.filename_template.format(plate, self.version_number, last_batch_index, batch_index)


    def predict_fluorophores(self, plate, target_folder, id_df):
        ### get model specific data
        prediction = self.predict(plate, target_folder, id_df)
        if self.predict_latent:
            pred_array, latent_array = prediction
            full_latent_filename_path = self.make_output_filename(plate, 0, pred_array.shape[0], latent=True)
            if full_latent_filename_path.endswith('.txt'):
                full_latent_filename_path = full_latent_filename_path.replace('.txt', '.npy')
            np.save(os.path.join(target_folder, full_latent_filename_path), latent_array)
        else:
            pred_array = prediction
        class_dict = self.cfg.class_assignment_dict
        new_cols = []
        unique_vals = np.unique(pred_array)
        assignment_dict = dict()
        for val in unique_vals:
            mask = pred_array == val
            assignment_dict[class_dict[val]] = mask
        return id_df.assign(**assignment_dict)