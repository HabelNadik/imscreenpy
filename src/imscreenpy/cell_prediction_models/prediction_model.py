
import os
import subprocess
import time
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from ..config import Config
from ..misc import all_files_exist, num_files_exisiting, print_time

class PredictionModel:
    """
    Base class for prediction models. Defines the interface for all prediction models.
    """

    def __init__(self, required_columns, name='PredictionModel', cfg=None):
        self.required_columns = required_columns
        self.name = name
        if cfg is None:
            self.cfg = Config(None)
        else:
            self.cfg = cfg
        return


    def get_missing_required_columns(self, input_df):
        return [column for column in self.required_columns if not column in input_df.columns]

    def has_required_columns(self, input_df):
        return all([column in input_df.columns for column in self.required_columns])


class ClusterPredictionModel(PredictionModel):
    """
    Class for prediction models that run cluster jobs. Extends PredictionModel.
    """

    def __init__(self, required_columns, annotation, slurm_script_path, batch_filename_template, version_number, channel_id_column,\
                  partition_size=50000, cfg=None, name='PredictionModel', job_name_prefix='prediction', id_df_filename_suffix='_preprocessing'):
        super().__init__(required_columns, cfg=cfg, name=name)
        self.annotation = annotation
        self.partition_size = partition_size
        self.slurm_script_path = slurm_script_path
        self.filename_template = batch_filename_template
        self.predict_latent = cfg.predict_latent
        if self.predict_latent:
            if isinstance(cfg.celltype_latent_filename_template, str):
                self.set_up_latent_prediction(cfg.celltype_latent_filename_template, cfg.celltype_latent_script_path, cfg.latent_dimensionality)    
            else:
                self.set_up_latent_prediction(cfg.latent_filename_template, cfg.latent_script_path, cfg.latent_dimensionality)
        self.version_number = version_number
        self.channel_id_column = channel_id_column
        self.job_name_prefix = job_name_prefix
        self.id_df_filename_suffix = id_df_filename_suffix

    def predict(self, plate, target_folder, id_df):
        if self.has_required_columns(id_df):
            prediction = self.run_slurm_prediction(plate, target_folder, id_df)
            return prediction
        else:
            print('Missing required columns for {}'.format(self.name))
            print(self.get_missing_required_columns(id_df))
            return

    def build_slurm_command(self, last_batch_index, batch_index, plate, raw_image_data_path, channel_id, id_df_path, prediction_filepath):
        command_jobname = "--job-name={}_{}_{}".format(self.job_name_prefix, batch_index, plate)
        if isinstance(self.version_number, int):
            command_exports = "--export=images_path={},target_channel_id={},df_path={},outpath={},start_index={},end_index={},model_version={}".format(raw_image_data_path, int(channel_id), id_df_path, prediction_filepath, last_batch_index, batch_index, self.version_number)
        elif isinstance(self.version_number, str):
            if isinstance(self.channel_id_column, str):
                target_channel = self.channel_id_column.split('_')[0]
            else:
                channel_list = [f.split('_')[0] for f in self.channel_id_column]
                target_channel = "\'" + ','.join(channel_list) + "\'"
            command_exports = "--export=images_path={},target_channels={},df_path={},outpath={},start_index={},end_index={},model_identifier={}".format(raw_image_data_path, target_channel, id_df_path, prediction_filepath, last_batch_index, batch_index, self.version_number)
        else:
            command_exports = "--export=images_path={},target_channel_id={},df_path={},outpath={},start_index={},end_index={}".format(raw_image_data_path, int(channel_id), id_df_path, prediction_filepath, last_batch_index, batch_index)
        if self.predict_latent:
            command = "sbatch {} {} {}".format(command_jobname, command_exports, self.latent_script_path)
        else:    
            command = "sbatch {} {} {}".format(command_jobname, command_exports, self.slurm_script_path)
        return command

    def set_up_latent_prediction(self, latent_filename_template, latent_script_path, latent_dimensionality):
        self.latent_filename_template = latent_filename_template
        self.latent_script_path = latent_script_path
        self.latent_dimensionality = latent_dimensionality
        self.predict_latent = True
        return
    
    def make_output_filename(self, plate, last_batch_index, batch_index, latent=False):
        if latent:
            return self.latent_filename_template.format(plate, self.version_number, last_batch_index, batch_index)
        else:
            return self.filename_template.format(plate, self.version_number, last_batch_index, batch_index)

    def run_slurm_prediction(self, plate, target_folder, input_df):
        ### get data for running job across channels
        plate_df = self.annotation[self.annotation[self.cfg.an_plate_col] == plate]
        raw_image_data_path = plate_df[self.cfg.an_raw_input_path_col].values[0]
        id_df_path = os.path.join(target_folder, '{}_id_df{}.csv'.format(plate, self.id_df_filename_suffix))
        all_prediction_filepaths = []
        if isinstance(self.channel_id_column, list):
            channel_id = None
        else:
            channel_id = int(plate_df[self.channel_id_column].values[0])
        num_files_to_be_generated = 0
        batch_indices = [i for i in range(self.partition_size, input_df.shape[0], self.partition_size)] + [input_df.shape[0]]
        last_batch_index = 0
        filepaths = []
        for _, batch_index in enumerate(batch_indices):
            ### build command
            prediction_filepath = os.path.join(target_folder, self.make_output_filename(plate, last_batch_index, batch_index))
            if self.predict_latent:
                latent_filepath = os.path.join(target_folder, self.make_output_filename(plate, last_batch_index, batch_index, latent=True))
            if not os.path.isfile(prediction_filepath):
                command = self.build_slurm_command(last_batch_index, batch_index, plate, raw_image_data_path, channel_id, id_df_path, prediction_filepath)
                print('Running command {}'.format(command))
                subprocess.run(command, shell=True)
                filepaths.append(prediction_filepath)
                if self.predict_latent:
                    filepaths.append(latent_filepath)
            last_batch_index = batch_index
        num_files_to_be_generated = len(filepaths)
        start_waiting = timer()
        is_waiting = True
        fluo_found_array = np.zeros(num_files_to_be_generated).astype(bool)
        while is_waiting:
            if ((len(filepaths) == 0) or all_files_exist(filepaths)):
                end_wait_fluo = timer()
                print('Found all files for prediction')
                print_time(start_waiting, end_wait_fluo, waitmode=False)
                pred_array = np.zeros((input_df.shape[0]))
                if self.predict_latent:
                    latent_pred_array = np.zeros((input_df.shape[0], self.latent_dimensionality))
                batch_indices = [i for i in range(self.partition_size, input_df.shape[0], self.partition_size)] + [input_df.shape[0]]
                last_batch_index = 0
                for _, batch_index in enumerate(batch_indices):
                    output_filename = self.make_output_filename(plate, last_batch_index, batch_index)
                    prediction_filepath = os.path.join(target_folder, output_filename)
                    prediction = np.loadtxt(prediction_filepath)
                    if (prediction.shape[0] == 0):
                        print('Prediction not fully here. Waiting 3 seconds and trying again')
                        time.sleep(3)
                        prediction = np.loadtxt(prediction_filepath)
                    print('At batch index {} trying to write array of shape {} into shape {}'.format(batch_index, prediction.shape, pred_array.shape))
                    if len(prediction.shape) > 1: ## handling of edge case where there is only one prediction
                        pred_array[last_batch_index:batch_index] = np.argmax(prediction, axis=1)
                    else:
                        pred_array[last_batch_index:batch_index] = np.argmax(prediction)
                    if self.predict_latent:
                        latent_filename = self.make_output_filename(plate, last_batch_index, batch_index, latent=True)
                        latent_prediction_filepath = os.path.join(target_folder, latent_filename)
                        latent_prediction = np.loadtxt(latent_prediction_filepath)
                        latent_pred_array[last_batch_index:batch_index,:] = latent_prediction
                    last_batch_index = batch_index
                num_positive = np.sum(pred_array)
                num_total = pred_array.shape[0]
                print('Done with predictions for {}. Percent positive cells is {} with {} positive out of {}'.format(plate, num_positive / float(num_total), num_positive, num_total))
                is_waiting = False
            if fluo_found_array.all():
                end_all = timer()
                is_waiting = False
                print('Done with predictions for all cells')
                print_time(start_waiting, end_all)
            else:
                wait_point = timer()
                if (wait_point - start_waiting) > 3600:
                    print('still waiting')
                    num_files_done = num_files_exisiting(all_prediction_filepaths)
                    print('{} out of {} files generated so far'.format(num_files_done, num_files_to_be_generated))
                    print_time(start_waiting, wait_point, waitmode=True)
                    start_waiting = timer()
        if self.predict_latent:
            return pred_array, latent_pred_array
        return pred_array
