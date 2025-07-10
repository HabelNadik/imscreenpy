import pandas as pd

from .viability_models import LD_stain_BayesianModel, ViabilityAAEmodel
from imscreenpy.config import Config


def add_viability_prediction(feature_df, id_df, db_path_or_connection, model_name, annotation, cfg, plate, output_folder, cfg_prefix=None):
    if cfg is None:
        cfg = Config(None, cfg_prefix=cfg_prefix)
        cfg.set_paths_patterns_properties()
    print('Adding viability prediction with model {}'.format(model_name))
    if model_name.lower().startswith('patchpy'):
        model_name = model_name.split('patchpy')[1]
        channel_id_column = 'DAPI_Channel_ID'
        print('Running patchpy prediction')
        ### fix dataframe by adding missing columns if necessary
        missing_columns = [f for f in cfg.get_aae_columns() if not (f in id_df.columns)]
        viability_model = ViabilityAAEmodel(cfg.get_aae_columns(), annotation, model_name, channel_id_column, batch_filename_template=cfg.viability_filename_template, \
                  slurm_script_path=cfg.viability_script_path, cfg=cfg, id_df_filename_suffix='_celltypes')
        viabilities = viability_model.predict(plate, output_folder, id_df)
    elif ('AAE' in model_name.upper()):
        print('Running regular prediction')
        if 'v' in model_name.lower():
            version_number = int(model_name.lower().split('v')[1])
        else:
            version_number = 0
        channel_id_column = 'DAPI_Channel_ID'
        viability_model = ViabilityAAEmodel(cfg.get_aae_columns(), annotation, version_number, channel_id_column, batch_filename_template=cfg.viability_filename_template, \
                  slurm_script_path=cfg.viability_script_path, cfg=cfg)
        viabilities = viability_model.predict(plate, output_folder, id_df)
    else:
        pass
    id_df = id_df.assign(Viable=viabilities)
    return id_df