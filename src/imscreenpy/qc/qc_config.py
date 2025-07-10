import os

from ..config import Config
from ..misc import string_to_numeric, num_list_from_string


class QCconfig(Config):

    def __init__(self, cfg_folder, cfg_prefix=None, nan_fraction_threshold=0.8, copy_cfg=None):
        if (cfg_folder is None) or (not os.path.isdir(cfg_folder)):
            repo_folder = os.path.join(os.path.dirname(__file__), '../')
            cfg_folder = os.path.join(repo_folder, 'config_files')
        super().__init__(cfg_folder, cfg_prefix=cfg_prefix, copy_cfg=copy_cfg)
        self.nan_fraction_threshold = nan_fraction_threshold
        ## CellProfiler columns that qc uses
        self.size_col = 'nuclei_AreaShape_Area'
        self.dapi_intensity_col = 'nuclei_Intensity_MedianIntensity_DNA'
        self.texture_entropy_col = 'nuclei_Texture_SumEntropy_DNA_3_00_256'
        self.texture_contrast_col = 'nuclei_Texture_Contrast_DNA_3_00_256'
        self.correlation_col = 'nuclei_Texture_Correlation_DNA_3_00_256'
        self.texture_sum_col = 'nuclei_Texture_SumAverage_DNA_3_00_256'
        self.texture_info_meas2_col = 'nuclei_Texture_InfoMeas2_DNA_3_00_256'
        self.solidity_col = 'nuclei_AreaShape_Solidity'
        self.eccentricity_col = 'nuclei_AreaShape_Eccentricity'
        ## thresholds and other exclusion parameters
        self.max_size_threshold = None ## size thresholds refer to area in pixels as in the column nuclei_AreaShape_Area
        self.min_size_threshold = None
        self.min_dapi_intensity_threshold = None
        self.min_solidity_threshold = None
        self.max_eccentricity_threshold = None
        self.min_entropy_threshold = None
        self.min_contrast_threshold = None
        self.texture_sum_min_value = None
        self.info_meas2_exclude_value = None
        self.correlation_exclude_values = None
        self._load_morph_qc_params()


    def _load_morph_qc_params(self):
        param_path = os.path.join(self.cfg_folder, self.cfg_prefix + 'morph_qc_params.txt')
        with open(param_path, 'r') as file:
            for line in file.readlines():
                if len(line.strip()) > 0:
                    key, value = line.strip().split('=')
                    key = key.upper().strip()
                    value = value.strip()
                    ### single value thresholds and exclude values
                    if key.upper() == 'MAX_SIZE_THRESHOLD':
                        self.max_size_threshold = string_to_numeric(value)
                    elif key == 'MIN_SIZE_THRESHOLD':
                        self.min_size_threshold = string_to_numeric(value)
                    elif key == 'MIN_DAPI_INTENSITY_THRESHOLD':
                        self.min_dapi_intensity_threshold = string_to_numeric(value)
                    elif key == 'MIN_SOLIDITY_THRESHOLD':
                        self.min_solidity_threshold = string_to_numeric(value)
                    elif key == 'MAX_ECCENTRICITY_THRESHOLD':
                        self.max_eccentricity_threshold = string_to_numeric(value)
                    elif key == 'MIN_ENTROPY_THRESHOLD':
                        self.min_entropy_threshold = string_to_numeric(value)
                    elif key == 'MIN_CONTRAST_THRESHOLD':
                        self.min_contrast_threshold = string_to_numeric(value)
                    elif key == 'TEXTURE_SUM_MIN_VALUE':
                        self.texture_sum_min_value = string_to_numeric(value)
                    elif key == 'INFO_MEAS2_EXCLUDE_VALUE':
                        self.info_meas2_exclude_value = string_to_numeric(value)
                    ## list based exclude values
                    elif key == 'CORRELATION_EXCLUDE_VALUES':
                        self.correlation_exclude_values = num_list_from_string(value)

        return