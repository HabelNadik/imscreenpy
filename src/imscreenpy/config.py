import os


class Config:

    def __init__(self, cfg_folder, cfg_prefix=None, copy_cfg=None, annotation=None, predict_latent=False):
        if (cfg_folder is None) or (not os.path.isdir(cfg_folder)):
            if (copy_cfg is None):
                cfg_folder = os.path.join(os.path.dirname(__file__), 'config_files')
            else:
                cfg_folder = copy_cfg.cfg_folder
        self.cfg_folder = cfg_folder
        if (cfg_prefix is None) or not isinstance(cfg_prefix, str):
            if (copy_cfg is not None):
                cfg_prefix = copy_cfg.cfg_prefix
            else:
                self.cfg_prefix = ''
        else:
            if not cfg_prefix.endswith('_'):
                cfg_prefix += '_'
            self.cfg_prefix = cfg_prefix
        ## initializing lists that need to be loaded from files at different points
        self.id_columns = None
        self._load_id_columns()
        self.morph_qc_columns = None
        self._load_morph_qc_columns()
        ## attributes that we need to access at all times referring to the annotation table
        self.an_seg_path_col = 'SegmentationOutputPathUncorrected'
        self.an_corr_seg_path_col = 'SegmentationOutputPathCorrected'
        self.an_raw_input_path_col = 'RawImageDataPath'
        self.an_transferlist_col = 'transferlist_filename'
        self.an_plate_col = 'PlateID'

        self.an_apc = 'APC'
        self.an_gfp ='GFP'
        self.an_pe = 'PE'
        self.an_dapi = 'DNA'
        self.an_plate_id_col = 'PlateID'
        self.an_target_col = 'Target'
        self.an_patient_id_col = 'InternalSampleNumber'
        #### additional field for other ids, needed for POP
        self.an_experiment_id_col = 'ExperimentID'
        self.an_condition_col = 'AdditionalCondition'
        ## attributes that we need to access all times referring to cellprofiler
        self.row_cpc = "Metadata_row"
        self.col_cpc = "Metadata_column"
        self.field_cpc = "Metadata_field"
        self.well_cpc = 'Metadata_Well'
        #### same for the image metadata
        self.im_well_cpc = 'Image_Metadata_Well'
        self.im_row_cpc = "Image_Metadata_row"
        self.im_col_cpc = "Image_Metadata_column"
        self.im_field_cpc = "Image_Metadata_field"
        self.im_metadata_columns = [self.im_well_cpc, self.im_row_cpc, self.im_col_cpc, self.im_field_cpc]

        self.un_id_cpc = 'unique_image_id'
        self.predict_latent = predict_latent
        ## information taht we need to store while the pipeline is run
        if (copy_cfg is None):
            self.viability_model_name = None
            self.expression_model_name = None
            self.expression_model_class_name = None
            ## model scripts and patterns
            self.viability_latent_script_path = None
            self.viability_script_path = None
            self.viability_filename_template = None
            self.viability_latent_filename_template = None
            self.celltype_script_path = None
            self.celltype_latent_script_path = None
            self.celltype_filename_template = None
            self.celltype_latent_filename_template = None
            self.latent_dimensionality = None
            ## model specific columns
            self.model_feature_columns = []
            self.annotation = annotation
            self.fluorophores = []
            self.fluo_column_patterns = []
            self.db_cell_table_name = None
            self.db_im_table_name = None
            self.expression_model_columns = []
            self.viability_model_columns = []
            self.im_qc_ignore_fluorophores = []
            self._load_class_assignments()
        else:
            self.viability_model_name = copy_cfg.viability_model_name
            self.expression_model_name = copy_cfg.expression_model_name
            ## model scripts and patterns
            self.viability_latent_script_path = copy_cfg.viability_latent_script_path
            self.viability_script_path = copy_cfg.viability_script_path
            self.viability_filename_template = copy_cfg.viability_filename_template
            self.viability_latent_filename_template = copy_cfg.viability_latent_filename_template
            self.celltype_script_path = copy_cfg.celltype_script_path
            self.celltype_filename_template = copy_cfg.celltype_filename_template
            self.celltype_latent_script_path = copy_cfg.celltype_latent_script_path
            self.celltype_latent_filename_template = copy_cfg.celltype_latent_filename_template
            ## model specific columns
            self.model_feature_columns = copy_cfg.model_feature_columns
            self.annotation = copy_cfg.annotation
            self.fluorophores = copy_cfg.fluorophores
            self.fluo_column_patterns = copy_cfg.fluo_column_patterns
            self.db_cell_table_name = copy_cfg.db_cell_table_name
            self.db_im_table_name = copy_cfg.db_im_table_name
            self.class_assignment_dict = copy_cfg.class_assignment_dict
            self.expression_model_columns = copy_cfg.expression_model_columns
            self.viability_model_columns = copy_cfg.viability_model_columns
            self.predict_latent = copy_cfg.predict_latent
            self.im_qc_ignore_fluorophores = copy_cfg.im_qc_ignore_fluorophores
            ## attributes to check for since they may only be present in newer versions
            if 'expression_model_class_name' in copy_cfg.__dict__.keys():
                self.expression_model_class_name = copy_cfg.expression_model_class_name

    
    ### functions to load relevant data from files
    def list_from_text_file(self, filepath):
        feature_list = []
        in_file = open(filepath, 'r')
        for line in in_file.readlines():
            if len(line.strip()) > 0:
                feature_list.append(line.strip())
        in_file.close()
        return feature_list
    
    def _prep_for_aae_model(self):
        self.set_paths_patterns_properties()
        add_cols = self.get_aae_columns()
        self.model_feature_columns += [f for f in add_cols if not (f in self.model_feature_columns)]
        return add_cols
    
    def add_feature_columns(self, list_or_path):
        if isinstance(list_or_path, str):
            add_cols = self.list_from_text_file(list_or_path)
        elif isinstance(list_or_path, list):
            add_cols = list_or_path
        self.model_feature_columns += [f for f in add_cols if not (f in self.model_feature_columns)]

    ## getters for columns

    def _load_id_columns(self):
        if os.path.isfile(os.path.join(self.cfg_folder, self.cfg_prefix + 'cp_id_columns.txt')):
            id_path = os.path.join(self.cfg_folder, self.cfg_prefix + 'cp_id_columns.txt')
        else:
            id_path = os.path.join(self.cfg_folder, 'cp_id_columns.txt')
        self.id_columns = self.list_from_text_file(id_path)
        return
    
    def _load_morph_qc_columns(self):
        morph_path = os.path.join(self.cfg_folder, 'cp_morphology_qc_columns.txt')
        self.morph_qc_columns = self.list_from_text_file(morph_path)
        return

    def get_id_columns(self):
        return self.id_columns
    
    def get_morphology_qc_columns(self):
        return self.morph_qc_columns
        
    def get_aae_columns(self):
        filepath = os.path.join(self.cfg_folder, 'aae_required_columns.txt')
        return self.list_from_text_file(filepath)

    def get_model_feature_columns(self):
        return self.model_feature_columns
    
    def set_intensity_columns(self, target_fluorophores, intensity_column_suffix='IllumCorrected'):
        if not (self.fluo_column_patterns is None):
            self.intensity_columns = []
            for fluo in target_fluorophores:
                for pattern in self.fluo_column_patterns:
                    self.intensity_columns.append(pattern.format(fluo))
                    if isinstance(intensity_column_suffix, str) and (len(intensity_column_suffix) > 0):
                        self.intensity_columns.append(pattern.format(fluo) + intensity_column_suffix)
        else:
            self.intensity_columns = None
        return

    def get_intensity_columns(self):
        return self.intensity_columns

    def get_all_set_columns(self):
        out = self.morph_qc_columns + self.model_feature_columns + self.id_columns
        if isinstance(self.intensity_columns, list):
            out += self.intensity_columns
        return out

    ## getters and setters for settings

    def set_expression_model(self, model_name):
        if ('aae' in model_name.lower()) or model_name.lower().endswith('tuned'):
            self.expression_model_columns = self._prep_for_aae_model()
        self.expression_model_name = model_name
        return

    def get_expression_model(self):
        return self.expression_model_name
    
    def get_expression_model_columns(self):
        return self.expression_model_columns

    def set_viability_model(self, model_name):
        if ('aae' in model_name.lower()) or model_name.lower().endswith('tuned'):
           self.viability_model_columns = self._prep_for_aae_model()
        self.viability_model_name = model_name
        return

    def get_viability_model(self):
        return self.viability_model_name
    
    def get_viability_model_columns(self):
        return self.viability_model_columns
    
    def get_predict_latent(self):
        return self.predict_latent
    
    def set_predict_latent(self, value):
        self.predict_latent = value
        return

    def set_paths_patterns_properties(self):
        paths_and_patterns_path = os.path.join(self.cfg_folder, self.cfg_prefix + 'paths_and_patterns.txt')
        with open(paths_and_patterns_path, 'r') as file:
            for line in file.readlines():
                if (len(line.strip()) > 0) and ('=' in line) and not (line.startswith('#')):
                    key, value = line.strip().split('=')
                    key = key.upper().strip()
                    value = value.strip()
                    if key == 'VIABILITY_LATENT_SCRIPT_PATH':
                        self.viability_latent_script_path = value
                    elif key == 'VIABILITY_SCRIPT_PATH':
                        self.viability_script_path = value
                    elif key == 'VIABILITY_FILENAME_TEMPLATE':
                        self.viability_filename_template = value
                    elif key == 'VIABILITY_LATENT_FILENAME_TEMPLATE':
                        self.viability_latent_filename_template = value
                    elif key == 'CELLTYPE_SCRIPT_PATH':
                        self.celltype_script_path = value
                    elif key == 'CELLTYPE_FILENAME_TEMPLATE':
                        self.celltype_filename_template = value
                    elif key == 'CELLTYPE_LATENT_SCRIPT_PATH':
                        self.celltype_latent_script_path = value
                    elif key == 'CELLTYPE_LATENT_FILENAME_TEMPLATE':
                        self.celltype_latent_filename_template = value
                    elif key == 'LATENT_DIMENSIONALITY':
                        self.latent_dimensionality = int(value)
                    elif key == 'FLUOROPHORES':
                        self.fluorophores = [f.strip() for f in value.strip().split(',')]
                    elif key == 'DB_CELL_TABLE_NAME':
                        self.db_cell_table_name = value
                    elif key == 'DB_IM_TABLE_NAME':
                        self.db_im_table_name = value
                    elif key == 'FLUO_COLUMN_PATTERNS':
                        self.fluo_column_patterns = [f.strip() for f in value.strip().split(',')]
        return
    
    def get_im_qc_ignore_flurophores(self):
        return self.im_qc_ignore_fluorophores
    
    def set_im_qc_ignore_flurophores(self, value):
        self.im_qc_ignore_fluorophores = value
        return

    

    def _load_class_assignments(self):
        class_assignment_path = os.path.join(self.cfg_folder, self.cfg_prefix + 'class_assignments.txt')
        if os.path.isfile(class_assignment_path):
            self.class_assignment_dict = dict()
            with open(class_assignment_path, 'r') as file:
                for line in file.readlines():
                    if (len(line.strip()) > 0) and ('=' in line) and not (line.startswith('#')):
                        key, value = line.strip().split('=')
                        key = int(key.strip())
                        value = value.strip()
                        self.class_assignment_dict[key] = value
        else:
            self.class_assignment_dict = None
        return
