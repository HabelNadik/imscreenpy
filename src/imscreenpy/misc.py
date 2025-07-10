from genericpath import isfile
import pandas as pd
import numpy as np
from string import ascii_uppercase
from timeit import default_timer as timer
import matplotlib.pyplot as plt


APC = 'APC'
GFP ='GFP'
PE = 'PE'
DAPI = 'DNA'
PLATE_ID_COL = 'PlateID'
PATIENT_ID_COL = 'InternalSampleNumber'

def platehist_from_bool(bool_in, table):
    rows = list(set(table['Metadata_row'].values))
    cols = list(set(table['Metadata_column'].values))
    rows.sort()
    cols.sort()
    arr = np.zeros((max(rows), max(cols)))
    arr[:,:] = np.nan
    for r in rows:
        for c in cols:
            well_arr = ((table['Metadata_row'] == r) & (table['Metadata_column'] == c)).values.astype(np.bool)
            value = np.sum(bool_in & well_arr) 
            arr[r-1,c-1] = value
    fig, ax = plt.subplots(figsize=(14,7))
    sh = ax.imshow(arr, cmap='coolwarm')
    ax.set_xticks(list(range(24)))
    ax.set_xticklabels(list(range(1, max(cols)+1)))
    ax.set_yticks(list(range(max(rows))))
    ax.set_yticklabels(['{}-{}'.format(i+1, ascii_uppercase[i]) for i in range(max(rows))])
    plt.colorbar(sh)
    fig.show()
    return

def string_to_numeric(in_string):
    if not isinstance(in_string, str):
        return None
    else:
        in_string = in_string.strip()
        if len(in_string) == 0:
            return None
        elif ('eps' in in_string.lower()):
            return np.finfo(np.float32).eps
        elif ('.' in in_string):
            return float(in_string)
        else:
            return int(in_string)

def num_list_from_string(in_string, sep=','):
    vals = in_string.split(sep)
    return [string_to_numeric(f) for f in vals]

def well_string_to_nums(well_string):
    row = ascii_uppercase.index(well_string[0]) + 1
    column = int(well_string[1:])
    return row, column

def nums_to_well_string(row, column):
    row_id = ascii_uppercase[row-1]
    col_id = str(column)
    if len(col_id) == 1:
        col_id = '0' + col_id
    return row_id + col_id

def get_control_wells(transfer_table, plate=None):
    if  not (plate is None):
        transfer_table = transfer_table[transfer_table['DestinationPlate'] == plate]
    control_wells = []
    for index, row in transfer_table.iterrows():
        if row['Type_A'] != 'Cpd':
            control_wells.append(row['DestWell'])
    return control_wells

def make_treated_well_boolean_list(cell_table, transfer_table):
    treated_wells = []
    for index, row in transfer_table.iterrows():
        if row['Type_A'] == 'Cpd':
            treated_wells.append(row['DestWell'])
    out_list = [(cell_table['Metadata_Well'] == well).values for well in treated_wells]
    return out_list


def make_well_boolean(cp_table, wells):
    out = cp_table['Metadata_Well'].isin(wells)
    return out

def make_control_boolean(cp_table, transfer_table):
    control_wells = get_control_wells(transfer_table)
    boolean = make_well_boolean(cp_table, control_wells)
    return boolean

def make_control_well_boolean_list(cell_table, transfer_table):
    control_wells = get_control_wells(transfer_table)
    out_list = [(cell_table['Metadata_Well'] == well).values for well in control_wells]
    return out_list

def make_pair_string_from_list(in_list, top10_only=True):
    out_string = ''
    length_before_break = 0
    if top10_only:
        sorted_10 = []
        if len(in_list[0]) > 2:
            wells = [(f[0], f[1]) for f in in_list]
        else:
            wells = in_list
        counts = [wells.count(f) for f in wells]
        zipped = list(zip(counts, wells))
        zipped.sort()
        for well in zipped[::-1]:
            sorted_10.append(well[1])
    in_list = sorted_10[:10]
    for elem in in_list:
        substring = '('
        for subelem in elem:
            substring += str(subelem) + ','
        out_string += substring[:len(substring)-1] + '), '
        length_before_break += len(substring)
        if length_before_break > 50:
            out_string += '\n'
            length_before_break = 0
    return out_string[:len(out_string)-2]


def make_wells_for_drug_dict(transfer_table, input_wells):
    all_wells = list(set(transfer_table['DestWell'].values))
    out_dict = dict()
    for well in all_wells:
        if well in input_wells:
            transfer_row = transfer_table[transfer_table['DestWell'] == well].iloc[0,:]
            if transfer_row['Type_A'] == 'Cpd':
                drug =  transfer_row['CompoundName']
                conc = transfer_row['AssayConc_uM']
            elif transfer_row['Type_A'].startswith('P'):
                if transfer_row['Type_A'].endswith('V'):
                    drug = 'Venetoclax'
                    conc = 10
                elif transfer_row['Type_A'].endswith('CD'):
                    drug = 'Cytarabine+Daunorubicin'
                    conc = 10
                elif transfer_row['Type_A'].endswith('BZ'):
                    drug = 'Bortezomib'
                    conc = 10
                elif transfer_row['Type_A'].endswith('ST'):
                    drug = 'Staurosporine'
                    conc = 10
                else:
                    if 'Type' in transfer_row.keys():
                        if transfer_row['Type'].endswith('V'):
                            drug = 'Venetoclax'
                            conc = 10
                        elif transfer_row['Type'].endswith('CD'):
                            drug = 'Cytarabine+Daunorubicin'
                            conc = 10
            else:
                drug = 'DMSO'
                conc = 0
            if ('CellLine' in transfer_table.columns):
                cell_line = transfer_row['CellLine']
                out_dict[well] = drug, conc, cell_line
            else:
                out_dict[well] = drug, conc
        else:
            out_dict[well] = 'None', 0
    return out_dict, all_wells

def add_treatments(cell_tables, plate_ids, transfer_table, consider_cell_lines=False):
    out_tables = []
    if not isinstance(cell_tables, list):
        cell_tables = [cell_tables]
    if not isinstance(plate_ids, list):
        plate_ids = [plate_ids]
    print('Adding treatments to {} cell tables'.format(len(cell_tables)))
    for i, local_table in enumerate(cell_tables):
        plate_id = plate_ids[i]
        sub_transfer = transfer_table[transfer_table['DestinationPlate'] == plate_id]
        table_wells = local_table['Metadata_Well'].values
        wells = list(set(table_wells))
        well_dict, wells = make_wells_for_drug_dict(sub_transfer, wells)
        drugs = []
        concentrations = []
        platename_col = []
        cell_lines = []
        for well in table_wells:
            if well in well_dict.keys():
                well_value = well_dict[well]
                if ('CellLine' in transfer_table.columns):
                    drug, conc, cell_line = well_value
                    cell_lines.append(cell_line)
                else:
                    drug, conc = well_value
                #if len(well_value) > 2:
                    #print('Warning. More than two values in well_value. This should not happen.')
                    #print(well_value)
                drugs.append(drug)
                concentrations.append(conc)
            else:
                #print(well + ' not represented in transferlist')
                drugs.append('None')
                concentrations.append(0)
                cell_lines.append('None')
            platename_col.append(plate_id)
        local_table = local_table.assign(Drug=drugs, Concentration=concentrations, Plate=platename_col)
        if (len(cell_lines) > 0) and consider_cell_lines:
            local_table = local_table.assign(CellLine=cell_lines)
        out_tables.append(local_table)
    return out_tables


def match_vis_folder_to_subtable(subtable, vis_folders, return_plate=False):
    plate = subtable['Plate'].values[0]
    for v_folder in vis_folders:
        if str(plate) in v_folder:
            if return_plate:
                return v_folder, plate
            else:
                return v_folder

def add_treatments_from_excel_arrays(use_df, treatment_annotation, dose_annotation, condition_annotation=None):
    rows = np.unique(use_df['Metadata_row'].values)
    cols = np.unique(use_df['Metadata_column'].values)
    treatments = np.zeros(use_df.shape[0]).astype(object)
    doses = np.zeros(use_df.shape[0])
    conditions = np.zeros(use_df.shape[0]).astype(object)
    for row in rows:
        for col in cols:
            row_col_mask = ((use_df['Metadata_row'] == row) & (use_df['Metadata_column'] == col)).to_numpy()
            treatment = treatment_annotation.iloc[row-1,col-1]
            dose = dose_annotation.iloc[row-1,col-1]
            treatments[row_col_mask] = treatment
            doses[row_col_mask] = dose
            if not (condition_annotation is None):
                conditions[row_col_mask] = condition_annotation.iloc[row-1,col-1]
    ass_dict = dict()
    ass_dict['Drug'] = treatments
    ass_dict['Concentration'] = doses
    if not (condition_annotation is None):
        ass_dict['Condition'] = conditions
    return use_df.assign(**ass_dict)



def excel_annotation_to_transferlist(excel_path, out_path, target_sheet=0, concentration_sheet=None, dest_plate_name=None):
    all_sheet_dict = pd.read_excel(excel_path, sheet_name=None, header=None)
    print('Checking excel sheet with the following sheet names')
    print(all_sheet_dict.keys())
    if ('Concentration' in all_sheet_dict.keys()) and ('Treatment' in all_sheet_dict.keys()):
        print('Concentration and Treatment sheets founds')
        in_tab = all_sheet_dict['Treatment']
        concentration_sheet = all_sheet_dict['Concentration']
    else:
        in_tab = pd.read_excel(excel_path, sheet_name=target_sheet, header=None)
    if ('Cellline' in all_sheet_dict.keys()):
        print('cell line sheet founds')
        cell_line_sheet = all_sheet_dict['Cellline']
    else:
        cell_line_sheet = None
    relevant_transferlist_columns = ['DestWell', 'DestinationPlate', 'CompoundName', 'AssayConc_uM', 'Type_A']
    wells = []
    compounds = []
    concentrations = []
    dest_plate_names = []
    well_types = []
    cell_lines = []
    if concentration_sheet is None:
        num_rows = in_tab.shape[0]
        num_columns = in_tab.shape[1]
        for i in range(num_rows):
            for k in range(num_columns):
                value = in_tab.iloc[i,k]
                if (' ' in value):
                    compound, conc, unit = value.split(' ')
                    well_type = 'Cpd'
                elif 'DMSO' in value:
                    compound = 'DMSO'
                    conc = 0
                    unit = 'uM'
                    well_type = 'control'
                if unit == 'nM':
                    conc = float(conc) / 1000
                elif unit == 'uM':
                    conc = float(conc)
                well_name = nums_to_well_string(i+1,k+1)
                wells.append(well_name)
                compounds.append(compound)
                concentrations.append(conc)
                dest_plate_names.append(dest_plate_name)
                well_types.append(well_type)
                if not (cell_line_sheet is None):
                    cell_lines.append(cell_line_sheet.iloc[i,k])
    else:
        if isinstance(concentration_sheet, int):
            concentration_sheet = pd.read_excel(excel_path, sheet_name=concentration_sheet, header=None)
        num_rows = in_tab.shape[0]
        num_columns = in_tab.shape[1]
        unit = 'uM'
        for i in range(num_rows):
            for k in range(num_columns):
                value = in_tab.iloc[i,k]
                if isinstance(value, str):
                    if not ('DMSO' in value):
                        compound = value
                        conc = float(concentration_sheet.iloc[i,k])
                        well_type = 'Cpd'
                    else:
                        compound = 'DMSO'
                        conc = 0               
                        well_type = 'control'
                else:
                    print('Warning! No value present in cell ({},{}) in treatment sheet. Assuming DMSO'.format(i,k))
                    print('Value in treatment sheet: {} Value in concentration sheet: {}'.format(in_tab.iloc[i,k], concentration_sheet.iloc[i,k]))
                    compound = 'DMSO'
                    conc = 0               
                    well_type = 'control'
                well_name = nums_to_well_string(i+1,k+1)
                wells.append(well_name)
                compounds.append(compound)
                concentrations.append(conc)
                dest_plate_names.append(dest_plate_name)
                well_types.append(well_type)
                if not (cell_line_sheet is None):
                    cell_lines.append(cell_line_sheet.iloc[i,k])
    frame_dict = dict()
    frame_dict['DestWell'] = wells
    frame_dict['DestinationPlate'] = dest_plate_names
    frame_dict['CompoundName'] = compounds
    frame_dict['AssayConc_uM'] = concentrations
    frame_dict['Type_A'] = well_types
    if len(cell_lines) > 0:
        frame_dict['CellLine'] = cell_lines
    frame = pd.DataFrame(frame_dict)
    if not (out_path is None):
        frame.to_csv(out_path, sep='\t')
    return frame
                
def get_feature_list(filepath='reduced_feature_list.txt'):
    feature_list = []
    in_file = open(filepath, 'r')
    for line in in_file.readlines():
        if len(line.strip()) > 0:
            feature_list.append(line.strip())
    in_file.close()
    return feature_list


def print_time(start, end, waitmode=False):
    elaps = (end - start) / 60.
    if not waitmode:
        print('This took {} minutes'.format(round(elaps, 3)), flush=True)
    else:
        print('Waited for {} minutes so far'.format(round(elaps, 3)), flush=True)
    return

def convert_annotation_to_local(annotation, nested_replace_patterns):
    """
    Convert file paths in an annotation DataFrame to a local format by replacing specified patterns.
    
    """
    columns_to_consider = ['RawImageDataPath', 'SegmentationOutputPathUncorrected', 'SegmentationOutputPathCorrected', 'CorrectedImagesPath']
    replacement_dict = dict()
    for col in columns_to_consider:
        vals = annotation[col].values
        for val in vals:
            if isinstance(val, str):
                rep = None
                for rep_pattern in nested_replace_patterns:
                    if rep_pattern[0] in val:
                        rep = val.replace(rep_pattern[0], rep_pattern[1])
                if not (rep is None):
                    replacement_dict[val] = rep
    out = annotation.replace(replacement_dict)
    return out

def get_fluorophores_and_celltypes(plate_id, annotation, cfg):
    sub_table = annotation[annotation[cfg.an_plate_col] == plate_id]
    fluos_out = []
    markers_out = []
    for fluo in cfg.fluorophores:
        marker_cell_content = sub_table[fluo].values[0]
        if isinstance(marker_cell_content, str):
            marker = marker_cell_content.strip()
            if isinstance(marker, str) and (marker != 'None'):
                fluos_out.append(fluo)
                markers_out.append(marker)
    return fluos_out, markers_out

def get_indices_to_match(input_df, df_to_match_to):
    im_id_col = 'unique_image_id'
    obj_num_col = 'ObjectNumber'
    match_indices = np.arange(0, df_to_match_to.shape[0], 1).reshape(-1,)
    unique_images_target = list(set(df_to_match_to[im_id_col].values))
    unique_input_images = list(set(input_df[im_id_col].values))
    for im in unique_images_target:
        if im in unique_input_images:
            im_target_mask = (df_to_match_to[im_id_col] == im).to_numpy()
            im_input_mask = (input_df[im_id_col] == im).to_numpy()
            input_obj_nums = input_df[im_input_mask][obj_num_col].values
            input_target_nums = df_to_match_to[im_target_mask][obj_num_col].values
            if not (input_obj_nums == input_target_nums).all():
                pass
    return

def all_files_exist(filelist):
    all_found = True
    for filepath in filelist:
        if not isfile(filepath):
            all_found = False
    return all_found


def num_files_exisiting(filelist):
    num_found_total = 0
    if isinstance(filelist[0], list):
        for sub_filelist in filelist:
            found_arr = np.array([isfile(f) for f in sub_filelist])
            num_found_total += np.sum(found_arr)
    elif isinstance(filelist[0], str):
        found_arr = np.array([isfile(f) for f in filelist])
        num_found_total += np.sum(found_arr)
    return num_found_total

def has_all_columns(df, column_list):
    return len([f for f in column_list if (f in df.columns)]) == len(column_list)

def make_target_column(cell_table, marker_columns, target_annotation):
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


def make_name(template, row, col, field):
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
    return template.format(row, col, field)


def make_string_arr(in_array):
    use_list = []
    for elem in in_array:
        if elem < 10:
            use_list.append('0')
        else:
            use_list.append('')
    #add_arr = np.array(use_list)
    #out_arr = np.core.defchararray.add(add_arr, in_array.astype(str))
    add_arr = np.array(use_list).astype(str)
    out_arr = np.core.defchararray.add(add_arr, in_array.astype(str))
    return out_arr

def make_unique_row_col_field_ids(rows, cols, fields, as_int=True):
    rows = rows.astype(np.uint16)
    cols = cols.astype(np.uint16)
    fields = fields.astype(np.uint16)
    if as_int:
        out = rows * 10000 + cols * 100 + fields
        unique_ids = out.astype(object)
    else:
        ### make string ids with two digits
        row_str_arr = make_string_arr(rows)
        col_str_arr = make_string_arr(cols)
        field_str_arr = make_string_arr(fields)
        ### combine them to get a unique string signature
        rowcol_ids = np.core.defchararray.add(row_str_arr, col_str_arr)
        unique_ids = np.core.defchararray.add(rowcol_ids, field_str_arr)
    return unique_ids


def format_p_value_string(p_val, n_digits=4):
    threshold = 10 ** (-n_digits)
    if p_val >= threshold:
        p_val_string = '= {}'.format(round(p_val, n_digits))
    else:
        p_val_string = '< {}'.format(threshold)
    return p_val_string

