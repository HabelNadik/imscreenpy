from os.path import join, isfile
from os import listdir
import datetime
from random import sample

from threading import Thread

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, ttest_1samp
### plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import seaborn as sns

from .misc import well_string_to_nums
## drug scoring, formatting of concentrations and drug names
from .drug_scoring.drug_scoring import round_conc, simplify_drug_names, reformat_concentrations, interpolate_for_ref_concentrations, calcuate_auc_std_err
from .drug_scoring.drug_scoring import drug_sigmoid, fit_sigmoid, calc_auc, make_ic50_dict, order_single_compound_list, filter_by_fit_error
from .drug_scoring.plotting import plot_curves
from .qc.qc_functions import filter_implausible_cellnumbers, filter_bad_replicates, filter_by_min_n_wells, cap_population_numbers
from .db_df_processing.df_functions import full_table_to_well_viability_table

DEBUG = False


def prepare_reference_concentrations(all_concentrations, strategy, labels=None, verbose=False):
    """
    Prepare the reference concentrations, ie the concentrations that all data will be mapped to to make AUC values comparable

    Parameters
    ----------
    all_concentrations : array_like
                        array of all concentrations in the data, can be None if strategy is a list of floats
                    
    strategy : str or list
                string: 'auto' or 'all'
                'all' will use all concentrations in the data
                'auto' will use the most common concentrations

                list: list of floats that are the reference concentrations

    labels : array_like
            array of labels for the concentrations. Needs to be set if strategy is 'auto'.
            These usually correspond to drugs or treatments and need to be set to for inference of the most common concentrations across treatments.
    Returns
    -------
    ref_concentrations : list
                        list of reference concentrations
    """
    valid_strategies = ['auto', 'all']
    if isinstance(strategy, list):
        if np.sum([isinstance(f, float) for f in strategy]) == len(strategy):
            return strategy
        else:
            print('Strategy is not a list of floats')
            return None
    elif isinstance(strategy, str):
        if not strategy in valid_strategies:
            print('Invalid strategy {} provided. Valid strategies are {} Returning None'.format(strategy, valid_strategies))
            return None
        if strategy == 'all':
            return list(np.unique(all_concentrations))
        elif strategy == 'auto':
            if labels is None:
                print('Need to set labels if strategy is auto. Returning None.')
                return None
            conc_tuples = []
            unique_labels = np.unique(labels)
            for lbl in unique_labels:
                unique_concs_tuple = tuple(np.unique(all_concentrations[labels == lbl]))
                #hashes_for_labels.append(hash(unique_concs_tuple))
                conc_tuples.append(unique_concs_tuple)
            #unique_hashes, hash_counts = np.unique(hashes_for_labels, return_counts=True)
            unique_concentrations = list(set(conc_tuples))
            concentration_counts = [conc_tuples.count(f) for f in unique_concentrations]
            sort_indices = np.argsort(concentration_counts)[::-1]
            use_concs = list(unique_concentrations[sort_indices[0]])
            #most_common_hash = unique_hashes[np.argmax(hash_counts)]
            #use_concs = all_concentrations[labels == labels[hashes_for_labels.index(most_common_hash)]]
            if verbose:
                print(f'Most common concentrations: {list(np.unique(use_concs))}')
            return list(np.unique(use_concs))
            


class LDTable_InhibitionReport:

    def __init__(self, whole_table, verbose=False, run_initial_qc=True, min_wells_per_drug_to_consider=3,\
                  reference_concentrations='auto', adjust_combination_concentrations=False, viable_cellnumber_column='NumberOfCells', initial_capping=True):
        """
        Class to generate inhibition reports from dataframes that have columns to denote cellnumbers, drugs, and concentrations

        Parameters
        ----------
        whole_table : pandas.DataFrame
                    DataFrame with cellnumber data. Needs to have columns 'Drug', 'Concentration', 'Metadata_Well', 'Plate', 'NumberOfCells' 

        verbose : bool
                Verbosity level

        run_initial_qc : bool
                Whether to run some minimal initial QC steps. Default is True

        min_wells_per_drug_to_consider : int
                Minimum number of wells per drug to consider. Default is 3. Drugs with fewer wells will be excluded
                
        reference_concentrations : str or list
                Strategy to determine reference concentrations. Can be a list of concentrations to map to, 'auto' or 'all'.
                'auto' will use the most common concentrations across treatments.
                'all' will use all concentrations.
                Default is 'auto'
        
        """
        ### calculate inhibitions for all drugs
        if run_initial_qc:
            whole_table = filter_implausible_cellnumbers(whole_table)
            whole_table = filter_bad_replicates(whole_table, target_columns=viable_cellnumber_column, verbose=True)
        ### calculate inhibitions for all drugs
        self.verbose = verbose
        self.table = simplify_drug_names(whole_table)
        if not (min_wells_per_drug_to_consider is None):
            self.table = filter_by_min_n_wells(self.table, 'Drug', min_wells_per_drug_to_consider)
        self.table = reformat_concentrations(self.table, adjust_for_combos=adjust_combination_concentrations)
        self.reference_concentrations = prepare_reference_concentrations(self.table['Concentration'].values, reference_concentrations, labels=self.table['Drug'].values, verbose=self.verbose)
        self.plates = list(set(self.table['Plate'].values))
        self.compounds = [f for f in list(set(self.table['Drug'].values)) if (f != 'DMSO') and (f != 'None')]
        if initial_capping:
            self.table = cap_population_numbers(self.table, [f for f in self.table.columns if ('NumberOfCells' in f)], subsetting_column='Plate')
        self.compounds.sort()
        self.drug_to_plate = dict()
        self.min_wells_per_drug_to_consider = min_wells_per_drug_to_consider
        for drug in self.compounds:
            plate = self.table[self.table['Drug'] == drug]['Plate'].values[0]
            self.drug_to_plate[drug] = plate
        self.reset_calculation_flags()
        self.viable_cellnumber_column = viable_cellnumber_column

    def control_plate_dict(self, target_column=None, normto1=False, return_limits=False, return_std=False, offtarget=False):
        out_dict = dict()
        for plate in self.plates:
            if normto1:
                mean_num, std_num = self.num_cells_control(target_column=target_column, normto1=normto1, target_plate=plate, offtarget=offtarget)
            elif return_std or return_limits:
                mean_num, std_num = self.num_cells_control(target_column=target_column, normto1=normto1, return_std=return_std, target_plate=plate, offtarget=offtarget)
            else:
                mean_num = self.num_cells_control(target_column=target_column, normto1=normto1, return_std=return_std, target_plate=plate, offtarget=offtarget)
                std_num = None
            if return_limits:
                upper = mean_num + std_num
                lower = mean_num - std_num
                out_dict[plate] = mean_num, upper, lower
            elif return_std:
                out_dict[plate] = mean_num, std_num
            else:
                out_dict[plate] = mean_num
        return out_dict

    def make_plate_dictionaries(self, target_column, offtarget=False):
        """
        Generate dictionaries of cellnumbers for reference

        Returns:
        cell_num_plate_dict: Average cellnumber in control of target population if given
        cell_num_norm_plate_dict: Same but normalized such that the average in the controls is 1
        plate_norm_target_dict: Same as above, but for target cells
        plate_ref_value_dict: fractions of target cells for each control well
        """
        cell_num_plate_dict = self.control_plate_dict()
        cell_num_norm_plate_dict = self.control_plate_dict(normto1=True, return_std=True)
        plate_norm_target_dict = self.control_plate_dict(target_column=target_column, normto1=True, return_std=True, offtarget=offtarget)
        plate_ref_value_dict = dict()
        for plate in self.plates:
            plate_ref_value_dict[plate] = self.calculate_control_fractions(target_column, target_plate=plate, offtarget=offtarget)
        return cell_num_plate_dict, cell_num_norm_plate_dict, plate_norm_target_dict, plate_ref_value_dict


    def process_concentrations(self, in_conc_list, sort=True):
        out_conc_list = []
        ref_conc_list = []
        for conc in in_conc_list:
            if isinstance(conc, float) or isinstance(conc, int):
                newconc = round_conc(conc)
            elif isinstance(conc, str):
                if '+' in conc:
                    left, right = conc.split('+')
                    newconc = round_conc(float(left) + float(right))
                else:
                    newconc = round_conc(float(conc))
            else:
                print('!!!!!!!!!!!! CONCENTRATION INCONVERTIBLE !!!!!!!!!!!!!!!!!')
                newconc = conc
                print(conc)
            if newconc in out_conc_list:
                use_index = out_conc_list.index(newconc)
                ref_sublist = ref_conc_list[use_index].copy()
                ref_sublist.append(conc)
                ref_conc_list[use_index] = ref_sublist
            else:
                out_conc_list.append(newconc)
                ref_conc_list.append([conc])
        if sort:
            zipped = list(zip(out_conc_list, ref_conc_list))
            zipped.sort()
            out_conc_list, ref_conc_list = zip(*zipped)
        return out_conc_list, ref_conc_list

    def num_cells_control(self, normto1=False, return_std=False, target_column=None, target_plate=None, offtarget=False):
        ## normto1 to additiionally normalize cell number to 1
        cell_numbers = []
        if target_plate is None:
            plates = self.plates
        else:
            plates = [target_plate]
        for plate in plates:
            plate_numbers = self.num_cells_control_plate(plate, target_column=target_column, offtarget=offtarget)
            cell_numbers += plate_numbers
            if self.verbose:
                print('Got cellnumbers for plate {} Mean is: {} Std. is {} min is {} max is {}'.format(plate, np.mean(plate_numbers), np.std(plate_numbers), np.amin(plate_numbers), np.amax(plate_numbers)))        
        if normto1:
            norm_cell_numbers = np.array(cell_numbers) / np.mean(cell_numbers)
            norm_std_dev = np.nanstd(norm_cell_numbers)
            return np.nanmean(norm_cell_numbers), norm_std_dev
        if return_std:
            return np.nanmean(cell_numbers), np.std(cell_numbers)
        return np.nanmean(cell_numbers)


    def abs_avg_score(self, input_dictionary):
        scores = []
        self.abs_auc_dict = dict()
        for compound in self.compounds:
            #concs, cellnumbers, _ = input_dictionary[compound]
            vals = input_dictionary[compound]
            concs = vals[0]
            cellnumbers = vals[1]
            inhibitions = 1- np.array(cellnumbers)
            score = np.nanmean(inhibitions)
            scores.append(score)
            self.abs_auc_dict[compound] = score
        return self.abs_auc_dict.copy()

    def reset_calculation_flags(self):
        self.calculated_abs_target = False
        self.calculated_pharmacoscopy_singles = False
        self.calculated_pharmacoscopy_aucs = False
        self.calculated_rbf_abs = False
        return
    
    

    def calculate_pharmacoscopy_scores(self, target_column, return_dictionaries=False, norm_to_plate=True):
        ## all these dictionaries have to be used for the final scoring and report generation
        self.dose_response_dict = dict()
        self.dose_fraction_dict = dict()
        self.cell_number_dict = dict()
        self.normalized_cell_number_dict = dict()
        self.rbf_dict = dict()
        self.normalized_offtarget_dict = dict()
        ## reference values for normalization
        if self.verbose:
            print('Starting to score {} compounds'.format(len(self.compounds)))
        if norm_to_plate:
            ### returns 
            cell_num_plate_dict, cell_num_norm_plate_dict, plate_norm_target_dict, plate_ref_value_dict = self.make_plate_dictionaries(target_column)
            _, _, plate_norm_offtarget_dict, plate_offtarget_frac_dict = self.make_plate_dictionaries(target_column, offtarget=True)
        else:
            num_cells_control, cellnum_std = self.num_cells_control(return_std=True)
            num_target_cells_control, target_cellnum_std = self.num_cells_control(return_std=True, target_column=target_column)
            num_cells_norm1, cellnum_std_norm1 = self.num_cells_control(normto1=True)
            ref_values = self.calculate_control_fractions(target_column)
            offtarget_ref_values = self.calculate_control_fractions(target_column, offtarget=True)
        ## scores that we use in our list
        intermediate_scores = []
        all_fractions = []
        max_intermediate = 0
        min_intermediate = 1.
        all_concentrations = []
        keep_compounds = []
        #cell_numbers_std_devs = []
        #all_std_devs_fractions = []
        all_rbfs_testing = []
        for drug in self.compounds:
            drug_table = self.table[self.table['Drug'] == drug]
            concentrations_per_drug_in = list(set(drug_table['Concentration'].values))
            processed_concentrations, ref_concentrations = self.process_concentrations(concentrations_per_drug_in)
            if (len(processed_concentrations) >= 4):
                if self.verbose:
                    print('Strange concentrations for drug {}!!!!'.format(drug))
                    print('Orig')
                    print(concentrations_per_drug_in)
                    print('Processed')
                    print(processed_concentrations)
            conc_sort_indices = np.argsort(processed_concentrations)
            concentrations_per_drug_out = []
            intermediate_scores_local = []
            cell_numbers_concentrations = []
            ## cell numbers normalized for pharmacoscopy plots
            normalized_cell_numbers_per_drug = []
            normalized_cell_number_std_devs_per_drug = []
            normalized_cell_number_points_per_drug = []
            ## rbfs for plots
            rbfs_per_drug = []
            rbfs_std_per_drug = []
            fractions_per_drug = []
            all_rbf_single_vals = []
            ## collecting std devs
            std_devs_cellnums = []
            std_devs_fractions = []
            ## offtargets for plots
            offtarget_fractions_per_drug = []
            offtarget_std_devs_per_drug = []
            offtarget_points_per_drug = []
            if norm_to_plate:
                drugplate = drug_table['Plate'].values[0]
                num_cells_control = cell_num_plate_dict[drugplate]
                ref_values = plate_ref_value_dict[drugplate]
                _, cellnum_std_norm1 = cell_num_norm_plate_dict[drugplate]
                offtarget_ref_values = plate_offtarget_frac_dict[drugplate]
            else:
                num_cells_control, cellnum_std = self.num_cells_control(return_std=True)
                ref_values = self.calculate_control_fractions(target_column)
                cellnum_std_norm1 = cellnum_std / num_cells_control
                offtarget_ref_values = self.calculate_control_fractions(target_column, offtarget=True)
            for i in range(len(processed_concentrations)):
                concs = ref_concentrations[conc_sort_indices[i]]
                p_conc = processed_concentrations[conc_sort_indices[i]]
                conc_table = drug_table[drug_table['Concentration'].isin(concs)]
                unique_conc_wells = conc_table['Metadata_Well'].unique()
                if (conc_table.shape[0] > 0):
                    wells_plates = self.make_wells_plate_tuples(conc_table)
                    fractions = []
                    cell_numbers_wells = []
                    normalized_cell_numbers_per_conc = []
                    offtarget_fractions = []
                    for well, plate in wells_plates:
                        well_table = conc_table[(conc_table['Metadata_Well'] == well) & (conc_table['Plate'] == plate)]
                        frac = self.calculate_fraction(target_column, conc_table, well, plate)
                        fractions.append(frac)
                        offtarget_fractions.append(self.calculate_fraction(target_column, conc_table, well, plate, offtarget=True))
                        local_well_cellnumber = np.sum(well_table[self.viable_cellnumber_column].values) - np.sum(well_table['NumberOfCells_{}'.format(target_column)].values)
                        cell_numbers_wells.append(local_well_cellnumber)
                        normalized_cell_numbers_per_conc.append(float(local_well_cellnumber) / num_cells_control)
                    if np.std(normalized_cell_numbers_per_conc) <= 2*(cellnum_std_norm1):
                        concentrations_per_drug_out.append(p_conc)
                        normalized_cell_numbers_per_drug.append(np.nanmean(normalized_cell_numbers_per_conc))
                        normalized_cell_number_points_per_drug.append(normalized_cell_numbers_per_conc)
                        normalized_cell_number_std_devs_per_drug.append(np.nanstd(normalized_cell_numbers_per_conc))
                        cell_numbers_concentrations.append(np.nanmean(cell_numbers_wells))
                        ## append Target fractions
                        mean_fraction = np.nanmean(fractions)
                        fractions_per_drug.append(mean_fraction)
                        std_devs_cellnums.append(np.nanstd(cell_numbers_wells))
                        std_devs_fractions.append(np.nanstd(fractions))
                        all_fractions.append(fractions)
                        keep_compounds.append(drug)
                        
                        rbf = self.relative_population_fraction(mean_fraction, ref_values)
                        rbf_single_vals = [self.relative_population_fraction(f, ref_values) for f in fractions]
                        all_rbf_single_vals.append(rbf_single_vals)
                        rbf_std = np.nanstd(rbf_single_vals)
                        rbfs_per_drug.append(rbf)
                        all_rbfs_testing.append(rbf)
                        rbfs_std_per_drug.append(rbf_std)
                        intermediate_score = 1. - rbf
                        ## append offtarget fractions
                        mean_offtarget_fraction = np.nanmean(offtarget_fractions)
                        relative_offtarget_fraction = self.relative_population_fraction(mean_offtarget_fraction, offtarget_ref_values)
                        r_offtarget_single_vals = [self.relative_population_fraction(f, offtarget_ref_values) for f in offtarget_fractions]
                        r_offtarget_std = np.nanstd(r_offtarget_single_vals)
                        offtarget_fractions_per_drug.append(relative_offtarget_fraction)
                        offtarget_points_per_drug.append(r_offtarget_single_vals)
                        offtarget_std_devs_per_drug.append(r_offtarget_std)
                        if intermediate_score > max_intermediate:
                            max_intermediate = intermediate_score
                            if self.verbose:
                                print('Found new max intermediate of {} for drug {}'.format(max_intermediate, drug))
                        if intermediate_score < min_intermediate:
                            min_intermediate = intermediate_score
                            if self.verbose:
                                print('Found new min intermediate of {} for drug {}'.format(min_intermediate, drug))
                        intermediate_scores_local.append(intermediate_score)
            all_concentrations.append(concentrations_per_drug_out)
            self.cell_number_dict[drug] = (concentrations_per_drug_out, cell_numbers_concentrations, std_devs_cellnums)
            self.dose_fraction_dict[drug] = (concentrations_per_drug_out, fractions_per_drug, std_devs_fractions)
            self.normalized_cell_number_dict[drug] = (concentrations_per_drug_out, normalized_cell_numbers_per_drug, normalized_cell_number_std_devs_per_drug, normalized_cell_number_points_per_drug)
            self.rbf_dict[drug] = (concentrations_per_drug_out, rbfs_per_drug, rbfs_std_per_drug, all_rbf_single_vals)
            self.normalized_offtarget_dict[drug] = (concentrations_per_drug_out, offtarget_fractions_per_drug, offtarget_std_devs_per_drug, offtarget_points_per_drug)
            intermediate_scores.append(intermediate_scores_local)
            
        ### showing rbf distribution for testing purposes
        #plt.hist(all_rbfs_testing, bins=50)
        #plt.show()
        self.compounds = list(set(keep_compounds))
        self.compounds.sort()
        if self.verbose:
            print('Keeping {} compounds'.format(len(self.compounds)))
        for i in range(len(self.compounds)):
            local_concentrations = all_concentrations[i]
            fractions = all_fractions[i]
            drug = self.compounds[i]
            scores = [f for f in intermediate_scores[i]]
            self.dose_response_dict[drug] = (local_concentrations, scores)
        self.calculated_pharmacoscopy_singles = True
        if return_dictionaries:
            return self.dose_response_dict.copy(), self.dose_fraction_dict.copy(), self.rbf_dict.copy()
        else:
            return
        
    def calculate_z_score_aucs(self, target_column):
        z_scores = dict()
        norm_num_cells, norm_std_cells = self.num_cells_control(normto1=True)
        for drug in self.compounds:
            concentrations, cell_numbers, _, _ = self.normalized_cell_number_dict[drug]
            concentrations, rbfs, rbfs_stds, _ = self.rbf_dict[drug]
            
            concentrations = np.array(concentrations).reshape(-1,1) / np.amax(concentrations)
            cell_numbers = np.array(cell_numbers).reshape(-1,1)
            score = (1. - np.mean(rbfs)) / np.mean(np.std(rbfs))
            z_scores[drug] = score
        return z_scores

    def calculate_pharmacoscopy_aucs(self, target_column, return_scores=False, sigmoid=True, return_all_data=False):
        #self.calculate_pharmacoscopy_scores(target_column)
        self.pharmacoscopy_aucs = dict()
        if self.verbose:
            print('########################################################')
        max_score = np.amax([np.amax(self.dose_response_dict[drug][1]) for drug in self.compounds if (len(self.dose_response_dict[drug][1]) >= 1)])
        self.curve_params = dict()
        if self.verbose:
            print('Calculating auc with max score {}'.format(max_score))
        for drug in self.compounds:
            concentrations, scores_singles = self.dose_response_dict[drug]
            _, cell_numbers, _, _ = self.normalized_cell_number_dict[drug]
            concentrations, rbfs, _, rbf_replicate_vals = self.rbf_dict[drug]
            if len(concentrations) == 0:
                self.pharmacoscopy_aucs[drug] = -1
            else:
                if sigmoid:
                    #score, params, _ = self.score_drug_sigmoid(concentrations, rbf_replicate_vals, [1.]*32, score_certainty=True)
                    score, params, _ = self.score_drug_sigmoid(concentrations, rbf_replicate_vals, [1.]*3, score_certainty=True, allow_negative_values=True)
                    self.pharmacoscopy_aucs[drug] = score
                    self.curve_params[drug] = params
                else:
                    concentrations = np.array(concentrations).reshape(-1,1) / np.amax(concentrations)
                    cell_numbers = np.array(cell_numbers).reshape(-1,1)
                    scores_singles = np.array(scores_singles).reshape(-1,1) / max_score
                    score = np.nanmean(scores_singles)
            DEBUG = False
            self.pharmacoscopy_aucs[drug] = score
        self.calculated_pharmacoscopy_aucs = True
        if return_all_data:
            return self.pharmacoscopy_aucs.copy(), self.curve_params.copy()
        elif return_scores:
            return self.pharmacoscopy_aucs.copy()
        else:
            return
    
    def sigmoid_aucs_rbf(self, target_column, return_scores=False, threaded=True):
        #self.calculate_pharmacoscopy_scores(target_column)
        self.calculated_pharmacoscopy_aucs = False
        self.calculated_abs_target = False
        self.pharmacoscopy_aucs = dict()
        self.curve_params = dict()
        self.rbf_score_sq_err = dict()
        self.abs_err_dict = dict()
        drug_index = 0
        threads = []
        for drug in self.compounds:
            if DEBUG:
                print('#{} Fitting sigmoid for '.format(drug_index+1) + drug)
            target_plate = self.drug_to_plate[drug]
            if threaded:
                t = Thread(target=self.score_and_add_to_dicts, args=(self.rbf_dict, drug, target_column, True))
                threads.append(t)
                t.start()
            else:
                self.score_and_add_to_dicts(self.rbf_dict, drug, target_column, do_rbf=True)
        if threaded:
            for t in threads:
                t.join()
        self.calculated_rbf_abs = True
        if return_scores:
            return self.pharmacoscopy_aucs.copy()
        else:
            return

    def score_and_add_to_dicts(self, input_dictionary, compound, target_column, do_rbf=False):
        vals = input_dictionary[compound]
        concentrations = vals[0]
        cellnumbers = vals[1]
        rep_cellnumbers = vals[3]


        sort_indices = np.argsort(concentrations)
        concentrations_for_scoring = [concentrations[index] for index in sort_indices]
        rep_cellnumbers_for_scoring = [rep_cellnumbers[i] for i in sort_indices]


        plate = self.drug_to_plate[compound]
        ref_cell_numbers = self.num_cells_control_plate(plate, target_column=target_column)
        if (np.mean(rep_cellnumbers[0]) <= 2) and (np.mean(ref_cell_numbers) >= 2): ## normalize ref cell numbers if rep was normalized
            ref_cell_numbers = [f / np.mean(ref_cell_numbers) for f in ref_cell_numbers]
        if do_rbf:
            rbf_ref_values = [1.]*3
            score, opt, sq_err, abs_err = self.score_drug_sigmoid(concentrations, rep_cellnumbers, rbf_ref_values, score_certainty=True, score_certainty_abs=True, allow_negative_values=True)
            self.pharmacoscopy_aucs[compound] = score
            self.rbf_score_sq_err[compound] = sq_err
            if DEBUG:
                print('Finished fitting sigmoid. Score is {}'.format(score))
            self.pharmacoscopy_aucs[compound] = score
        else:
            score, opt, abs_err = self.score_drug_sigmoid(concentrations_for_scoring, rep_cellnumbers_for_scoring, ref_cell_numbers, score_certainty=True)
            self.abs_auc_dict[compound] = score
            self.abs_score_sq_err[compound] = abs_err
        self.abs_err_dict[compound] = abs_err
        self.curve_params[compound] = opt
        return


    def sigmoid_aucs_abs(self, target_column, input_dictionary, threaded=False):
        scores = []
        self.abs_auc_dict = dict()
        self.curve_params = dict()
        self.abs_score_sq_err = dict()
        self.abs_err_dict = dict()
        if self.verbose:
            print('Starting to score {} compounds. Threading is set to {}'.format(len(self.compounds), threaded))
        #self.calculated_pharmacoscopy_aucs = False
        self.calculated_rbf_abs = False
        threads = []
        for compound in self.compounds:
            if self.verbose:
                print(f'Processing drug {compound}', flush=True)
            if compound in input_dictionary.keys():
                if threaded:
                    t = Thread(target=self.score_and_add_to_dicts, args=(input_dictionary, compound, target_column))
                    threads.append(t)
                    t.start()
                else:
                    self.score_and_add_to_dicts(input_dictionary, compound, target_column)
        if threaded:
            for t in threads:
                t.join()
        self.calculated_abs_target = True
        return self.abs_auc_dict.copy()

    
    def score_drug_sigmoid(self, sorted_concentrations, sorted_nested_replicate_values, reference_values, score_certainty=False, score_certainty_abs=False, allow_negative_values=False):
        """
        fit sigmoud curve and calculate score based on it
        @params
        sorted_concentrations : drug concentrations, extracted from the output table
        sorted_nested_replicate_values : replicate values of relative blast fraction or cell(type) fraction relative to neg. ctrl.
        reference_values :  values in negative control, either RBF in neg ctrl or normalized cell numbers in neg ctrl
                            These will be used as the values for the concentratio of 0
        """
        if len(sorted_concentrations) == 0:
            if score_certainty and score_certainty_abs:
                return -1, [0,0,0], 0, 0
            elif score_certainty or score_certainty_abs:
                return -1, [0,0,0], 0
            else:
                return -1, [0,0,0]
        else:
            ## make flattened list of sorted nested replicate values for fit error calculation
            values_to_use = reference_values + [f for sublist in sorted_nested_replicate_values for f in sublist]
            temp_concentrations = [[sorted_concentrations[i]] * len(sorted_nested_replicate_values[i]) for i in range(len(sorted_nested_replicate_values))]
            concentrations_to_use = [0.] * len(reference_values) + [f for sublist in temp_concentrations for f in sublist]
            ## transform both to numpy arrays for masking and curve fitting
            value_arr = np.array(values_to_use)
            conc_arr = np.array(concentrations_to_use)
            if np.sum(conc_arr) > 0:
                conc_arr /= np.nanmax(conc_arr)
            else:
                if self.verbose:
                    print('Found drugs with only zero concentrations')
                    print('Scoring drug with the following concentrations and reference values')
                    print(sorted_concentrations)
                    print(sorted_nested_replicate_values)
            ## calculate score based on mean dose response values
            mean_dose_response_values = [np.nanmean(f) for f in sorted_nested_replicate_values]
            #print('calcuating auc for drug with mean dose response values {} and concentrations {}'.format(mean_dose_response_values, sorted_concentrations))
            interpolated_dose_response_values = interpolate_for_ref_concentrations(mean_dose_response_values, sorted_concentrations, sorted_ref_concentrations=self.reference_concentrations, initial_value=1., do_clip=not allow_negative_values)
            #print('Working with interpolated values {}'.format(interpolated_dose_response_values))
            score = calc_auc(interpolated_dose_response_values, ref_concentrations=self.reference_concentrations)
            if (score is None) or np.isnan(score):
                print('Got score of None for dr values {}'.format(mean_dose_response_values))
            ## fit sigmoid to data and use the fit to calculate the error
            opt, cov = fit_sigmoid(conc_arr, value_arr)
            #sig_values = self.drug_sigmoid(conc_arr, opt[0], opt[1], opt[2])
            sig_values = drug_sigmoid(conc_arr, opt[0], opt[1], opt[2])
            sig_values_for_score = sig_values[conc_arr > 0]
            measurements_for_score = value_arr[conc_arr > 0]
            if DEBUG:
                print('Using the following unique values to inform scoring and MSE')
                print('all concs : {}'.format(np.unique(conc_arr)))
                print('Concs > 0 : {}'.format(np.unique(conc_arr[conc_arr > 0])))
                print('sig values: {}'.format(np.unique(sig_values)))
                print('sig values conc > 0: {}'.format(np.unique(sig_values[conc_arr > 0])))
            if self.verbose:
                print('Calculated sigmoid score of {}'.format(score))
            if score_certainty and score_certainty_abs:
                #mean_squared_error = np.mean((np.array(sig_values_for_score).flatten() - np.array(measurements_for_score).flatten())**2)
                #mean_abs_error = np.mean(np.abs(np.array(sig_values_for_score).flatten() - np.array(measurements_for_score).flatten()))
                #return score, opt, mean_squared_error, mean_abs_error
                out_err = calcuate_auc_std_err(sorted_concentrations, sorted_nested_replicate_values)
                return score, opt, out_err, out_err
            elif score_certainty:
                #mean_squared_error = np.mean((np.array(sig_values_for_score).flatten() - np.array(measurements_for_score).flatten())**2)
                #return score, opt, mean_squared_error
                out_err = calcuate_auc_std_err(sorted_concentrations, sorted_nested_replicate_values)
                return score, opt, out_err
            elif score_certainty_abs:
                mean_abs_error = np.mean(np.abs(np.array(sig_values_for_score).flatten() - np.array(measurements_for_score).flatten()))
                return score, opt, mean_abs_error
            else:
                return score, opt


    def make_pharmacoscopy_certainty_score_table(self, target_column, out_path):
        self.calculate_pharmacoscopy_aucs(target_column)
        scores = []
        prec_scores = []
        for drug in self.compounds:
            scores.append(self.pharmacoscopy_aucs[drug])
            concs, rbfs, rbfs_std, _ = self.rbf_dict[drug]
            _, norm_cell_nums, norm_cell_nums_std, _ = self.normalized_cell_number_dict[drug]
            cells_std_normed = np.array(norm_cell_nums_std) / np.array(norm_cell_nums)
            rbfs_std_normed = np.array(rbfs_std) / np.array(rbfs)
            rbf_prec = 1./(np.sum(rbfs_std_normed) / len(concs))
            cell_num_prec = 1./(np.sum(cells_std_normed) / len(concs))
            prec_score = (rbf_prec + cell_num_prec) / 2
            prec_scores.append(prec_score)
        print('Minimum precision score is {} and maximum precision score is {}'.format(np.nanmin(prec_scores), np.nanmax(prec_scores)))
        df_dict = dict()
        df_dict['Drug'] = self.compounds
        df_dict['Score'] = scores
        df_dict['PrecisionScore'] = prec_scores
        df = pd.DataFrame(df_dict)
        df.to_csv(out_path)
        return


                
    def sort_drug_dict(self, drug_dict, is_single_val=True, bottom_clip_value=None):
        all_drugs = []
        all_scores = []
        for drug in drug_dict.keys():
            all_drugs.append(drug)
            score = drug_dict[drug]
            if is_single_val:
                all_scores.append(score)
            else:
                all_scores.append(np.nanmean(score))
        sort_indices = np.argsort(all_scores)[::-1]
        out_drugs = []
        out_scores = []
        for index in sort_indices:
            drug = all_drugs[index]
            out_drugs.append(drug)
            if bottom_clip_value is None:
                out_scores.append(drug_dict[drug])
            else:
                out_scores.append(np.clip(drug_dict[drug], bottom_clip_value, None))
        return out_drugs, out_scores
        
        
    def inhibition_score(self, value, all_ref_values):
        return value - np.mean(all_ref_values)
    
    def relative_population_fraction(self, population_fraction, ref_population_fractions):
        rpf = population_fraction / np.mean(ref_population_fractions)
        return rpf        
            
    def make_inhibition_plot(self, drugs_per_page=10, drugs_per_column=5, plot_curves=False):
        """
        This function plots an inhibition plot for all 117 drugs in the assay.
        This will be returned as a list of figures with the plots for 20 drugs each.
        """
        num_pages = int(len(self.compounds) / drugs_per_page) + 1
        drug_index = 0
        out_plots = []
        max_inhibition = np.amax([np.amax(self.dose_response_dict[drug][1]) for drug in self.compounds if (len(self.dose_response_dict[drug][1]) >= 1)])
        for p in range(num_pages):
            fig, ax = plt.subplots(nrows=drugs_per_column,ncols=2, figsize=(10, 5*drugs_per_column))
            for k in range(drugs_per_page):
                if (drug_index + k) == len(self.compounds):
                    break
                compound = self.compounds[drug_index + k]
                concentrations, inhibitions = self.dose_response_dict[compound]
                inhibitions = np.array(inhibitions).flatten() / max_inhibition
                if k < drugs_per_column:
                    col_index = 0
                    row_index = k
                else:
                    col_index = 1
                    row_index = k - drugs_per_column
                log_concentrations = - np.log10(concentrations)
                conc_string_list = [str(f) for f in concentrations]
                ax[row_index,col_index].scatter(log_concentrations, inhibitions)
                ax[row_index,col_index].set_xticks(log_concentrations)
                ax[row_index,col_index].set_xticklabels(conc_string_list)
                ax[row_index,col_index].set_title(compound)
            drug_index += drugs_per_page
            out_plots.append(fig)
        self.plots = out_plots
        return self.plots

    def calculate_abs_scores(self, target_column, norm_to_plate=True, sigmoid=False, return_sqerr=False, return_replicate_values=False):
        abs_dose_response_dict = self.make_response_dict_abs(target_column, norm_to_plate=norm_to_plate)
        if sigmoid:
            auc_dict = self.sigmoid_aucs_abs(target_column, abs_dose_response_dict)
        else:
            auc_dict = self.abs_avg_score(abs_dose_response_dict)
        if return_sqerr:
            if return_replicate_values:
                return auc_dict, out_sqerr, abs_dose_response_dict
            out_sqerr = self.abs_score_sq_err.copy()
            return auc_dict, out_sqerr
        elif return_replicate_values:
            return auc_dict, abs_dose_response_dict
        else:
            return auc_dict

    
    def diff_tox_abs(self, target_column1, target_column2, norm_to_plate=True):
        abs_response_dict1 = self.make_response_dict_abs(target_column1, norm_to_plate=norm_to_plate)
        self.reset_calculation_flags()
        abs_response_dict2 = self.make_response_dict_abs(target_column2, norm_to_plate=norm_to_plate)
        self.reset_calculation_flags()
        aucs_1 = self.sigmoid_aucs_abs(target_column1, abs_response_dict1)
        auc_errs1 = self.abs_score_sq_err.copy()
        self.reset_calculation_flags()
        aucs_2 = self.sigmoid_aucs_abs(target_column2, abs_response_dict2)
        auc_errs2 = self.abs_score_sq_err.copy()
        return self.calculate_diff_scores(aucs_1, aucs_2, sqerr_dicts=[auc_errs1, auc_errs2])


    def calculate_diff_scores(self, auc_dict1, auc_dict2, sqerr_dicts=None):
        out_dict = dict()
        for compound in self.compounds:
            score1 = auc_dict1[compound]
            score2 = auc_dict2[compound]
            if sqerr_dicts is None:
                diff_score = (score1 - score2)
            else:
                sqerr1 = sqerr_dicts[0][compound]
                sqerr2 = sqerr_dicts[1][compound]
                diff_score = (score1 - score2) / np.mean([sqerr1, sqerr2])
            out_dict[compound] = diff_score
        return out_dict

    def single_drug_barplot(self, drug, in_ax, ax_title, do_abs, norm_to_plate,\
         plate_norm_all_dict=None, plate_norm_target_dict=None, use_dict=None,\
             sigmoid=False, show_std_devs_target=False, comparator_dict=None, change_color=False):
        if not do_abs:
            concs, fractions, fraction_devs, points = self.rbf_dict[drug]
            concs, numbers, number_devs, num_points = self.normalized_cell_number_dict[drug]
        else:
            if use_dict is None:
                concs, fractions, fraction_devs, points = self.abs_dose_response_dict[drug]
            else:
                concs, fractions, fraction_devs, points = use_dict[drug]
            if not (comparator_dict is None):
                concs, numbers, number_devs, num_points = comparator_dict[drug]
                do_abs = False
        if len(points) > 0:
            if norm_to_plate:
                plate = self.table[self.table['Drug'] == drug]['Plate'].values[0]
                norm_cell_num_all, norm_cell_num_std = plate_norm_all_dict[plate]
                upper_line_all = norm_cell_num_all + norm_cell_num_std
                lower_line_all = norm_cell_num_all - norm_cell_num_std
                norm_cell_num_target, norm_cell_num_target_std = plate_norm_target_dict[plate]
                upper_line_target = norm_cell_num_target + norm_cell_num_target_std
                lower_line_target = norm_cell_num_target - norm_cell_num_target_std
            heights = []
            yerrs = []
            colors = []
            concs_for_sig = []
            indices_for_sig = []
            if not do_abs:
                indices = list(range(len(concs) * 2))
                all_points = []
                point_indices = []
            else:
                all_points = []
                point_indices = []
                indices = list(range(len(concs)))
            for c in range(len(concs)):
                heights.append(fractions[c])
                yerrs.append(fraction_devs[c])
                colors.append('#E07526') ## orange for target cells
                points_to_add = points[c]
                if not do_abs:
                    heights.append(numbers[c])
                    points_to_add = points[c] + num_points[c]
                    point_indices += [2*indices[c]] * len(points[c]) + [(2*indices[c]) + 1] * len(num_points[c])
                    ## append y errs for error bars
                    yerrs.append(number_devs[c])
                    ## append colors for color list
                    colors.append('#6078EB') ## blue for all cells (or for target cells if we do absolute scoring)
                    if sigmoid:
                        concs_for_sig += [concs[c]] * len(points[c]) + [concs[c]] * len(num_points[c])
                        indices_for_sig += [2*indices[c]] * len(points[c]) 
                        if c < (len(concs) - 1):
                            mean_conc = np.nanmean([concs[c], concs[c+1]])
                            concs_for_sig += [mean_conc] * len(points[c]) + [mean_conc] * len(num_points[c])
                            indices_for_sig += [(2*indices[c])+0.5] * len(points[c]) + [(2*indices[c]) + 0.5] * len(num_points[c])
                        indices_for_sig += [(2*indices[c]) + 1] * len(num_points[c])
                else:
                    point_indices += [indices[c]] * len(points[c])
                    if sigmoid:
                        concs_for_sig += [concs[c]] * len(points[c])
                        indices_for_sig += [indices[c]] * len(points[c])
                        if c < (len(concs) - 1):
                            mean_conc = np.nanmean([concs[c], concs[c+1]])
                            concs_for_sig += [mean_conc] * len(points[c])
                            indices_for_sig += [indices[c] + .5] * len(points[c])
                all_points += points_to_add
            if sigmoid:
                params = self.curve_params[drug]
                if (len(concs_for_sig) == 0) or (np.nan in concs_for_sig):
                    sig_curve = None
                else:
                    conc_arr = np.array(concs_for_sig)# / np.nanmax(concs_for_sig)
                    #sig_curve = self.drug_sigmoid(conc_arr, params[0], params[1], params[2])
                    sig_curve = drug_sigmoid(conc_arr, params[0], params[1], params[2])
            else:
                sig_curve = None
            in_ax.bar(indices, heights, yerr=yerrs, color=colors, alpha=0.5)
            in_ax.plot(indices, [norm_cell_num_all]*len(indices), color='lightgrey')
            #in_ax.plot(indices, [upper_line_all]*len(indices), color='lightgrey')
            #in_ax.plot(indices, [lower_line_all]*len(indices), color='lightgrey')
            in_ax.scatter(point_indices, all_points, color='black')
            if sigmoid and (not (sig_curve is None)):
                in_ax.plot(indices_for_sig, sig_curve, color='red')
            if (not do_abs) or show_std_devs_target:
                in_ax.plot(indices, [norm_cell_num_target]*len(indices), color='#E07526', alpha=0.6)
                #in_ax.plot(indices, [upper_line_target]*len(indices), color='#E07526', alpha=0.6)
                #in_ax.plot(indices, [lower_line_target]*len(indices), color='#E07526', alpha=0.6)
                in_ax.set_xticks(np.linspace(0.5, indices[-1]-0.5, num=len(concs)))
            else:
                in_ax.set_xticks(np.linspace(0, indices[-1], num=len(concs)))
            in_ax.set_xticklabels([str(f) + ' uM' for f in concs])
            if change_color:
                in_ax.set_title(ax_title, {'color' : 'red'}, fontsize=14)
            else:
                in_ax.set_title(ax_title, fontsize=14)
            plt.setp(in_ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
            return in_ax
        else:
            if DEBUG:
                print('No points found for drug ' + drug)
            return in_ax
        

    def pharmacoscopy_style_plot(self, target_column, input_figtitle, savetitle=None, norm_to_plate=True,\
         add_scatter=False, plot_all_bars=True, do_abs=False, sigmoid=True, plot_offtarget=False,\
          sigmoid_filter_threshold=None, sigmoid_filter_highlight=False):
        response_dict_target_cohort = None
        target_err_dict = None
        response_dict_comparator_cohort = None
        auc_dict_target_cohort = None
        if isinstance(target_column, list):
            ### calculate scores for the two populations
            print('Making differential plot for populations {} and {}'.format(target_column[0], target_column[1]))
            norm_all_dicts = []
            norm_target_dicts = []
            auc_dictionaries = []
            comparator_column = target_column[1]
            for i, t_col in enumerate(target_column):
                if norm_to_plate:
                    _, plate_norm_all_dict, plate_norm_target_dict, _ = self.make_plate_dictionaries(t_col)
                    norm_all_dicts.append(plate_norm_all_dict)
                    norm_target_dicts.append(plate_norm_target_dict)
                else:
                    plate_norm_all_dict = None
                    plate_norm_target_dict = None
                if not self.calculated_abs_target:
                    auc_dict = self.calculate_abs_scores(t_col, norm_to_plate=norm_to_plate, sigmoid=sigmoid)
                else:
                    auc_dict = self.abs_auc_dict
                auc_dictionaries.append(auc_dict)
                if t_col == comparator_column:	
                    response_dict_comparator_cohort = self.abs_dose_response_dict.copy()
                else:
                    response_dict_target_cohort = self.abs_dose_response_dict.copy()
                    target_err_dict =  self.abs_err_dict.copy()
                    auc_dict_target_cohort = auc_dict.copy()
                self.reset_calculation_flags()


                sorted_drugs, sorted_scores = self.sort_drug_dict(auc_dict)
                norm_cell_num, norm_std = self.num_cells_control(return_std=True)
            ### prepare data for barplots, should be sufficient to store the comparator population at index one in a dict that can be passed
            diff_score_dict = self.calculate_diff_scores(auc_dictionaries[0], auc_dictionaries[1])        
            sorted_drugs, sorted_scores = self.sort_drug_dict(diff_score_dict) 
            self.abs_err_dict = target_err_dict.copy()        
        elif not do_abs:
            if not self.calculated_pharmacoscopy_singles:
                self.calculate_pharmacoscopy_scores(target_column, norm_to_plate=norm_to_plate)
                #self.sigmoid_aucs_rbf(target_column)
            if self.verbose:
                print('Finished calculating pharmacoscopy scores')
            if sigmoid:
                if True:#if not self.calculated_rbf_abs:
                    self.sigmoid_aucs_rbf(target_column)
                    print('Finished calculating sigmoid style scores')
            else:
                if not self.calculated_pharmacoscopy_aucs:
                    self.calculate_pharmacoscopy_aucs(target_column)
                    print('Finished calculating aucs, starting to plot')
            ex_drug = sample(self.compounds, 1)[0]
            print('Example is score {} for drug {}'.format(self.pharmacoscopy_aucs[ex_drug], ex_drug))
            sorted_drugs, sorted_scores = self.sort_drug_dict(self.pharmacoscopy_aucs)
            if norm_to_plate:
                _, plate_norm_all_dict, plate_norm_target_dict, _ = self.make_plate_dictionaries(target_column)
                print('Normalized to plate, going through dict')
                for plate in self.plates:
                    print('all dict: Plate is {} targets are {}'.format(plate, plate_norm_all_dict[plate]))
                    print('target dict: Plate is {} targets are {}'.format(plate, plate_norm_target_dict[plate]))
            else:
                plate_norm_all_dict = None
                plate_norm_target_dict = None
                norm_cell_num_all, norm_std_all = self.num_cells_control(normto1=True)
                norm_cell_num_targets, norm_std_targets = self.num_cells_control(normto1=True, target_column=target_column)
        else:
            if norm_to_plate:
                _, plate_norm_all_dict, plate_norm_target_dict, _ = self.make_plate_dictionaries(target_column)
            else:
                plate_norm_all_dict = None
                plate_norm_target_dict = None
            if not self.calculated_abs_target:
                auc_dict = self.calculate_abs_scores(target_column, norm_to_plate=norm_to_plate, sigmoid=sigmoid)
            else:
                auc_dict = self.abs_auc_dict
            sorted_drugs, sorted_scores = self.sort_drug_dict(auc_dict)
        if not (sigmoid_filter_threshold is None):
            keep_indices = [i for i in range(len(sorted_drugs)) if self.abs_err_dict[sorted_drugs[i]] < sigmoid_filter_threshold]
            if not sigmoid_filter_highlight:
                sorted_drugs = [sorted_drugs[i] for i in keep_indices]
                sorted_scores = [sorted_scores[i] for i in keep_indices]
                keep_indices = list(range(len(sorted_drugs)))    
        else:
            keep_indices = list(range(len(sorted_drugs)))
        sorted_scores_for_titles = sorted_scores.copy()
        #sorted_scores = [(f/max(sorted_scores)) for f in sorted_scores]
        top_drugs = sorted_drugs[:40]
        top_scores = sorted_scores[:40]
        
        print('Creating pharmacoscopy-style plot for {} drugs and {} scores'.format(len(sorted_drugs), len(sorted_scores)))
        #fig = plt.figure(figsize=(28, 14))
        fig = plt.figure(figsize=(30, 16), constrained_layout=True)
        gs = fig.add_gridspec(5,6)
        #annotation_axis = fig.add_subplot(gs[:,0])
        #annotation_axis.text(0, 0, 'Experiment ID: 001 \n Readout: %CD34+CD117+ cells \n Sample type: bone marrow \n channels \n DAPI: DAPI \n CD34: APC \n CD117: PE \n CD3: FITC')
        full_chart_axis = fig.add_subplot(gs[:,:2])
        top_chart_axis = fig.add_subplot(gs[:,2:4])
        sns.barplot(x=sorted_scores, y=sorted_drugs, ax=full_chart_axis, palette='coolwarm')
        full_chart_axis.set_xlabel('Inhibition score', fontsize=18)
        full_chart_axis.set_title('Inhibition scores for all compounds', fontsize=16)
        sns.barplot(x=top_scores, y=top_drugs, ax=top_chart_axis, palette='coolwarm')
        #top_chart_axis.set_title('Top 40 drugs')
        top_chart_axis.set_xlabel('Inhibition score', fontsize=18)
        top_chart_axis.set_title('Inhibition scores for top 40 compounds', fontsize=16)
        top_drug_index = 0
        for i, lbl in enumerate(top_chart_axis.get_ymajorticklabels()):
            lbl.set_fontsize(14)
            if sigmoid_filter_highlight and (not (i in keep_indices)):
                lbl.set_color('red')
        if sigmoid_filter_highlight:        
            for i, lbl in enumerate(full_chart_axis.get_ymajorticklabels()):
                if not (i in keep_indices):
                    lbl.set_color('red')
        for k in range(5):
            for i in range(4, 6):
                if top_drug_index < len(top_drugs):
                    barchart_axis = fig.add_subplot(gs[k,i])
                    drug = top_drugs[top_drug_index]
                    if drug.strip() == 'IPI-145':
                        drug_name = 'Duvelisib (IPI-145)'
                    elif drug.strip() == 'I-BET762':
                        drug_name = 'Molibresib (I-BET762)'
                    elif drug.strip() == 'MLN4924':
                        drug_name = 'Pevonedistat (MLN4924)'
                    elif drug.strip() == 'CB-839':
                        drug_name = 'Telaglenastat (CB-839)'
                    elif drug.strip() == 'EPZ-6438':
                        drug_name = 'Tazemetostat (EPZ-6438)'
                    else:
                        drug_name = drug
                    if sigmoid:
                        if do_abs:
                            #err_val = self.abs_score_sq_err[drug]
                            err_val = self.abs_err_dict[drug]
                        else:
                            #err_val = self.rbf_score_sq_err[drug]
                            err_val = self.abs_err_dict[drug]
                        use_score = sorted_scores_for_titles[top_drug_index]
                        use_title = '#{}: '.format(top_drug_index+1) + drug_name + '\n Score: {} Err: {}'.format(round(use_score, 3), round(err_val, 3))
                    else:
                        use_title = '#{}: '.format(top_drug_index+1) + drug_name
                    if not (top_drug_index in keep_indices):
                        barchart_axis = self.single_drug_barplot(drug, barchart_axis, use_title, do_abs, norm_to_plate,\
                         plate_norm_all_dict=plate_norm_all_dict, plate_norm_target_dict=plate_norm_target_dict,\
                             sigmoid=sigmoid, change_color=True, use_dict=response_dict_target_cohort, comparator_dict=response_dict_comparator_cohort)
                    else:    
                        barchart_axis = self.single_drug_barplot(drug, barchart_axis, use_title, do_abs, norm_to_plate,\
                            plate_norm_all_dict=plate_norm_all_dict, plate_norm_target_dict=plate_norm_target_dict,\
                                sigmoid=sigmoid, use_dict=response_dict_target_cohort, comparator_dict=response_dict_comparator_cohort)
                    top_drug_index += 1
        #fig.suptitle(input_figtitle)
        #fig.tight_layout()
        if savetitle is None:
            fig.show()
        else:
            fig.savefig(savetitle)
        if add_scatter:
            savetitle_scatter = savetitle.split('.')[0] + '_top_drug_scatter.png'
            self.top_drug_scatter_multiplot(top_drugs[:10], savetitle_scatter)
        if plot_all_bars:
            num_steps = len(sorted_drugs)
            if num_steps > 120:
                nrows = 20
                ncols = 15
                fig, ax = plt.subplots(figsize=(60, 70), nrows=nrows, ncols=ncols)
            elif (num_steps >= 100) and (num_steps <= 120):
                nrows = 10
                ncols = 12
                fig, ax = plt.subplots(figsize=(40, 36), nrows=nrows, ncols=ncols)
                
            else:
                n_drugs_per_row = 5
                ncols = n_drugs_per_row
                if (num_steps % n_drugs_per_row) == 0:
                    nrows = int(num_steps / n_drugs_per_row)
                else:
                    nrows = int(num_steps / n_drugs_per_row) + 1
                fig, ax = plt.subplots(figsize=(4*n_drugs_per_row, 3*nrows), nrows=nrows, ncols=n_drugs_per_row)
            top_drug_index = 0
            for i in range(nrows):
                for k in range(ncols):
                    if top_drug_index < num_steps:
                        drug = sorted_drugs[top_drug_index]
                        if sigmoid:
                            if do_abs:
                                err_val = self.abs_score_sq_err[drug]
                            else:
                                #err_val = self.rbf_score_sq_err[drug]
                                err_val = self.abs_err_dict[drug]
                            use_score = sorted_scores_for_titles[top_drug_index]
                            use_title = '#{}: '.format(top_drug_index+1) + drug + '\n Score: {} Err: {}'.format(round(use_score, 3), round(err_val, 3))
                        else:
                            use_title = '#{}: '.format(top_drug_index+1) + drug
                        if nrows > 1:
                            ax[i,k] = self.single_drug_barplot(drug, ax[i,k], use_title, do_abs, norm_to_plate, plate_norm_all_dict=plate_norm_all_dict,\
                                                                plate_norm_target_dict=plate_norm_target_dict, sigmoid=sigmoid, use_dict=response_dict_target_cohort, comparator_dict=response_dict_comparator_cohort)
                        else:
                            ax[k] = self.single_drug_barplot(drug, ax[k], use_title, do_abs, norm_to_plate, plate_norm_all_dict=plate_norm_all_dict,\
                                                              plate_norm_target_dict=plate_norm_target_dict, sigmoid=sigmoid, use_dict=response_dict_target_cohort, comparator_dict=response_dict_comparator_cohort)
                        top_drug_index += 1
            fig.tight_layout()
            bar_savetitle = savetitle.split('.')[0] + '_top{}_bars'.format(top_drug_index+1) + '.pdf'
            fig.savefig(bar_savetitle)
        if isinstance(target_column, list):
            return auc_dict_target_cohort
        else:
            return None     


    def get_plots(self):
        return self.plots
    
    def top_drug_scatter_multiplot(self,top_drugs, savename, x_col='Log_PE_Intensity', y_col='Log_APC_Intensity'):
        control_table = self.table[self.table['Drug'] == 'DMSO']
        control_x = control_table[x_col].values
        control_y = control_table[y_col].values
        x_vals_all = []
        y_vals_all = []
        concentrations_all = []
        for drug in top_drugs:
            drug_table = self.table[self.table['Drug'] == drug]
            concentrations_per_drug_in = list(set(drug_table['Concentration'].values))
            x_vals_per_drug = []
            y_vals_per_drug = []
            concentrations_per_drug_in.sort()
            concentrations_per_drug_out = []
            for conc in concentrations_per_drug_in:
                conc_table = drug_table[drug_table['Concentration'] == conc]
                if conc_table.shape[0] > 0:
                    concentrations_per_drug_out.append(conc)
                    x_vals_per_drug.append(conc_table[x_col].values)
                    y_vals_per_drug.append(conc_table[y_col].values)
            x_vals_all.append(x_vals_per_drug)
            y_vals_all.append(y_vals_per_drug)
            concentrations_all.append(concentrations_per_drug_out)
        concentration_num = max([len(f) for f in concentrations_all])
        fig, ax = plt.subplots(ncols=concentration_num, nrows=len(top_drugs), figsize=(10*concentration_num,10*len(top_drugs)))
        for i in range(len(top_drugs)):
            drug = top_drugs[i]
            y_vals_per_drug_local = y_vals_all[i]
            x_vals_per_drug_local = x_vals_all[i]
            concentrations_per_drug_local = concentrations_all[i]
            for k in range(len(concentrations_per_drug_local)):
                ax[i,k].scatter(control_x, control_y, color='grey', alpha=0.05)
                ax[i,k].scatter(x_vals_per_drug_local[k], y_vals_per_drug_local[k], color='red', alpha=0.3)
                ax[i,k].set_title('{} - {} uM'.format(drug, concentrations_per_drug_local[k]))
                ax[i,k].set_xlabel(x_col)
                ax[i,k].set_ylabel(y_col)
        fig.tight_layout()
        fig.savefig(savename)
        
    def write_pharmacoscopy_aucs(self, writepath):
        out_text = 'Drug\tScore'
        sorted_drugs, sorted_scores = self.sort_drug_dict(self.pharmacoscopy_aucs)
        for drug, score in list(zip(sorted_drugs, sorted_scores)):
            out_text += drug + '\t' + str(score) + '\n'
        out_text = out_text.strip()
        outfile = open(writepath, 'w')
        outfile.write(out_text)
        outfile.close()
        return
    
    def scoring_dict_to_df(self, in_dict, column_names=['Drug', 'Score'], do_sort=True):
        if do_sort:
            first_column, second_column = self.sort_drug_dict(in_dict)
        else:
            first_column = list(in_dict.keys())
            second_column = [in_dict[f] for f in first_column]
        df_dict = dict()
        df_dict[column_names[0]] = first_column
        df_dict[column_names[1]] = second_column
        df = pd.DataFrame(df_dict)
        return df

    def get_wells_for_drug(self, drug, concentration):
        subtable = self.table[(self.table['Drug'] == drug) & (self.table['Concentration'] == concentration)]
        wells = list(set(subtable['Metadata_Well'].values))
        r_c_pairs = [well_string_to_nums(well) for well in wells]
        return r_c_pairs

    
    def pos_ctrl_mini_plot(self, target_column, savename=None, norm_to_plate=True, pos_control_drugs=['Cytarabine+Daunorubicin', 'Cytarabine+Mitoxantrone', 'Cytarabine+Etopside']):
        num_drugs_found = 0
        for drug in pos_control_drugs:
            if drug in self.table['Drug'].values:
                num_drugs_found += 1
        if not (num_drugs_found == len(pos_control_drugs)):
            print('Pos ctrl drugs not found! Aborting!')
            return
        z_prime_all = self.z_prime_raw('Viable', pos_control_drugs=['Cytarabine+Daunorubicin'], pos_control_concentrations=['5+5'])
        z_prime_target = self.z_prime_raw(target_column, pos_control_drugs=['Cytarabine+Daunorubicin'], pos_control_concentrations=['5+5'])
        ### plot all left, rest right
        _, plate_norm_all_dict, plate_norm_target_dict, _ = self.make_plate_dictionaries(target_column)
        auc_dict_target = self.calculate_abs_scores(target_column, norm_to_plate=norm_to_plate)
        auc_dict_all = self.calculate_abs_scores(self.viable_cellnumber_column, norm_to_plate=norm_to_plate)
        use_dict_target = self.make_response_dict_abs(target_column).copy()
        use_dict_all = self.make_response_dict_abs('Viable').copy()
        norm_cell_num, norm_std = self.num_cells_control(return_std=True)
        barplot_size = 4
        fig, ax = plt.subplots(nrows=len(pos_control_drugs), ncols=2, figsize=(barplot_size*2,len(pos_control_drugs)*barplot_size))
        for i in range(len(pos_control_drugs)):
            drug = pos_control_drugs[i]
            if i == 0:
                use_title_all = 'All cells {} \n Z\'= {}'.format(drug, round(z_prime_all, 3))
                use_title_target = '{} cells {} \n Z\'= {}'.format(target_column, drug, round(z_prime_target, 3))
            else:
                use_title_all = drug + ' All'
                use_title_target = drug  + ' Target'
            ax[i,0] = self.single_drug_barplot(drug, ax[i,0], use_title_all, True, norm_to_plate, use_dict=use_dict_all, plate_norm_all_dict=plate_norm_all_dict, plate_norm_target_dict=plate_norm_target_dict)
            ax[i,1] = self.single_drug_barplot(drug, ax[i,1], use_title_target, True, norm_to_plate, use_dict=use_dict_target, plate_norm_all_dict=plate_norm_all_dict, plate_norm_target_dict=plate_norm_target_dict, show_std_devs_target=True)
        fig.tight_layout()
        if savename is None:
            fig.show()
        else:
            fig.savefig(savename)
        return
    
    def make_wells_plate_tuples(self, in_table):
        all_plates = in_table['Plate'].values
        all_wells = in_table['Metadata_Well'].values
        wells_plates = []
        for well, plate in list(zip(all_wells, all_plates)):
            if (not (well, plate) in wells_plates):
                wells_plates.append((well, plate))
        wells_plates.sort()
        return wells_plates
    
    def calculate_dose_responses(self, target_column):
        self.dose_response_dict = dict()
        ref_values = self.calculate_control_fractions(target_column)
        all_fractions = []
        for drug in self.compounds:
            drug_table = self.table[self.table['Drug'] == drug]
            concentrations = list(set(drug_table['Concentration'].values))
            processed_concentrations, ref_concentrations = self.process_concentrations(concentrations)
            conc_sort_indices = np.argsort(processed_concentrations)
            inhibitions = []
            use_concs = []
            for i in range(len(processed_concentrations)):
                conc = concentrations[conc_sort_indices[i]]
                use_concs.append(conc)
                concs = ref_concentrations[conc_sort_indices[i]]
                conc_table = drug_table[drug_table['Concentration'].isin(concs)]
                wells_plates = self.make_wells_plate_tuples(conc_table)
                fractions = []
                for well, plate in wells_plates:
                    frac = self.calculate_fraction(target_column, conc_table, well, plate)
                    fractions.append(frac)
                mean_fraction = np.mean(fractions)
                all_fractions.append(mean_fraction)
                score = self.inhibition_score(mean_fraction, ref_values)
                inhibitions.append(score)
            #processed_concentrations.sort()
            self.dose_response_dict[drug] = (use_concs, inhibitions)
        return
    
    def make_response_dict_abs(self, target_column, norm_to_plate=True):
        """
        Generate dictionary that stores the dose response data for each drug
        Each drug is a key in the dictionary, the values are tuples of the form: (concentrations, cellnumbers, cellnumbers_std_devs, replicate_values)
        concetrations are unique concentrations as floats
        cellnumbers are averages per concentration
        cellnumbers_std_devs are the standard deviations of the cellnumbers per concentration
        replicate_values is a nested list of cellnumbers per well in the same order as the concentrations
        """
        self.abs_dose_response_dict = dict()
        if norm_to_plate:
            c_dict = self.control_plate_dict(target_column=target_column)
            control_number = None # will be set in the loop
        else:
            control_number = self.num_cells_control()
        for compound in self.compounds:
            drug_table = self.table[(self.table['Drug'] == compound)]
            concentrations = np.unique(drug_table['Concentration'].to_numpy())
            if norm_to_plate:
                plate = drug_table['Plate'].values[0]
                control_number = c_dict[plate]
            ### prev
            #processed_concentrations, ref_concentrations = self.process_concentrations(concentrations)
            
            cellnumbers = []
            cellnumbers_std = []
            replicate_values = []
            use_concs = []
            ## prev
            #for i in range(len(processed_concentrations)):
            ## new
            for i, conc in enumerate(concentrations):
                ### prev
                #conc = processed_concentrations[conc_sort_indices[i]]
                #concs_here = ref_concentrations[conc_sort_indices[i]]
                #conc_sub_table = drug_table[drug_table['Concentration'].isin(concs_here)]
                ## new
                conc_sub_table = drug_table[drug_table['Concentration'] == conc]
                wells = list(set(conc_sub_table['Metadata_Well'].values))
                cellnumbers_per_conc = []
                for well in wells:
                    well_table = conc_sub_table[(conc_sub_table['Metadata_Well'] == well)]
                    num_targets_here = self.get_num_targets(well_table, target_column)
                    #print('Found {} viable cells for target {},  drug {} at conc {} in well {}'.format(num_targets_here, target_column, compound, conc, well))
                    num_cells_normed = num_targets_here / control_number
                    cellnumbers_per_conc.append(num_cells_normed)
                cellnumbers.append(np.mean(cellnumbers_per_conc))
                cellnumbers_std.append(np.std(cellnumbers_per_conc))
                replicate_values.append(cellnumbers_per_conc)
                use_concs.append(conc)
                ### prev
                #concentrations = use_concs
                #concentrations.sort()
                self.abs_dose_response_dict[compound] = concentrations, cellnumbers, cellnumbers_std, replicate_values
        return self.abs_dose_response_dict.copy()
        
    def calculate_control_fractions(self, target_column, target_plate=None, offtarget=False):
        """
        Calculate the fraction of cells with the name 'target_column' within the negative control wells

        Parameters
        ----------
        target_column : str
            Name of the column for which fractions should be calculated
        target_plate : str or int
            Plate for which fractions should be calculated. If None, all plates will be considered
        offtarget : bool
            Whether to calculate the fraction of off-target cells. Default is False

        Returns
        -------
        control_fractions : list
            List of control fractions for each well in the negative control
        
        """
        if target_plate is None:
            plates = self.plates
        else:
            plates = [target_plate]
        control_fractions = []
        for plate in plates:
            plate_table = self.table[self.table['Plate'] == plate]
            control_wells = list(set(plate_table[(plate_table['Drug'] == 'DMSO') | (plate_table['Drug'] == 'None')]['Metadata_Well'].values))
            for well in control_wells:
                control_fraction = self.calculate_fraction(target_column, plate_table, well, plate, offtarget=offtarget)
                control_fractions.append(control_fraction)
        return control_fractions
    
    
    def add_off_target_column(self, target_column):
        all_cells = self.table['NumberOfCells'].to_numpy()
        target_cells = self.table['NumberOfCells_{}'.format(target_column)].to_numpy()
        off_target_values = all_cells - target_cells
        self.table = self.table.assign(NumberOfCells_OffTarget=off_target_values)
        return

    def calculate_fraction(self, target_column, part_table, target_well, target_plate, offtarget=False):
        subtable = part_table[(part_table['Metadata_Well'] == target_well) & (part_table['Plate'] == target_plate)]
        if subtable.shape[0] == 0:
            num_cells  = 0
            return 0
        else:
            num_cells = subtable['NumberOfCells'].values[0]
            if num_cells == 0:
                return 0
            elif (target_column != 'Viable') and (target_column != self.viable_cellnumber_column):
                if not offtarget:
                    num_pos = subtable['NumberOfCells_{}'.format(target_column)].values[0]
                else:
                    num_pos = num_cells - subtable['NumberOfCells_{}'.format(target_column)].values[0]
            else:
                num_pos = subtable['NumberOfCells'].values[0]
            return float(num_pos) / num_cells

    def num_cells_control_plate(self, target_plate, target_column=None, offtarget=False):
        """
        Calculate the number of cells in the control wells for a given plate

        Parameters
        ----------
        target_plate : str or int
            Plate for which the number of cells in the control wells should be calculated
        target_column : str
            Name of the column for which the number of cells should be calculated. If None, the number of viable cells will be calculated. Default is None
        offtarget : bool
            Whether to calculate the offtarget score, ie substract the target population from the total number of cells. Default is False
        
        Returns
        -------
        cell_numbers : list
            List of cell numbers for each control well in the target plate
        
        """
        plate_table = self.table[self.table['Plate'] == target_plate]
        control_wells = list(set(plate_table[(plate_table['Drug'] == 'DMSO')]['Metadata_Well'].values))
        use_target_column = self.format_input_column(target_column, ref_table=plate_table)
        if use_target_column == self.viable_cellnumber_column:
            offtarget = False # no offtarget for viable cells
        if DEBUG:
            print('Calculating number of cells in control for target column {}'.format(target_column))
        viable_numbers = plate_table[plate_table['Metadata_Well'].isin(control_wells)][use_target_column].to_numpy()
        if offtarget:
            out_numbers = plate_table[plate_table['Metadata_Well'].isin(control_wells)][self.viable_cellnumber_column].to_numpy() - viable_numbers
        else:
            out_numbers = viable_numbers
        return list(out_numbers)

    def format_input_column(self, in_column, ref_table=None):
        if ref_table is None:
            ref_table = self.table
         ### check whether columns are correct
        if in_column in ref_table.columns:
            out_column = in_column
        elif (in_column is None) or (in_column == 'Viable') or (in_column == self.viable_cellnumber_column):
            out_column = self.viable_cellnumber_column
        elif (self.viable_cellnumber_column + '_' + in_column) in ref_table.columns:
            out_column = self.viable_cellnumber_column + '_' + in_column
        else:
            print('Column {} not found in table, aborting'.format(in_column))
            return
        return out_column


    def get_num_targets(self, well_table, target_column):
        if (target_column is None) or (target_column == 'Viable') or (target_column.strip() == 'NumberOfCells'):
            num_targets_here = well_table['NumberOfCells'].values[0]
        else:
            if well_table['Metadata_Well'].unique().shape[0] > 1:
                print('WARNING!! More than one well found for target table, aborting')
                num_targets_here = np.sum((well_table['NumberOfCells_{}'.format(target_column)].to_numpy()))
                print('Well, Drug and concentrations are: {}, {}, {}'.format(well_table['Metadata_Well'].values, well_table['Drug'].values, well_table['Concentration'].values))
            else:
                num_targets_here = well_table['NumberOfCells_{}'.format(target_column)].values[0]
        return num_targets_here

    def z_prime_raw(self, target_column, pos_control_drugs=['Cytarabine+Daunorubicin'], pos_control_concentrations=[5.,]):
        cellnum_dict = self.control_plate_dict(return_std=False, target_column=target_column)
        target_cellnumbers = []
        for drug in pos_control_drugs:
            drug_table = self.table[self.table['Drug'] == drug]
            if pos_control_concentrations is None:                
                pos_control_concentrations = list(set(drug_table['Concentration'].values))
            for conc in pos_control_concentrations:
                conc_table = drug_table[drug_table['Concentration'] == conc]
                wells = list(set(conc_table['Metadata_Well'].values))
                local_target_cell_numbers = [conc_table[conc_table['Metadata_Well']  == well]['NumberOfCells'].values[0] for well in wells]
                target_cellnumbers += local_target_cell_numbers
        all_cellnumbers_ctrl = []
        pos_control_table = self.table[(self.table['Drug'].isin(pos_control_drugs)) & (self.table['Concentration'].isin(pos_control_concentrations))]
        if DEBUG:
            print('Positive control table has shape {}'.format(pos_control_table.shape))
        pos_control_plates = list(set(pos_control_table['Plate'].values))
        for plate in pos_control_plates:
            all_cellnumbers_ctrl += self.num_cells_control_plate(plate, target_column=target_column)
        mean_ctrl = np.nanmean(all_cellnumbers_ctrl)
        std_ctrl = np.nanstd(all_cellnumbers_ctrl)
        mean_pos = np.nanmean(target_cellnumbers)
        std_pos = np.nanstd(target_cellnumbers)
        if DEBUG:
            print('Calculating Z-prime factor for the following values: Plates: {} positive ctrl-mean {}, positive ctrl-std: {}, ctrl-mean:{}, ctrl-std:{}'.format(pos_control_plates, round(mean_pos, 3), round(std_pos, 3), round(mean_ctrl, 3), round(std_ctrl, 3)))
        z_prime = 1. - ((3*(std_ctrl+std_pos)) / np.abs(mean_ctrl - mean_pos) )

        return z_prime



class InhibitionReport(LDTable_InhibitionReport):

    def __init__(self, whole_table, verbose=False, adjust_combination_concentrations=True, min_wells_per_drug_to_consider=3, reference_concentrations='auto'):
        """
        Class to generate inhibition reports from single-cell dataframes that have columns to denote viable cells, drugs, and concentrations

        Parameters
        ----------
        whole_table : pandas.DataFrame
                    DataFrame with single-cell data. Needs to have columns 'Drug', 'Concentration', 'Metadata_Well', 'Plate', 'Viable' 

        verbose : bool
                Verbosity level

        adjust_combination_concentrations : bool
                Adjust concentrations for combinations by checking if they are strings and whether they have been halved by accident. Default is True

        min_wells_per_drug_to_consider : int
                Minimum number of wells per drug to consider. Default is 3. Drugs with fewer wells will be excluded

        reference_concentrations : str or list
                Strategy to determine reference concentrations. Can be a list of concentrations to map to, 'auto' or 'all'.
                'auto' will use the most common concentrations across treatments.
                'all' will use all concentrations.
                Default is 'auto'
        
        """
        self.sc_table = whole_table
        whole_table = full_table_to_well_viability_table(whole_table, [])
        LDTable_InhibitionReport.__init__(self, whole_table, verbose=verbose, min_wells_per_drug_to_consider=min_wells_per_drug_to_consider, reference_concentrations=reference_concentrations)

class ReportWriter:


    def __init__(self, report):
        self.report = report
        pass

    
    def write_report(self, report_title, case_id=None):
        # Create the PdfPages object to which we will save the pages:
        # The with statement makes sure that the PdfPages object is closed properly at
        # the end of the block, even if an Exception occurs.
        figures = self.report.get_plots()
        if (report_title is None) or (len(report_title) == 0):
            report_title = 'output.pdf'
        if not (report_title[len(report_title) - 4:] == '.pdf'):
            report_title += '.pdf'
        with PdfPages(report_title) as pdf:
            for fig in figures:
                pdf.savefig(fig)
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = report_title.split('.')[0]
            d['Author'] = 'Ben Haladik'
            if not (case_id is None):
                d['Subject'] = case_id
            d['CreationDate'] = datetime.datetime.today()
        for fig in figures:
            plt.close(fig)
        return
        

    def write_report_as_image(self, report_titles, case_id=None):
        # Create the PdfPages object to which we will save the pages:
        # The with statement makes sure that the PdfPages object is closed properly at
        # the end of the block, even if an Exception occurs.
        figures = self.report.get_plots()
        if (report_titles is None):
            num_list = list(range(len(figures)))
            report_titles = [ str(f) + '_output.png' for f in num_list]
        for i in range(len(figures)):
            fig = figures[i]
            fig.savefig(report_titles[i])
        for fig in figures:
            plt.close(fig)
        return


def plot_compound_class_barplot(sorted_classes, sorted_masks, sorted_scores, savepath):
    fig, ax = plt.subplots(figsize=(2*len(sorted_classes), 7))
    barlist = ax.bar(sorted_classes, sorted_scores, color='grey')
    for i, mask in enumerate(sorted_masks):
        if mask > 0:
            barlist[i].set_color('lightblue')
    ax.set_xlabel('Z score')
    plt.setp(ax.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    fig.tight_layout()
    fig.savefig(savepath)
    return

def plot_compound_class_barplot_sns_style(in_df, savepath):
    classes = list(set(in_df['Class'].values))
    ### infer order
    mean_aucs_per_class = [np.mean(in_df[in_df['Class'] == f]['Z-score'].to_numpy()) for f in classes]
    sort_indices = np.argsort(mean_aucs_per_class)[::-1]
    sorted_classes = [classes[i] for i in sort_indices]
    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(2*n_classes, 13), nrows=2)
    sns.barplot(x='Class', y='Z-score', data=in_df, ax=ax[0], capsize=.3, edgecolor='black', order=sorted_classes, facecolor=(0, 0, 0, 0))
    sns.swarmplot(x='Class', y='Z-score', data=in_df, ax=ax[0], color='black', size=10, order=sorted_classes)
    sns.barplot(x='Class', y='AUC', data=in_df, ax=ax[1], capsize=.3, edgecolor='black', order=sorted_classes, facecolor=(0, 0, 0, 0))
    sns.swarmplot(x='Class', y='AUC', data=in_df, ax=ax[1], color='black', size=10, order=sorted_classes)
    plt.setp(ax[0].get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    plt.setp(ax[1].get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    fig.tight_layout()
    fig.savefig(savepath)
    return


def get_response_dicts_for_patient(patient_id, output_folder, annotation, get_replicate_values=False, target_population=None, via_table=None, do_abs=False):
    if target_population is None:
        target_population = annotation[annotation['InternalSampleNumber'] == patient_id]['Target'].values[0]
    if via_table is None:
        table_paths = [join(output_folder, f) for f in listdir(output_folder) if f.endswith('.csv')]
        via_tables = [f for f in table_paths if ('viabilities' in f)]
        if len(via_tables) > 0:
            via_table = pd.read_csv(via_tables[0])
            #print('Loaded table. Printing drugs')
            #print(list(set(via_table['Drug'].values)))
            via_table = via_table.dropna(subset=['Drug'])
            inh_report = LDTable_InhibitionReport(via_table)
        else:
            full_tables = [f for f in table_paths if ('full.csv' in f)]
            full_table = pd.read_csv(full_tables[0])
            inh_report = InhibitionReport(full_table)
    else:
        via_table = via_table.dropna(subset=['Drug'])
        inh_report = LDTable_InhibitionReport(via_table)
    if do_abs:
        auc_dict, rbf_dict = inh_report.calculate_abs_scores(target_population, norm_to_plate=True, sigmoid=True, return_sqerr=False, return_replicate_values=True)
    else:
        inh_report.calculate_pharmacoscopy_scores(target_population)
        #auc_dict, param_dict = inh_report.calculate_pharmacoscopy_aucs(target_population, return_all_data=True)
        # below is implementation that is in line with cohort level analysis. Above is old implementation that is commented out
        auc_dict = inh_report.sigmoid_aucs_rbf(target_population, return_scores=True)
    param_dict = inh_report.curve_params.copy()
    # end of newer implementation
    if get_replicate_values:
        if not do_abs:
            rbf_dict = inh_report.rbf_dict
        return auc_dict, param_dict, rbf_dict
    return auc_dict, param_dict


def make_multipatient_report(patient_list, patient_folder, annotation, highlight='first', use_top=10,\
                             show_highlight_dots=True, exclude_threshold=0.2, savename=None, highlight_via_table=None,\
                             highlight_population=None, rec_layout=True, rank_internal=True, do_abs=False):
    highlight_array = np.zeros(len(patient_list))
    if highlight == 'first':
        highlight_array[0] = 1
    elif isinstance(highlight, str) and (highlight in patient_list):
        highlight_array[patient_list.index(highlight)] = 1
    elif isinstance(highlight, list):
        for i, p in enumerate(patient_list):
            if p in highlight:
                highlight_array[i] = 1
    auc_dicts = []
    param_dicts = []
    rbf_dicts = []
    sorting_compounds = None
    joint_compounds = None
    p_highlight_index = 0
    for i, p in enumerate(patient_list):
        print(join(patient_folder, p))
        if highlight_array[i]:
            if not (highlight_via_table is None):
                auc_dict, param_dict, rbf_dict = get_response_dicts_for_patient(p, None, annotation, get_replicate_values=True,\
                                                                       target_population=highlight_population,\
                                                                       via_table=highlight_via_table, do_abs=do_abs)
            else:
                auc_dict, param_dict, rbf_dict = get_response_dicts_for_patient(p, join(patient_folder, p), annotation, get_replicate_values=True,\
                                                                           target_population=highlight_population, do_abs=do_abs)
            p_highlight_index += 1
        else:
            auc_dict, param_dict, rbf_dict = get_response_dicts_for_patient(p, join(patient_folder, p),\
                                                                       annotation, get_replicate_values=True, do_abs=do_abs)
        if joint_compounds is None:
            joint_compounds = list(auc_dict.keys())
        else:
            joint_compounds = [f for f in joint_compounds if f in auc_dict.keys()]
        if highlight_array[i]:
            ic50_dict = make_ic50_dict(joint_compounds, rbf_dict, curve_param_dict=param_dict)
            sorting_compounds, highlight_scores, highlight_ic50_strings = order_single_compound_list(auc_dict, rbf_dict=rbf_dict, ic50_dict=ic50_dict, return_ic50_strings=True, sort_by_ic50=True)
            print('Got the following ic50 strings')
            print(highlight_ic50_strings)
            if (not (exclude_threshold is None)) and (exclude_threshold > 0.):
                print('#######################')
                print('#######################')
                print('Applying error thresholds')
                #print([calculate_mean_error(rbf_dict, f) for f in sorting_compounds[:10]])
                #sorting_compounds = [f for f in sorting_compounds if (calculate_mean_error(rbf_dict, f) < exclude_threshold)]
                sorting_compounds, sorting_indices = filter_by_fit_error(sorting_compounds, rbf_dict, exclude_threshold, return_indices=True)
                highlight_scores = [highlight_scores[i] for i in sorting_indices]
                highlight_ic50_strings = [highlight_ic50_strings[i] for i in sorting_indices]
            print('found highlight array. First few sorted compounds are {}'.format(sorting_compounds[:5]))
        auc_dicts.append(auc_dict)
        param_dicts.append(param_dict)
        rbf_dicts.append(rbf_dict)
    if sorting_compounds is None:
        sorting_compounds = joint_compounds
    #else:
    #    sorting_compounds = [f for f in sorting_compounds if (f in joint_compounds)]
    if isinstance(use_top, int) and use_top > 0:
        sorting_compounds = sorting_compounds[:use_top]
        print('Only showing the following top {} compounds'.format(use_top))
        print(sorting_compounds)
        input_ranks = None
    elif isinstance(use_top, list) and isinstance(use_top[0], str):
        sorting_indices = [sorting_compounds.index(f) for f in use_top]
        if rank_internal:
            input_ranks = [i+1 for i in sorting_indices]
            print('## Ranking internally with the following indices')
            print(input_ranks)
        else:
            input_ranks = None
        #sorting_compounds = [f for f in sorting_compounds if (f in use_top)]
        sorting_compounds = [sorting_compounds[i] for i in sorting_indices]
        highlight_scores = [highlight_scores[i] for i in sorting_indices]
        print('using the following compounds')
        print(sorting_compounds)
        
    _, _, _ = plot_curves(auc_dicts, rbf_dicts, sorting_compounds, highlight_array, show_highlight_dots=show_highlight_dots,\
                savename=savename, highlight_scores=highlight_scores, rec_layout=rec_layout, input_ranks=input_ranks, ic50_string_list=highlight_ic50_strings)
    return auc_dicts