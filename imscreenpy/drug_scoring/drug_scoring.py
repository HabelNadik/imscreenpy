import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import sklearn

def drug_sigmoid(x, a, b, c):
    x = np.log10(x + 1)
    x += np.amin(x)
    x /= np.amax(x)
    if c > 0:
        out = a + ((1. - a) / (1 + (x/b)**c ) )    
    else:
        out = a + ((1. - a) / (1 + (1./(x/b))**abs(c) ) )
    return out


def fit_sigmoid(x, y, verbose=False, sigma=None, absolute_sigma=False):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    unique_x = np.unique(x)
    initial_guess = [np.nanmin(y), np.mean(unique_x), 5.]
    try:
        opt, pcov = curve_fit(drug_sigmoid, x, y, method='dogbox', bounds=([-1, np.mean([unique_x[0], unique_x[1]]), 1.],[1, 10., 10.]), p0=initial_guess, sigma=sigma, absolute_sigma=absolute_sigma)
        #opt, pcov = curve_fit(drug_sigmoid, x, y, method='dogbox', bounds=([np.finfo(np.float64).eps, np.mean([unique_x[0], unique_x[1]]), 2.],[1, 10., 10.]), p0=initial_guess)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, Exception):
        if verbose:
            print('caught fit error, creating pseudo fit')
        if np.mean(y) < 1.:
            opt = [np.mean(y[(x != (unique_x[0]))]), np.mean([unique_x[0], unique_x[1]]), 5.]
            pcov = np.zeros((3,3))
        else:
            opt = [1., 0.01, 1.]
            pcov = np.zeros((3,3))
    return opt, pcov


def concentrations_to_log_scale(input_concentrations, ref_min_conc=None):
    """
    Convert concentrations to log scale, to match the log scale conversion in our dose response curve
    
    input paramters:
    input_concentrations (list or np.ndarray) - concentrations that we want to convert
    """
    ### convert to array if we have a list
    if isinstance(input_concentrations, list):
        conc_arr = np.array(input_concentrations).flatten()
    else:
        conc_arr = input_concentrations.flatten()
    ### declare the output array
    out_arr = np.zeros(conc_arr.shape)
    ### log10 transform all values higher than 0
    out_arr[conc_arr > 0] = np.log10(conc_arr[conc_arr > 0])
    ### log10(0) is undefined, so we set that value to rounded minimum - 1, to get a linear spacing between our values
    if ref_min_conc is None:
        out_arr[conc_arr == 0] = np.amin(out_arr[conc_arr > 0]) - 1
    else:
        out_arr[conc_arr == 0] = np.log10(ref_min_conc) - 1
    return out_arr


def make_curve_values(curve_params, min_val, max_val, ref_min_conc=None):
    """
    Make the values for plotting for our dose response curves
    
    input parameters:
    curve_params (list) - parameters of the dose response curve
    min_val (float) - minimum value at which the curve starts, usually zero
    max_val (float) - maximum value where the curve ends. Should be our maximum concentration
    """
    x = np.linspace(min_val, max_val, num=5000)
    ### if we have a reference min concentration
    if not (ref_min_conc is None):
        ## and we pseudoconcentrations that are more than 10x smaller than our reference
        min_mask = (x <= (ref_min_conc/10.)) & (x > 0)
        num_problematic_concs = np.sum(min_mask)
        if num_problematic_concs > 0:
            ## we need to adjust our pseudoconcentrations such that our plot still works
            x = x[(x > (ref_min_conc/10.)) | (x == 0)]
    y = drug_sigmoid(x, curve_params[0], curve_params[1], curve_params[2])
    return x,y

def make_min_conc_ref_dict(compounds, rbf_dicts):
    out_dict = dict()
    for cpd in compounds:
        for rbf_dict in rbf_dicts:
            if cpd in rbf_dict.keys():
                concs, _, _, _ = rbf_dict[cpd]
                min_val = np.amin([f for f in concs if (f > 0)])
                if cpd in out_dict.keys():
                    if min_val < out_dict[cpd]:
                        out_dict[cpd] = min_val
                else:
                    out_dict[cpd] = min_val
        print('Got minimum of {} for drug {}'.format(min_val, cpd))
    return out_dict

def order_single_compound_list(auc_dict, ic50_dict=None, rbf_dict=None, return_ic50_strings=False, sort_by_ic50=False):
    """
    Order compound list for a single sample either by AUC or jointly by IC50 first and then AUC
    
    """

    ### sort by AUC first
    drugs = list(auc_dict.keys())
    scores = [auc_dict[f] for f in drugs]
    auc_indices = np.argsort(scores)[::-1]
    auc_sorted_drugs = []
    auc_sorted_scores = []
    for auc_index in auc_indices:
        if not np.isnan(scores[auc_index]):
            auc_sorted_drugs.append(drugs[auc_index])
            auc_sorted_scores.append(scores[auc_index])
    ### return immediately if we don't have the infomration we need for IC50 values
    if (ic50_dict is None) or (rbf_dict is None):
        return auc_sorted_drugs, auc_sorted_scores
    else:
        out_ic50_strings = []
        ### pre-sort ic50 values by auc
        ic50_vals = [ic50_dict[f] for f in auc_sorted_drugs]
        ### get the max concentrations, so that we can make the strings
        max_concentrations = [np.nanmax(rbf_dict[f][0]) for f in auc_sorted_drugs]
        ### then make the strings
        for i, val in enumerate(ic50_vals):
            if np.isnan(val):
                out_ic50_strings.append('>{} uM'.format(max_concentrations[i]))
            else:
                out_ic50_strings.append('{} uM'.format(round(val, 3)))
        if not sort_by_ic50:
            ### return if we don't want to sort by IC50
            if return_ic50_strings:
                return auc_sorted_drugs, auc_sorted_scores, out_ic50_strings
            else:
                return auc_sorted_drugs, auc_sorted_scores, ic50_vals
        else:
            ### sort ic50 vals
            ic50_sort_indices = np.argsort(ic50_vals)
            out_ic50_values_sorted = []
            out_ic50_strings_sorted = []
            ic50_auc_sorted_drugs = []
            ic50_auc_sorted_scores = []
            ### iterate over sort indices
            for sorting_index in ic50_sort_indices:
                ic50 = ic50_vals[sorting_index]
                if not np.isnan(ic50):
                    ### append all relevant information if we have the IC50
                    out_ic50_values_sorted.append(ic50)
                    out_ic50_strings_sorted.append(out_ic50_strings[sorting_index])
                    ic50_auc_sorted_drugs.append(auc_sorted_drugs[sorting_index])
                    ic50_auc_sorted_scores.append(auc_sorted_scores[sorting_index])
            ### add the drugs in order of AUC
            for i, drug in enumerate(auc_sorted_drugs):
                ### but only for the ones where don't have an IC50, ie which we haven't sorted above yet
                if not (drug in ic50_auc_sorted_drugs):
                    ic50_auc_sorted_drugs.append(drug)
                    ic50_auc_sorted_scores.append(auc_sorted_scores[i])
                    out_ic50_values_sorted.append(ic50_vals[i])
                    out_ic50_strings_sorted.append(out_ic50_strings[i])
            if return_ic50_strings:
                return ic50_auc_sorted_drugs, ic50_auc_sorted_scores, out_ic50_strings_sorted
            else:
                return ic50_auc_sorted_drugs, ic50_auc_sorted_scores, out_ic50_values_sorted


def filter_by_fit_error(compound_list, rbf_dict, error_threshold, return_indices=False):
    keep_compounds = []
    keep_indices = []
    #print('###########################################')
    #print('######## Performing filtering #############')
    for i, compound in enumerate(compound_list):
        concs, rbfs, rbfs_std, points = rbf_dict[compound]
        rbfs_expanded = [1., 1., 1.]
        concs_expanded = [0., 0., 0.]
        for n, pt in enumerate(points):
            concs_expanded += [concs[n]] * len(pt)
            rbfs_expanded += pt   
        params, _ = fit_sigmoid(concs_expanded, rbfs_expanded)
        sigmoid_vals = drug_sigmoid(np.array(concs_expanded[3:]), params[0], params[1], params[2])
        errors = np.abs(np.array(rbfs_expanded[3:] - sigmoid_vals))
        err = np.nanmean(errors)
        if err < error_threshold:
            keep_compounds.append(compound)
            keep_indices.append(i)
    if return_indices:
        return keep_compounds, keep_indices
    return keep_compounds

def calculate_replicate_differences(drug, rbf_dict):
    concentrations, rbfs_per_drug, rbfs_std_per_drug, all_rbf_single_vals = rbf_dict[drug]
    errs = []
    for i, conc in enumerate(concentrations):
        rbf_vals = all_rbf_single_vals[i]
        mean_val = np.mean(rbf_vals)
        errs.append(np.mean([(f - mean_val)**2 for f in rbf_vals]))
    return np.mean(errs)



def score_compound_classes(target_auc_dict, target_rbf_dict, annotation, comparison_auc_dict_list=None, seaborn_style=False):
    compounds = list(target_auc_dict.keys())
    #compound_classes = [annotation[annotation['Synonym in transferlist'] == f]['Report compound class'].values[0] for f in compounds]
    compound_classes = []
    for cpd in compounds:
        if cpd in annotation['Synonym in transferlist'].values:
            compound_classes.append(annotation[annotation['Synonym in transferlist'] == cpd]['Report compound class'].values[0])
        elif cpd in annotation['CompoundName'].values:
            compound_classes.append(annotation[annotation['CompoundName'] == cpd]['Report compound class'].values[0])
        else:
            print('WARNING! No compound class found for compound {}'.format(cpd))
    unique_compound_classes = list(set(compound_classes))
    compound_class_scores = []
    compound_class_masks = []
    ### lists for generating dataframe for seaborn style plotting
    compound_class_aucs_expanded = []
    compound_class_scores_expanded = []
    compound_class_names_expanded = []
    compound_names_expanded = []
    sig_threshold = 0.05 / len(unique_compound_classes)


    for c_class in unique_compound_classes:
        member_compounds = [compounds[i] for i in range(len(compound_classes)) if (compound_classes[i] == c_class)]
        non_member_compounds = [compounds[i] for i in range(len(compound_classes)) if (compound_classes[i] != c_class)]
        auc_values = [target_auc_dict[f] for f in member_compounds]
        non_member_auc_values =  [target_auc_dict[f] for f in non_member_compounds]
        ### gather data for seaborn style plotting
        compound_class_aucs_expanded += auc_values
        compound_class_names_expanded += [c_class] * len(auc_values)

        compound_names_expanded += member_compounds
        if comparison_auc_dict_list is None:
            error_values = [calculate_replicate_differences(f, target_rbf_dict) for f in member_compounds]
            inhibition_higher = []
            inhibition_z_scores_sns = []
            for auc_val, err_val in zip(auc_values, error_values):
                if auc_val > err_val:
                    inhibition_higher.append(1)
                else:
                    inhibition_higher.append(0)
                ### gather data for seaborn style plotting
                z_score_sns = (auc_val - np.nanmean(non_member_auc_values)) / np.nanstd(non_member_auc_values)
                inhibition_z_scores_sns.append(z_score_sns)
            #compound_class_scores.append(np.mean(auc_values))
            if np.sum(inhibition_higher) > (len(inhibition_higher) / 2):
                compound_class_masks.append(1.)
            else:
                compound_class_masks.append(0.)
            if len(auc_values) > 1:
                member_std = np.nanstd(auc_values)
                member_val = np.nanmean(auc_values)
            else:
                member_std = np.nanstd(non_member_auc_values)
                member_val = auc_values[0]
            z_std = np.mean([member_std, np.nanstd(non_member_auc_values)])
            non_member_mean = np.mean(non_member_auc_values)
            compound_class_scores.append((member_val - non_member_mean) / z_std)
            compound_class_scores_expanded += inhibition_z_scores_sns


        else:
            masks_per_compound = []
            scores_per_compound = []

            for m_c in member_compounds:
                target_score = target_auc_dict[m_c]
                scores_per_compound_and_sample = []
                for auc_dict in comparison_auc_dict_list:
                    if m_c in auc_dict.keys():
                        scores_per_compound_and_sample.append(auc_dict[m_c])
                ### calculate the z score
                score = (target_score - np.nanmean(scores_per_compound_and_sample)) / np.nanstd(scores_per_compound_and_sample) 
                scores_per_compound.append(score)
                ### do one sample t-test to see if our value is significantly different from all other samples
                test_stat, p = stats.ttest_1samp(scores_per_compound_and_sample, popmean=target_score)
                ### test statistic will be above zero if mean of scores_per_compound is greater than target_score, and vice versa
                ### so we want to have a significant negative value
                if (test_stat < 0) and (p < sig_threshold):
                    masks_per_compound.append(1.)
                else:
                    masks_per_compound.append(0.)

            if len(scores_per_compound) > 1:
                compound_class_scores.append(np.mean(scores_per_compound))
            else:
                compound_class_scores.append(scores_per_compound[0])
            if np.sum(masks_per_compound) >= (len(masks_per_compound) / 2):
                compound_class_masks.append(1)
            else:
                compound_class_masks.append(0.)
            
            compound_class_scores_expanded += scores_per_compound
    if not seaborn_style:
        sort_indices = np.argsort(compound_class_scores)[::-1]
        sorted_classes = [unique_compound_classes[i] for i in sort_indices]
        sorted_masks = [compound_class_masks[i] for i in sort_indices]
        sorted_scores = [compound_class_scores[i] for i in sort_indices]
        return sorted_classes, sorted_masks, sorted_scores
    else:
        df_dict = dict()
        df_dict['Z-score'] = compound_class_scores_expanded
        df_dict['AUC'] = compound_class_aucs_expanded
        df_dict['Class'] = compound_class_names_expanded
        df_dict['Compound'] = compound_names_expanded
        df = pd.DataFrame(df_dict)
        return df


def make_ic50_dict(compounds, rbf_dict, curve_param_dict=None):
    """
    Make a dictionary that contains IC50 values for each compoound if available
    """
    out_dict = dict()
    for cpd in compounds:
        concs, _, _, rbfs_per_conc = rbf_dict[cpd]
        if curve_param_dict is None:
            rbfs_expanded = [1., 1., 1.]
            concs_expanded = [0., 0., 0.]
            for n, pt in enumerate(rbfs_per_conc):
                concs_expanded += [concs[n]] * len(pt)
                rbfs_expanded += pt
            curve_params, _ = fit_sigmoid(concs_expanded, rbfs_expanded)
        else:
            curve_params = curve_param_dict[cpd]
        ic50_val = approximate_ic50(rbf_dict, cpd, curve_params=curve_params)
        out_dict[cpd] = ic50_val
    return out_dict

def approximate_ic50(rbf_dict, compound, curve_params=None):
    concs, rbfs, rbfs_std, points = rbf_dict[compound]
    if (curve_params is None):
        rbfs_expanded = [1., 1., 1.]
        concs_expanded = [0., 0., 0.]
        for n, pt in enumerate(points):
            concs_expanded += [concs[n]] * len(pt)
            rbfs_expanded += pt
        curve_params, _ = fit_sigmoid(concs_expanded, rbfs_expanded)
    fake_concs = np.linspace(np.finfo(np.float16).eps, np.amax(concs), num=50000)
    sigmoid_vals = drug_sigmoid(fake_concs, curve_params[0], curve_params[1], curve_params[2])
    #print('Calculating IC50 for {} with min conc {} max conc {}'.format(compound, np.amin(concs), np.amax(concs)))
    #print('Response values: Min {} Max {} Mean {}'.format(np.nanmin(sigmoid_vals), np.nanmax(sigmoid_vals), np.nanmean(sigmoid_vals)))
    #print('Min value is for fake concentration {} Max is for fake concentration {}'.format(fake_concs[np.argmin(sigmoid_vals)], fake_concs[np.argmax(sigmoid_vals)]))
    if (np.nanmin(sigmoid_vals) < -0.5) or (np.nanmax(sigmoid_vals) > 3):
        point_arr = np.array([np.mean(f) for f in points])
        conc_arr = np.array(concs)
        ic50_mask = point_arr <= 0.5
        if np.sum(ic50_mask) > 0:
            if np.sum(ic50_mask) == np.max(sigmoid_vals.shape):
                out_val = np.nanmin(concs)
            elif np.sum(ic50_mask) > 0:
                values_below_50 = conc_arr[ic50_mask]
                values_above_50 = conc_arr[~ic50_mask]
                out_val = np.nanmean([np.nanmin(values_below_50), np.nanmax(values_above_50)])
            else:
                out_val = np.nan
    if (.5 in sigmoid_vals):
        out_val = fake_concs[list(sigmoid_vals).index(.5)]
    else:
        ic50_mask = (sigmoid_vals <= 0.5)
        if np.sum(ic50_mask) == np.max(sigmoid_vals.shape):
            out_val = np.nanmin(concs)
        elif np.sum(ic50_mask) > 0:
            values_below_50 = fake_concs[ic50_mask]
            values_above_50 = fake_concs[~ic50_mask]
            out_val = np.nanmean([np.nanmin(values_below_50), np.nanmax(values_above_50)])
        else:
            out_val = np.nan
    return out_val

def round_conc(input_concentration):  
    str_conc = format(input_concentration, 'f')
    before_zero = str_conc[:str_conc.index('.')]
    after_zero = str_conc[str_conc.index('.')+1:]
    num_digits = len(after_zero)
    num_digits_to_use = None
    for i in range(num_digits):
        if after_zero[i] != '0':# detect first non zero position
            if before_zero == '0':# round from there if the input value is <0
                if (after_zero[i] == '9'):
                    num_digits_to_use = i
                else:
                    num_digits_to_use = i + 1
                break
            elif i > 2:# ignore values after 2 decimal places if input value >0
                num_digits_to_use = 2
                break
            else:# round as usual otherwise
                num_digits_to_use = i+1
                break
    return round(input_concentration, num_digits_to_use)

def process_conc(conc):
    if isinstance(conc, float) or isinstance(conc, int):
        newconc = round_conc(conc)
    elif isinstance(conc, str):
        if ('+' in conc):
            left, right = conc.split('+')
            newconc = round_conc(float(left) + float(right))
        else:
            newconc = round_conc(float(conc))
    return newconc


def reformat_concentrations(in_df, adjust_for_combos=False):
    in_concs = in_df['Concentration'].to_numpy()
    in_drugs = in_df['Drug'].to_numpy()
    if adjust_for_combos:
        in_drugs = in_df['Drug'].to_numpy()
        adj_concs = []
        for i, conc in enumerate(in_concs):
            if isinstance(conc, float):
                adj_concs.append(conc)
            elif isinstance(in_drugs[i], str):
                ### concentrations for combinations are erroneously set to half for the new transferlist
                if (('+' in in_drugs[i]) and (not ('(+)' in in_drugs[i]))) and (not (conc in [0.01, 0.001, 0.1, 1., 10.])) and (not isinstance(conc, str)):
                    proc_conc = 2 * conc
                else:
                    proc_conc = conc
                adj_concs.append(proc_conc)
            else:
                adj_concs.append(0.)
        in_concs = adj_concs
    out_concs = [process_conc(f) for f in in_concs]
    return in_df.assign(Concentration=out_concs)


def simplify_drug_name(name):
    if name.strip() == 'IPI-145':
        drug_name = 'Duvelisib (IPI-145)'
    elif name.strip() == 'I-BET762':
        drug_name = 'Molibresib (I-BET762)'
    elif name.strip() == 'MLN4924':
        drug_name = 'Pevonedistat (MLN4924)'
    elif name.strip() == 'CB-839':
        drug_name = 'Telaglenastat (CB-839)'
    elif name.strip() == 'EPZ-5676':
        drug_name = 'Pinemetostat (EPZ-5676)'
    elif name.strip() == 'EPZ-6438':
        drug_name = 'Tazemetostat (EPZ-6438)'
    elif name.strip().startswith('CPD') and ('rubicin' in name):
        name_split = name.split(';')
        drug_name = name_split[1]
        drug_name = drug_name[0].upper() + drug_name[1:]
    elif (' (' in name):
        drug_name = name.split(' (')[0]
    else:
        drug_name = name
    return drug_name

def simplify_drug_names(input_table):
    drug_names = list(input_table['Drug'].values)
    simplified_names = []
    for name in drug_names:
        drug_name = simplify_drug_name(name)
        simplified_names.append(drug_name)
    return input_table.assign(Drug=simplified_names)


def calc_auc(dr_values, ref_concentrations=None):
    """
    calculate the approximate AUC for a list of dr values.
    We assume that the there is only one value per concentration, which is the mean
    """
    if len(dr_values) == 0:
        return 0
    else:
        ## we only have one value per concentration, which is the mean
        dr_values = np.array(dr_values).flatten()
        inhibitions = np.zeros(dr_values.shape[0] + 1)
        inhibitions[1:] = 1. - np.array(dr_values).flatten()
        if not (ref_concentrations is None):
            use_concentrations = np.zeros(dr_values.shape[0] + 1)
            use_concentrations[1:] = ref_concentrations
        auc = 0.
        num_steps = 0.
        prev_value = 0
        prev_conc = 0
        ## and we use trapezoids to compute the auc
        for i, current in enumerate(inhibitions):
            if (ref_concentrations is None):
                # without concentration information, we assume that the width between each concentration is 1. So we take prev as the area of the rectancle  
                width = 1
            else:
                # with concentration information, we use the concentration difference as the width of the rectangle
                width = (use_concentrations[i] - prev_conc)
                prev_conc = use_concentrations[i]
            ## we approximate the auc with the trapezoid rule, ie the area of the rectangle plus the area of the triangle that sits on top
            auc += (prev_value * width) + ((current - prev_value) * width / 2)
            prev_value = current
            num_steps += 1
        ## then we just normalize by the number of steps
        if ref_concentrations is None:
            return auc / num_steps
        else:
            return auc
    

def make_sampling_array(sorted_nested_replicate_values):
    test_arr = np.zeros((len(sorted_nested_replicate_values), max([len(f) for f in sorted_nested_replicate_values])+1))
    test_arr[:,:] = np.nan
    for i in range(len(sorted_nested_replicate_values)):
        if len(sorted_nested_replicate_values[i]) > 1:
            test_arr[i][:len(sorted_nested_replicate_values[i])] = sorted_nested_replicate_values[i]
        else:
            test_arr[i, 0] = sorted_nested_replicate_values[i][0]
    return test_arr
    


def calcuate_auc_std_err(sorted_concentrations, sorted_nested_replicate_values, n_iterations=1000):
    aucs = []
    arr_for_sampling = make_sampling_array(sorted_nested_replicate_values)
    #print(arr_for_sampling)
    for _ in range(n_iterations):
        use_arr = np.zeros(arr_for_sampling.shape)
        for i in range(arr_for_sampling.shape[0]):
            use_arr[i] = sklearn.utils.resample(arr_for_sampling[i], replace=True)
        mean_dose_response_values = np.nanmean(use_arr, axis=1)
        use_sorted_concentrations = sorted_concentrations
        if np.sum(~np.isnan(mean_dose_response_values)) >= 2:
            #print(mean_dose_response_values)
            if np.sum(np.isnan(mean_dose_response_values)) > 0:
                keep_mask = ~np.isnan(mean_dose_response_values)
                mean_dose_response_values = mean_dose_response_values[keep_mask]
                use_sorted_concentrations = [sorted_concentrations[j] for j in range(len(sorted_concentrations)) if keep_mask[j]]
            #print(mean_dose_response_values)
            #print(sorted_concentrations)
            interpolated_dose_response_values = interpolate_for_ref_concentrations(mean_dose_response_values, use_sorted_concentrations, sorted_ref_concentrations=[0.01, 0.1, 1.], initial_value=1., do_clip=True)
            #print('Working with interpolated values {}'.format(interpolated_dose_response_values))
            score = calc_auc(interpolated_dose_response_values, ref_concentrations=[0.01, 0.1, 1.])
            aucs.append(score)
    return np.std(aucs)

def interpolate_for_ref_concentrations(dr_values, sorted_concentrations, sorted_ref_concentrations=[0.01, 0.1, 1.], initial_value=1., verbose=False, do_clip=True):
    use_sorted_ref_concentrations = np.zeros(len(sorted_ref_concentrations) + 1)
    use_sorted_ref_concentrations[1:] = sorted_ref_concentrations
    use_sorted_concentrations = np.zeros(len(sorted_concentrations) + 1)
    use_sorted_concentrations[1:] = sorted_concentrations
    use_dr_values = np.ones(len(dr_values) + 1) * initial_value
    use_dr_values[1:] = dr_values
    out_values = np.interp(use_sorted_ref_concentrations, use_sorted_concentrations, use_dr_values, left=dr_values[0], right=dr_values[-1])
    out_values = out_values[1:]
    if verbose:
        print('Interpolated values: {} from responses {} and concentrations {}'.format(dr_values, use_dr_values, sorted_concentrations))
    if do_clip:
        return list(np.clip(out_values, 0., 1.))
    else:
        return out_values

