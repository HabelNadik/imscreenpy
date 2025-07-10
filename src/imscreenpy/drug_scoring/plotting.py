import numpy as np

import matplotlib.pyplot as plt

from .drug_scoring import simplify_drug_name, make_min_conc_ref_dict, concentrations_to_log_scale, fit_sigmoid, make_curve_values

def make_ticks_and_labels(input_concentrations):
    """
    Make ticks and tick labels for input concentrations which are log scaled.
    Ticks are the positions where we want to mark things on the x axis, ie the concentrations that we use
    Tick labels are the strings that we write at those positions
    
    input parameters:
    input_concentrations (list or np.ndarray) - concentrations to make ticks for - log-scaled
    """
    ticks = list(np.unique(input_concentrations))
    ticks.sort()
    min_tick_value = np.amin(ticks) ### minimum tick value is our representation for zero
    tick_format_string = '{} uM'
    tick_labels = [tick_format_string.format(0)]
    for tick in ticks:
        if tick != min_tick_value: ## we skip the smallest tick because that's just our representation of zero, as we did it above in the function concentrations_to_log_scale
            tick_num = np.power(10, tick)
            ### sometimes the log transform does not correctly recover whole numbers due to decimals being rounded internally, so in those cases we round
            if (tick_num > 1) and ((round(tick_num) - tick_num) < 0.05):
                tick_num = round(tick_num)
            ### this can also happen with small numbers. Here, the problem seems to be multiples of 5 in particular, where we add a lot of 9s
            if tick_num < 1:
                str_conc = format(tick_num, 'f')
                pos_minus = str_conc.index('.') + 1
                if str_conc.count('9') >= 2:
                    tick_num = round(tick_num, str_conc.index('9')-pos_minus)
                else:
                    tick_num = float(str_conc)
            tick_labels.append(tick_format_string.format(tick_num))
    return ticks, tick_labels

def plot_response_curve(viability_df, target, target_drug, conc_col='Concentration', drug_col='Drug', nctrl='DMSO', in_ax=None, colors=['#ff6961', '#3FB06A', '#403f3f']):
    """
    Plot dose response curve for a given column in a viability dataframe

    Parameters:
    -----------
    viability_df : pd.DataFrame
        DataFrame containing viability data
    target_column : str or list of str
        List of columns or column in the DataFrame to plot
    target_drug : str
        Drug name to plot, if None, we assume that the dataframe contains a single drug, 
        will throw an error if there is more than one drug and target_drug is set to None
    conc_col : str
        Name of the column that stores concentrations. Default Concentration
    drug_col : str
        Name of the column that stores drug anmes
    nctrl : str
        Name of the control condition, default DMSO. If None, no normalization to control will be performed
    in_ax : matplotlib.pyplot.Axes
        Axes to plot on, if None, a new figure is created and returned

    Returns:
    --------
    matplotlib.pyplot.Axes
        Axes object containing the plot, or matplotlib.pyplot.Figure if in_ax is None
    """
    if isinstance(target, str):
        target = [target]
    if not isinstance(colors, list):
        colors = [colors] * len(target)
    all_x_scatter = []
    all_y_scatter = []
    all_x_curve = []
    all_y_curve = []
    
    for target_column in target:
        if not (nctrl is None):
            ## we assume that our target values are not normalized, so we we will do that here
            nctrl_mean = np.nanmean(viability_df[viability_df['Drug'] == nctrl][target_column].to_numpy())
            normed_vals = viability_df[target_column].to_numpy() / nctrl_mean
            viability_df = viability_df.assign(**{target_column: normed_vals})
            drug_df = viability_df[viability_df[drug_col] == target_drug].sort_values(by='Concentration')
            concentrations = [0., 0., 0.] + drug_df[conc_col].to_list()
            target_values = [1., 1., 1.] + drug_df[target_column].to_list()
        else:
            drug_df = viability_df[viability_df[drug_col] == target_drug].sort_values(by='Concentration')
            concentrations = drug_df[conc_col].to_list()
            target_values = drug_df[target_column].to_list()
        min_ref_conc = np.nanmin(drug_df['Concentration'].values)
        params, _ = fit_sigmoid(concentrations, target_values)

        log_concentrations_scatter = concentrations_to_log_scale(concentrations)

        x,y = make_curve_values(params, min(concentrations), max(concentrations), ref_min_conc=min_ref_conc)
        log_x_curve = concentrations_to_log_scale(x, ref_min_conc=min_ref_conc)

        all_x_scatter.append(log_concentrations_scatter)
        all_y_scatter.append(target_values)
        all_x_curve.append(log_x_curve)
        all_y_curve.append(y)

    ticks, tick_labels = make_ticks_and_labels(log_concentrations_scatter)

    if in_ax is None:
        fig, ax = plt.subplots()
        return_obj = fig
    else:
        ax = in_ax
    z_order_index = 1
    for i, target_column in enumerate(target):
        ax.plot(all_x_curve[i], all_y_curve[i], color=colors[i], linewidth=1.5, zorder=z_order_index)
        z_order_index += 1
    for i, target_column in enumerate(target):    
        ax.scatter(all_x_scatter[i], all_y_scatter[i], color=colors[i], edgecolors='black', label=target_column, zorder=z_order_index)
        z_order_index += 1
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    
    ax.set_ylim(-0.1, 1.35)
    if nctrl is None:
        ax.set_ylabel('Viability', fontsize=9)
    else:
        ax.set_ylabel('Relative viability', fontsize=9)
    ax.set_xlabel('Concentration', fontsize=9)
    ax.set_title(target_drug, fontsize=11)
    ax.legend()
    if in_ax is None:
        return fig
    else:
        return ax

def plot_curves_for_drug(target_populations, target_drug, viability_df, conc_col='Concentration', drug_col='Drug', nctrl='DMSO', in_ax=None, colors=None):
    """
    Plot dose response curves for a given drug and target populations

    Parameters:
    -----------
    target_populations : list of str
        List of target populations to plot
    target_drug : str
        Drug name to plot
    viability_df : pd.DataFrame
        DataFrame containing viability data
    conc_col : str
        Name of the column that stores concentrations. Default Concentration
    drug_col : str
        Name of the column that stores drug names. Default Drug
    nctrl : str
        Name of the control condition, default DMSO. If None, no normalization to control will be performed
    in_ax : matplotlib.pyplot.Axes
        Axes to plot on, if None, a new figure is created and returned
    colors : list of str or None
        List of colors for each population, if None, default colors are used

    Returns:
    --------
    matplotlib.pyplot.Axes or matplotlib.pyplot.Figure
        Axes object containing the plot, or matplotlib.pyplot.Figure if in_ax is None
    """
    
    if in_ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        ax = in_ax

    if colors is None:
        colors = ['#ff6961', '#77dd77', '#fdfd96', '#84b6f4', '#ffb347'][:len(target_populations)]

    for i, pop in enumerate(target_populations):
        ax = plot_response_curve(viability_df, pop, target_drug=target_drug, conc_col=conc_col, drug_col=drug_col, nctrl=nctrl, in_ax=ax, color=colors[i])

    return ax

def plot_curves(auc_dicts, rbf_dicts, sorted_compounds, highlight_array, savename=None, show_highlight_dots=True, highlight_scores=None, rec_layout=True, input_ranks=None, ic50_string_list=None):
    n_figures = len(sorted_compounds)
    n_curves = len(auc_dicts)
    x_values_for_plotting = []
    y_values_for_plotting = []
    x_ticks_per_compound = []
    x_ticklabels_per_compound = []
    highlighted_rbfs = []
    highlighted_rbf_indices = []
    highlight_index = 0
    min_conc_ref_dict = make_min_conc_ref_dict(sorted_compounds, rbf_dicts)
    for compound in sorted_compounds:
        x_values_per_compound = []
        y_values_per_compound = []
        previous_concs = None
        previous_x = None
        #print('compound is {}'.format(compound))
        min_x_val = None
        for i, rbf_dict in enumerate(rbf_dicts):
            if compound in rbf_dict.keys():
                concs, rbfs, rbfs_std, points = rbf_dict[compound]

                rbfs_expanded = [1., 1., 1.]
                concs_expanded = [0., 0., 0.]
                for n, pt in enumerate(points):
                    concs_expanded += [concs[n]] * len(pt)
                    rbfs_expanded += pt
                if len(concs_expanded) <= 3:
                    x_values_per_compound.append([0])
                    y_values_per_compound.append([1])
                else:
                    params, _ = fit_sigmoid(concs_expanded, rbfs_expanded)
                    #### processing of concentrations to create log scale plot for scatter values
                    log_concentrations_scatter = concentrations_to_log_scale(concs_expanded)
                    #### processing of concentrations to create log scale plot for curve fit
                    x,y = make_curve_values(params, min(concs_expanded), max(concs_expanded), ref_min_conc=min_conc_ref_dict[compound])
                    log_x_curve = concentrations_to_log_scale(x, ref_min_conc=min_conc_ref_dict[compound])
                    if min_x_val is None:
                        min_x_val = np.nanmin(log_x_curve)
                        previous_concs = concs_expanded
                        previous_x = x
                    elif np.nanmin(log_x_curve) < min_x_val:
                        #print('Found too small x value at index {}'.format(i))
                        #print('Previous was {} now is {}'.format(min_x_val, np.nanmin(log_x_curve)))
                        min_x_val = np.nanmin(log_x_curve)
                        #print('Previous concs and x are')
                        #print(previous_concs)
                        #print(previous_x)
                        #print('Now we have')
                        #print(concs_expanded)
                        #print(x)
                    
                    ticks, tick_labels = make_ticks_and_labels(log_concentrations_scatter)
                    if highlight_array[i]:
                        print('Checking data for highlighted sample')
                        print('Compound is {}'.format(compound))
                        print('Concentrations are the following')
                        print(concs_expanded)
                        print('Ticks and labels are')
                        print(ticks)
                        print(tick_labels)
                        print('RBFs are the following')
                        print(rbfs_expanded)
                        highlighted_rbfs.append(rbfs_expanded)
                        highlighted_rbf_indices.append(log_concentrations_scatter)
                        highlight_index += 1
                        x_ticks_per_compound.append(ticks)
                        x_ticklabels_per_compound.append(tick_labels)
                    x_values_per_compound.append(log_x_curve)
                    y_values_per_compound.append(y)
            else:
                x_values_per_compound.append([0])
                y_values_per_compound.append([1])
        x_values_for_plotting.append(x_values_per_compound)
        y_values_for_plotting.append(y_values_per_compound)
    highlight_index = 0
    if rec_layout:
        nrows = 5
        ncols = int(n_figures/5)
        if (n_figures % 5 != 0):
            ncols += 1
        fig, ax = plt.subplots(figsize=(8*ncols, 5*nrows), nrows=nrows, ncols=ncols)
        drug_index = 0
        for i in range(nrows):
            for k in range(ncols):
                if drug_index < len(sorted_compounds):
                    local_highlight_indices = []
                    for n in range(n_curves):
                        if highlight_array[n]:
                            local_highlight_indices.append(n)
                        elif (np.amax(y_values_for_plotting[drug_index][n]) <= 1.35) and (len(y_values_for_plotting[drug_index][n]) > 1):
                            ax[i,k].plot(x_values_for_plotting[drug_index][n], y_values_for_plotting[drug_index][n], color='black', alpha=0.1, linewidth=3.5)
                    for lh_index in local_highlight_indices:
                        ax[i,k].plot(x_values_for_plotting[drug_index][lh_index], y_values_for_plotting[drug_index][lh_index], color='red', linewidth=3.5)
                        if show_highlight_dots:
                            ax[i,k].scatter(highlighted_rbf_indices[highlight_index][3:], highlighted_rbfs[highlight_index][3:], color='red')
                        highlight_index += 1
                    compound_name = simplify_drug_name(sorted_compounds[drug_index])
                    if not (highlight_scores is None):
                        if input_ranks is None:
                            comp_title = 'Rank {}: {} \n Score: {}'.format(drug_index+1, compound_name, round(highlight_scores[drug_index], 3))
                        else:
                            comp_title = 'Rank {}: {} \n Score: {}'.format(input_ranks[drug_index], compound_name, round(highlight_scores[drug_index], 3))
                    else:
                        if input_ranks is None:
                            comp_title = 'Rank {}: {}'.format(drug_index+1, compound_name)
                        else:
                            comp_title = 'Rank {}: {}'.format(input_ranks[drug_index], compound_name)
                    if not (ic50_string_list is None):
                        comp_title += '\n RBF-IC50 ' + ic50_string_list[drug_index]
                    ax[i,k].set_title(comp_title, fontsize=16, fontweight='bold')
                    ax[i,k].set_xticks(x_ticks_per_compound[drug_index])
                    ax[i,k].set_xticklabels(x_ticklabels_per_compound[drug_index])
                    ax[i,k].set_ylim(-0.1, 1.35)
                    ax[i,k].set_ylabel('Relative blast fraction', fontsize=13)
                    ax[i,k].set_xlabel('Compound concentration', fontsize=13)
                drug_index += 1
    else:
        drug_index = 0
        fig, ax = plt.subplots(figsize=(8, 5*n_figures), nrows=n_figures)
        for i in range(n_figures):
            for k in range(n_curves):
                if highlight_array[k]:
                    ax[i].plot(x_values_for_plotting[i][k], y_values_for_plotting[i][k], color='red', linewidth=3.5)
                    if show_highlight_dots:
                        ax[i].scatter(highlighted_rbf_indices[highlight_index], highlighted_rbfs[highlight_index], color='red')
                    highlight_index += 1
                if (np.amax(y_values_for_plotting[i][k]) <= 1.35) and (len(y_values_for_plotting[i][k]) > 1):
                    ax[i].plot(x_values_for_plotting[i][k], y_values_for_plotting[i][k], color='black', alpha=0.1, linewidth=3.5)
            compound_name = simplify_drug_name(sorted_compounds[i])
            if not (highlight_scores is None):
                if input_ranks is None:
                    comp_title = 'Rank {}: {} \n Score: {}'.format(drug_index+1, compound_name, round(highlight_scores[drug_index], 3))
                else:
                    comp_title = 'Rank {}: {} \n Score: {}'.format(input_ranks[drug_index], compound_name, round(highlight_scores[drug_index], 3))
            else:
                if input_ranks is None:
                    comp_title = 'Rank {}: {}'.format(drug_index+1, compound_name)
                else:
                    comp_title = 'Rank {}: {}'.format(input_ranks[drug_index], compound_name)
            if not (ic50_string_list is None):
                comp_title += '\n RBF-IC50' + ic50_string_list[drug_index]
            ax[i].set_title(comp_title, fontsize=16, fontweight='bold')
            ax[i].set_xticks(x_ticks_per_compound[i])
            ax[i].set_xticklabels(x_ticklabels_per_compound[i])
            ax[i].set_ylim(-0.1, 1.35)
            ax[i].set_ylabel('Relative blast fraction', fontsize=12)
            ax[i].set_xlabel('Compound concentration', fontsize=12)
            drug_index += 1
    fig.tight_layout()
    if savename is None:
        fig.show()
    else:
        fig.savefig(savename)
    return  x_values_for_plotting, y_values_for_plotting, sorted_compounds