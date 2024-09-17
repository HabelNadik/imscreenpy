import numpy as np
import pandas as pd

from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel

from ..cell_prediction_models.prediction_model import ClusterPredictionModel



class LD_stain_BayesianModel:

    def __init__(self, model_selection_strategy='MEDIAN'):
        self.current_model = None
        self.use_model = None
        self.model_selection_strategy = model_selection_strategy
        self.data = None
        self.std_dev_threshold = 2
        self.is_fit = False
        print('Initializing bayesian model with selection strategy {} and std. dev threshold of {}'.format(self.model_selection_strategy, self.std_dev_threshold))

    def fit(self, vals_to_fit, iter_fit=True):
        if iter_fit:
            mixture = self.fit_iter(vals_to_fit)
            return self
        else:
            mixture = self._fit_model(vals_to_fit)
        self.is_fit = True
        return self

    def _fit_model(self, vals_to_fit):
        vals_to_fit = np.array(vals_to_fit).reshape(-1,1)
        self.data = vals_to_fit
        #noise_dist = pomegranate.distributions.NormalDistribution(np.median(vals_to_fit), np.std(vals_to_fit))
        noise_dist = Normal(np.array(np.median(vals_to_fit)).reshape(1,), np.array(np.std(vals_to_fit)).reshape(1,), covariance_type='diag')
        noise_dist.freeze()
        left_dist = Normal(np.array(np.mean(vals_to_fit) - np.std(vals_to_fit)).reshape(1,), np.array(np.std(vals_to_fit) / 2).reshape(1,), covariance_type='diag')
        right_dist = Normal(np.array(np.mean(vals_to_fit) + np.std(vals_to_fit)).reshape(1,), np.array(np.std(vals_to_fit) / 2).reshape(1,), covariance_type='diag')
        mixture = GeneralMixtureModel([left_dist, noise_dist, right_dist], priors=[0.4, 0.2, 0.4])
        mixture = mixture.fit(vals_to_fit)
        mixture.freeze()
        self.current_model = mixture.copy()
        self.set_means()
        return mixture


    def fit_iter(self, input_data, num_iter=5, felix_style=True):
        filter_threshold = self._get_filter_threshold(input_data)
        print('Performing filtering before iterative fitting')
        vals_to_fit = input_data[input_data > filter_threshold].reshape(-1,1)
        print('calculated threshold {} for input data with min {} max {} mean {}. Fitting on {} samples above threshold'.format(filter_threshold, np.min(input_data), np.max(input_data), np.mean(input_data), vals_to_fit.shape[0]))
        all_models = []
        all_thresholds = []
        for i in range(num_iter):
            mixture = self._fit_model(vals_to_fit)
            threshold = self._get_threshold_from_model(mixture, felix_style=felix_style)
            all_models.append(mixture)
            all_thresholds.append(threshold)
            print('Fit {} model(s)'.format(i+1), flush=True)
        if self.model_selection_strategy == 'MEDIAN':
            med = np.median(all_thresholds)
            self.use_model = all_models[all_thresholds.index(med)]
            self.set_means()
            print('Set use model with threshold {} from threshold selection with variance {}'.format(med, np.var(all_thresholds)))
        elif self.model_selection_strategy == 'PROBABILITY':
            probabilities = [np.sum(model.probability(input_data)) for model in all_models]
            self.use_model = all_models[np.argmax(probabilities)]
            print('Set use model with probability {} from set of probabilities with variance {} and mean {}'.format(np.max(probabilities), np.var(probabilities), np.mean(probabilities)))
        self.current_model = self.use_model
        self.data = vals_to_fit
        self.is_fit = True
        return self.use_model
        
    def get_threshold(self, felix_style=True):
        if self.use_model is None:
            self.use_model = self.current_model
        print('Using felix style to get threshold')
        print(felix_style)
        return self._get_threshold_from_model(self.use_model, felix_style=felix_style)

    def _get_filter_threshold(self, intensity_values, num_bins=50):
        print('Getting filter threshold. Input data contains {} nans to be removed'.format(np.sum(np.isnan(intensity_values))), flush=True)
        intensity_values = intensity_values[~np.isnan(intensity_values)]
        
        mean_val = np.mean(intensity_values)
        std_val = np.std(intensity_values)
        subset = intensity_values[intensity_values < (mean_val - (0.5 * std_val))]
        print('Calculating filtering threshold on subset with min {} max {} mean {}'.format(np.min(subset), np.max(subset), np.mean(subset)))
        hist, bins = np.histogram(subset, bins=num_bins)
        for i in range(len(hist)-1, 0, -1):
            if hist[i] == 0:
                threshold = bins[i-1]
                print('Found threshold {} at histogram bin with index {}-1. Threshold directly above is {}'.format(threshold, i, bins[i]))
                return threshold
        else:
            if np.amin(subset) < np.finfo(np.float32).eps:
                return np.amin(subset)
            else:
                return np.amin(subset) - np.finfo(np.float32).eps

    def _get_threshold_from_model(self, model, felix_style=False):
        if model.distributions[0].parameters[0] < model.distributions[2].parameters[0]:
            use_index = 0
        else:
            use_index = 2
        if felix_style:
            threshold = model.distributions[use_index].parameters[0] + (self.std_dev_threshold * model.distributions[use_index].parameters[1])
        else:
            sorted_data = np.sort(self.data)
            predictions = model.predict(sorted_data)
            #print('Generated predictions with the following unique values {}'.format(np.unique(predictions)))
            positive_indices = predictions == use_index
            threshold = sorted_data[positive_indices][0] + np.finfo(np.float16).eps
        return threshold

    def set_means(self):
        if self.use_model is None:
            self.means_ = np.array([self.current_model.distributions[i].parameters[0] for i in range(len(self.current_model.distributions))])
        else:
            self.means_ = np.array([self.use_model.distributions[i].parameters[0] for i in range(len(self.use_model.distributions))])

    def predict(self, input_data):
        if not self.is_fit:
            if isinstance(input_data, pd.DataFrame):
                input_data = input_data.to_numpy()
                print('Viability model! dataframe found. converting to numpy array with shape {}'.format(input_data.shape), flush=True)
            self.fit_iter(input_data)
        threshold = self.get_threshold()
        return input_data < threshold



AML_VIABILITY_LATENT_SCRIPT_DEFAULT_PATH = '/research/lab_gsf/bhaladik/ExTrAct-AML/experimental_code/extract_aml/cluster_scripts/subprocesses/run_viability_prediction_with_latent_on_batch.sh'
AML_VIABILITY_SCRIPT_DEFAULT_PATH = '/research/lab_gsf/bhaladik/ExTrAct-AML/experimental_code/extract_aml/cluster_scripts/subprocesses/run_fixed_size_viability_prediction.sh'
AML_VIABILITY_FILENAME_TEMPLATE = '{}_viability_predictions_gmm_v{}_start{}_end{}.txt'
AML_VIABILITY_LATENT_FILENAME_TEMPLATE = '{}_viability_predictions_gmm_v{}_start{}_end{}_latent_representation.txt'


class ViabilityAAEmodel(ClusterPredictionModel):

    def __init__(self, required_columns, annotation, version_number, channel_id_column, batch_filename_template=AML_VIABILITY_FILENAME_TEMPLATE, \
                  slurm_script_path=AML_VIABILITY_SCRIPT_DEFAULT_PATH, partition_size=50000, cfg=None, name='Viability_AAEmodel', id_df_filename_suffix='_celltypes'):
        super().__init__(required_columns, annotation, slurm_script_path, batch_filename_template, version_number, channel_id_column,\
                  partition_size=partition_size, cfg=cfg, name=name, job_name_prefix='viabiiity', id_df_filename_suffix=id_df_filename_suffix)
        


    