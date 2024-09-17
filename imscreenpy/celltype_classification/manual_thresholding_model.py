import numpy as np


class ManualModel:

    def __init__(self, threshold):
        self.threshold = threshold
        self.data = None

    def fit(self, vals_to_fit):
        self.data = vals_to_fit
        return self


    def get_threshold(self):
        return self.threshold

    def set_means(self):    
        self.means_ = np.array([np.mean(self.data) - np.std(self.data), np.mean(self.data), np.mean(self.data) + np.std(self.data)])

    def predict(self, input_data):
        threshold = self.get_threshold()
        return input_data > threshold