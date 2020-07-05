# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:46:50 2019

@author: Yann Avok
"""
from pandas import concat
from MissingValuesHandler.mixins import (DataPreprocessingMixin, 
                                        ModelMixin, 
                                        PlotMixin)


class RandomForestImputer(DataPreprocessingMixin, ModelMixin, PlotMixin):
    """
    This class uses a random forest to replace missing values in a dataset. 
    It tackles:
    - Samples having missing values in one or more features.
    - Samples having a missing target value and missing values in one or more 
      features: both of them will be predicted and replaced.
      
    Samples that only have a missing target value but None in the features 
    can be predicted with another algorithm(the main one). 
    
    So it will be better to put them in a test set(they won't be considered').
    The main idea is to use random forest's definition of proximity to find 
    the values that are best fit to replace the missing ones.
    
    We have the following parts: 
        
    I - DataPreprocessingMixin:
        1- We get the dataset, sample it if the users requires it,
        isolate samples having potential missing target values, 
        separate the features and the target variable and predict 
        their types:
            - protected method: __data_sampling
            - protected method: __isolate_samples_with_no_target_value
            - protected method: __check_variables_name_validity
            - protected method: __separate_features_and_target_variable
            - protected method: __predict_feature_type
            - protected method: __predict_target_variable_type
            
        2- We retrieve the missing values coordinates(row and column) 
        and fill in the nan cells with initial values:
            - protected method: __retrieve_nan_coordinates
            - protected method: __make_initial_guesses
            
        3- We encode the features and the target variables 
        if they need to be encoded. As of yet, decision trees in Scikit-Learn 
        don't handle categorical variables. So encoding is necessary:
            - protected method: __encode_features
            - protected method: __encode_target_variable
    
    II - ModelMixin
        4- We build our model, fit it, evaluate it and we keep the model having 
        the best out of bag score. We then use it to build the proximity matrix:
            - protected method: __build_ensemble_model
            - protected method: __fit_and_evaluate_ensemble_model
            - protected method: __fill_one_modality
            - protected method: __build_proximity_matrices
            - public method : build_proximity_matrix
        
        5- We use the proximity matrix to compute weighted averages for all 
        missing values(both in categorical and numerical variables):
            - protected method: __compute_weighted_averages
            - protected method:__replace_missing_values_in_encoded_dataframe
            
        6- We check if every value that has been replaced has converged after 
        n iterations. If that's not the case, we go back to step 3 and go for 
        another round of n iterations:
            - protected method: __compute_std_and_entropy
            - protected method: __check_for_final_convergence
            - protected method: __replace_missing_values_in_features_frame
        
        7- We keep the last predictions for the missing target values(if any):
            - protected method: __retrieve_combined_predictions
            - protected method: __replace_missing_values_in_target_variable
    
        8- If all values have converged, we stop everything and save the dataset 
        if a path has been given. 'train' is the main function:
             - protected method: __save_new_dataset
    
    III - PlotMixin
        - protected method: _numerical_categorical_plots
        - public method: get_mds_coordinates
        - public method: show_mds_plot
        - public method: create_weighted_averages_plots
        - public method: create_target_pred_plot
        
    IV - RandomForestImputer
        public  method: train
        
    DATA RETRIEVAL WITH:
       - public method: get_ensemble_model_parameters
       - public method: get_features_type_predictions
       - public method: get_sample
       - public method: get_target_variable_type_prediction
       - public method: get_ensemble_model
       - public method: get_encoded_features
       - public method: get_target_variable_encoded
       - public method: get_proximity_matrix
       - public method: get_distance_matrix
       - public method: get_nan_features_predictions
       - public method: get_nan_target_values_predictions
       - public method: get_mds_coordinates
       - public method: show_mds_plot
       - public method: create_weighted_averages_plots
       - public method: create_target_pred_plot
    """       
    def __init__(self,
                 data, 
                 target_variable_name, 
                 ordinal_features_list=None, 
                 forbidden_features_list=None, 
                 training_resilience=2,  
                 n_iterations_for_convergence=5):
        DataPreprocessingMixin.__init__(self,
                                        data,
                                        target_variable_name, 
                                        ordinal_features_list, 
                                        forbidden_features_list)
        ModelMixin.__init__(self, 
                            training_resilience,  
                            n_iterations_for_convergence)
        PlotMixin.__init__(self)

    def train(self, 
              decimals=0, 
              sample_size=0,
              n_quantiles=0,
              path_to_save_dataset=None):
        """
        This is the main function. At run time, every other private functions 
        will be executed one after another.

        Parameters
        ----------
        decimals : int, optional
            The default is 0.
        sample_size : int, optional
            The default is 0.
        n_quantiles : int, optional
            The default is 0.
        path_to_save_dataset : str, optional
            The default is None

        Returns
        -------
        final_dataset : pandas.core.frame.DataFrame

        """
        #Initializing training
        total_iterations = 0
        self._reinitialize_key_vars()
        self._data_sampling(title="[DATA SAMPLING]: ", 
                            sample_size=sample_size, 
                            n_quantiles=n_quantiles)
        self._check_variables_name_validity()
        self._isolate_samples_with_no_target_value(title="[ISOLATING SAMPLES"\
                                                    "WITH NO TARGET VALUE]: ")
        self._separate_features_and_target_variable()  
        self._predict_feature_type(title="[PREDICTING FEATURE TYPE]: ")
        self._predict_target_variable_type(title="[PREDICTING TARGET VARIABLE"\
                                            "TYPE]: ") 
        self._retrieve_nan_coordinates(title="[RETRIEVING NAN COORDINATES]: ")
        self._make_initial_guesses(title="[MAKING INITIAL GUESSES]: ")
        self._encode_target_variable()
        self._retrieve_target_variable_class_mappings()
        
        while not self._has_converged:
            for iteration in range(1, self._last_n_iterations + 1):
                total_iterations += 1
                self._encode_features()
                #1- MODEL BULDING
                text = (f"[{iteration}/{total_iterations}-"
                        "BUILDING RANDOM FOREST]: ")
                self._build_ensemble_model(title=text) 
        
                #2- FITTING AND EVALUATING THE MODEL
                text = (f"[{iteration}-FITTING AND EVALUATING MODEL]: ")
                self._fit_and_evaluate_ensemble_model(title=text)
                
                #3- BUILDING PROXIMITY MATRIX
                text=(f"[{iteration}-BUILDING PROXIMITY MATRIX TREES/OOB "
                      f"{self._estimator.n_estimators}"
                      f"/{self._best_oob_score}]: ")
                self._proximity_matrix = self.build_proximity_matrix(title=text)
                self._retrieve_combined_predictions()  
                #4- COMPUTING WEIGHTED AVERAGES
                text = f"[{iteration}-COMPUTING WEIGHTED AVERAGES]: "
                self._compute_weighted_averages(title=text, 
                                                decimals=decimals)
                        
                #5- REPLACING NAN VALUES IN ENCODED DATA 
                text = f"[{iteration}-REPLACING MISSING VALUES]: "
                self._replace_missing_values_in_features_frame(title=text)
            self._compute_std_and_entropy()
            self._check_and_remove_convergent_values()
            self._check_for_final_convergence()
        print(f"\n- TOTAL ITERATIONS: {total_iterations}")
        self._replace_missing_values_in_target_variable()
        #We save the final dataset if a path is given
        all_data = (self._features, self._target_variable)
        final_dataset = concat(all_data, axis=1)  
        final_dataset = self._reconstruct_original_data(final_dataset, sample_size)
        self._save_new_dataset(final_dataset, path_to_save_dataset)
        return  final_dataset 

    
    
    
