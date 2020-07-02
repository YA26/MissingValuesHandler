# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:46:50 2019

@author: Yann Avok
"""
from collections import defaultdict, deque, Counter
import MissingValuesHandler.custom_exceptions as customs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from MissingValuesHandler.data_type_identifier import DataTypeIdentifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from MissingValuesHandler import constants as const
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold
from copy import copy
import matplotlib.pyplot as plt 
import progressbar as pb
import numpy as np
import pandas as pd
import os

class RandomForestImputer():
    """
    This class uses a random forest to replace missing values in a dataset. 
    It tackles:
    - Samples having missing values in one or more features.
    - Samples having a missing target value and missing values in one or more 
      features: both of them will be predicted and replaced.
      
    Samples that only have a missing target value but none in the features 
    can be predicted with another algorithm(the main one). 
    
    So it will be better to put them in a test set(they won't be considered').
    The main idea is to use random forest's definition of proximity to find 
    the values that are best fit to replace the missing ones.
    
    We have the following parts: 

    1- We get the dataset, isolate samples having potential missing target 
    values, separate the features and the target variable 
    and predict their types:
        - private method: __isolate_samples_with_no_target_value
        - private method: __check_variables_name_validity
        - private method: __separate_features_and_target_variable
        - private method: __predict_feature_type
        - private method: __predict_target_variable_type
        
    2- We retrieve the missing values coordinates(row and column) 
    and fill in the nan cells with initial values:
        - private method: __retrieve_nan_coordinates
        - private method: __make_initial_guesses
        
    3- We encode the features and the target variables 
    if they need to be encoded. As of yet, decision trees in Scikit-Learn 
    don't handle categorical variables. So encoding is necessary:
        - private method: __encode_features
        - private method: __encode_target_variable
    
    4- We build our model, fit it, evaluate it and we keep the model having 
    the best out of bag score. We then use it to build the proximity matrix:
        - private method: __build_ensemble_model
        - private method: __fit_and_evaluate_ensemble_model
        - private method: __fill_one_modality
        - private method: __build_proximity_matrices
        - public method : build_proximity_matrix
    
    5- We use the proximity matrix to compute weighted averages for all 
    missing values(both in categorical and numerical variables):
        - private method: __compute_weighted_averages
        - private method:__replace_missing_values_in_encoded_dataframe
        
    6- We check if every value that has been replaced has converged after 
    n iterations. If that's not the case, we go back to step 3 and 
    go for another round of n iterations:
        - private method: __compute_standard_deviations
        - private method: __check_for_convergence
        - private method: __replace_missing_values_in_original_dataframe
        
    7- If all values have converged, we stop everything and save the dataset 
    if a path has been given. 'train' is the main function:
         - private method: __save_new_dataset
         - public  method: train
     """   
    class Decorators():
        """
        Timer decorator: used to time functions
        """
        @staticmethod
        def timeit(method):
            def timed(*args, **kwargs):
                widgets = [kwargs["title"], 
                           pb.Percentage(), 
                           ' ', 
                           pb.Bar(marker="#"), 
                           ' ', 
                           pb.ETA()]
                timer = pb.ProgressBar(widgets=widgets, maxval=100).start()
                kwargs["update"] = timer.update
                kwargs["maxval"] = timer.maxval        
                result = method(*args, **kwargs)
                timer.finish()            
                return result
            return timed
 
              
    def __init__(self, 
                 data, 
                 target_variable_name, 
                 ordinal_features_list=None, 
                 forbidden_features_list=None, 
                 training_resilience=2,  
                 n_iterations_for_convergence=5):
        """
        Constructor
        
        Parameters
        ----------
        data : pandas.core.frame.DataFrame
        
        target_variable_name : str
        
        ordinal_features_list : list, optional
            The default is None.
        forbidden_features_list : list, optional
            The default is None.
        training_resilience : int, optional
            The default is 2.
        n_iterations_for_convergence : int, optional
            The default is 5.

        Raises
        ------
        customs.TrainingResilienceValueError()
            EXCEPTION RAISED WHEN TRAINING_RESILIENCE LOWER THAN 2

        Returns
        -------
        None.
        """
        if training_resilience < 2:
            raise customs.TrainingResilienceValueError()
            
        #Data type identifier object
        self.__data_type_identifier = DataTypeIdentifier()
        
        #Main variables
        self.__original_data = None
        self.__original_data_backup = data.copy(deep=True)
        self.__original_data_sampled = pd.DataFrame()
        self.__orginal_data_temp = pd.DataFrame()
        self.__data_null_index = None
        self.__idx_no_target_value = None
        self.__features = None
        self.__target_variable = None
        self.__features_type_predictions = None
        self.__target_variable_type_prediction = None
        self.__encoded_features_model = None
        self.__encoded_features_pred = None
        self.__target_var_encoded = None
        self.__proximity_matrix = []
        self.__distance_matrix = []
        self.__missing_values_coordinates = []
        self.__number_of_nan_values = 0
        self.__label_encoder_features = LabelEncoder()
        self.__label_encoder_target_vars = LabelEncoder()
        self.__mappings_target_variable = defaultdict()
        self.__standard_deviations = defaultdict()
        self.__converged_values = defaultdict()
        self.__all_weighted_averages = defaultdict(list)
        self.__all_weighted_averages_copy = defaultdict(list)
        self.__nan_target_variable_preds = defaultdict(list)
        self.__predicted_target_value = defaultdict()
        self.__training_resilience = training_resilience
        self.__nan_values_remaining_check = deque(maxlen=training_resilience) 
        self.__last_n_iterations = n_iterations_for_convergence
        self.__has_converged = None
        self.__target_variable_name = target_variable_name 
        
        if ordinal_features_list is None:
            self.__ordinal_vars = []
        else:
            self.__ordinal_vars = ordinal_features_list
        if forbidden_features_list is None:
            self.__forbidden_features = []
        else:
            self.__forbidden_features = forbidden_features_list
            
        #Random forest variables
        self.__estimator = None
        self.__n_estimators = None
        self.__additional_estimators = None
        self.__max_depth = None
        self.__min_samples_split = None 
        self.__min_samples_leaf = None
        self.__min_weight_fraction_leaf = None 
        self.__max_features = None
        self.__max_leaf_nodes = None
        self.__min_impurity_decrease = None
        self.__min_impurity_split = None
        self.__n_jobs = None 
        self.__random_state = None
        self.__verbose = None
        self.__bootstrap = True
        self.__oob_score = True
        self.__best_oob_score = 0
        self.__warm_start = True
        
        #Weighted averages in case of sampling
        self.__all_weighted_averages_sample = None
        self.__converged_values_sample = None
        self.__divergent_values_sample = None
        self.__predicted_target_value_sample = None
        self.__target_value_predictions_sample = None
        
     
    def set_ensemble_model_parameters(self,
                                      additional_estimators=20,
                                      n_estimators=30,
                                      max_depth=None,
                                      min_samples_split=20,
                                      min_samples_leaf=20,
                                      min_weight_fraction_leaf=0.0, 
                                      max_features='auto',
                                      max_leaf_nodes=None,
                                      min_impurity_decrease=0.0,
                                      min_impurity_split=None,
                                      n_jobs=-1,
                                      random_state=None,
                                      verbose=0):
        """
        Parameters
        ----------
        additional_estimators : int, optional
            The default is 20.
        n_estimators : int, optional
            The default is 30.
        max_depth : int, optional
            The default is None.
        min_samples_split : int, optional
            The default is 20.
        min_samples_leaf : int, optional
            The default is 20.
        min_weight_fraction_leaf : float, optional
            The default is 0.0.
        max_features : str, optional
            The default is 'auto'.
        max_leaf_nodes : int, optional
            The default is None.
        min_impurity_decrease : float, optional
            The default is 0.0.
        min_impurity_split : float, optional
            The default is None.
        n_jobs : int, optional
            DESCRIPTION. The default is -1.
        random_state : int, optional
            DESCRIPTION. The default is None.
        verbose : int, optional
            The default is 0.

        Returns
        -------
        None.
        """

        self.__additional_estimators = additional_estimators
        self.__n_estimators = n_estimators
        self.__max_depth = max_depth
        self.__min_samples_split = min_samples_split 
        self.__min_samples_leaf = min_samples_leaf
        self.__min_weight_fraction_leaf = min_weight_fraction_leaf
        self.__max_features = max_features
        self.__max_leaf_nodes = max_leaf_nodes
        self.__min_impurity_decrease = min_impurity_decrease
        self.__min_impurity_split = min_impurity_split
        self.__n_jobs = n_jobs 
        self.__random_state = random_state
        self.__verbose = verbose
 
    
    def get_ensemble_model_parameters(self):
        """
        Retrives random forest regressor or classifier parameters
        
        Returns
        -------
        dict
        """
        return {"n_estimators":self.__n_estimators,                
                "additional_estimators":self.__additional_estimators,
                "max_depth":self.__max_depth,           
                "min_samples_split":self.__min_samples_split,                 
                "min_samples_leaf":self.__min_samples_leaf,
                "min_weight_fraction_leaf":self.__min_weight_fraction_leaf,         
                "max_features":self.__max_features,                     
                "max_leaf_nodes":self.__max_leaf_nodes,                   
                "min_impurity_decrease":self.__min_impurity_decrease,            
                "min_impurity_split":self.__min_impurity_split,               
                "n_jobs":self.__n_jobs,                           
                "random_state":self.__random_state,                     
                "verbose":self.__verbose,                          
                "bootstrap":self.__bootstrap,                        
                "oob_score":self.__oob_score,                        
                "warm_start":self.__warm_start}
    
    
    def get_features_type_predictions(self):
        """ 
        Retrieves all features predictions type whether they are numerical 
        or categorical.
        
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self.__features_type_predictions
     
     
    def get_sample(self):
        """
        Retrieves sample on which the ensemble model has been trained on.

        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self.__original_data_sampled
        
        
    def get_target_variable_type_prediction(self):
        """
        Retrieves prediction about the type of the target variable whether it 
        is numerical or categorical.
        
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self.__target_variable_type_prediction
     
        
    def get_ensemble_model(self):
        """
        Random forest model (classifier or regressor)
        
        Returns
        -------
        sklearn.ensemble._forest
        """
        return self.__estimator
    
    
    def get_encoded_features(self):
        """
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self.__encoded_features_model
    
    
    def get_target_variable_encoded(self):
        """
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self.__target_var_encoded
      
        
    def get_proximity_matrix(self):
        """
        Retrieves the last proximity matrix built with the optimal 
        random forest.
        
        Returns
        -------
        numpy.ndarray
        """
        return self.__proximity_matrix
    
    
    def get_distance_matrix(self):
        """
        Retrieves distance matrix which is equals to 1 - proximity matrix.

        Returns
        -------
        numpy.ndarray
        """
        if len(self.__distance_matrix) == 0:
            self.__distance_matrix = 1-self.__proximity_matrix
        return self.__distance_matrix
            
    
    def get_nan_features_predictions(self, option):
        """
        Predictions for nan values  
        
        Parameters
        ----------
        option : str
            - all: retrieves both convergent and divergent nan values  
            - conv: retrieves all nan values that converged.
            - div: retrieves all nan values that were not able to converge
        
        Returns
        -------
        dict
        """
        dict_a_options = {"all":self.__all_weighted_averages_sample, 
                          "conv":self.__converged_values_sample,
                          "div":self.__divergent_values_sample}
        
        dict_b_options = {"all":self.__all_weighted_averages_copy,
                          "conv":self.__converged_values,
                          "div":self.__all_weighted_averages}
        
        if not dict_a_options[option] and self.__data_null_index:       
            dict_= {(self.__data_null_index[coordinate[0]], coordinate[1]):
                    predicted_value 
                    for coordinate, predicted_value 
                    in dict_b_options[option].items()}
            dict_a_options[option] = dict_         
        return (dict_a_options[option] 
                if dict_a_options[option] 
                else dict_b_options[option])
        
    
    def get_nan_target_values_predictions(self, option):
        """
        Predictions for potential nan target values

        Parameters
        ----------
        option : str
            - all: all predictions of potential missing target values 
            - one: last predicted values for potential missing target values

        Returns
        -------
        dict
        """
        dict_a_options = {"all":self.__nan_target_variable_preds, 
                          "one":self.__predicted_target_value}
        
        dict_b_option = {"all": self.__target_value_predictions_sample,
                         "one":self.__predicted_target_value_sample}
        
        if dict_a_options[option]:
            if not  dict_b_option[option] and self.__data_null_index:
                dict_ = {(self.__data_null_index[coordinate]):predicted_value 
                         for coordinate, predicted_value 
                         in dict_a_options[option].items()}
                dict_b_option[option] = dict_
        return  (dict_b_option[option] 
                 if dict_b_option[option] 
                 else dict_a_options[option])


    def get_mds_coordinates(self, n_dimensions, distance_matrix):
        """
        Multidimensional scaling coordinates to reduce distance matrix 
        to n_dimensions(< n_dimensions of distance matrix)
        
        Parameters
        ----------
        n_dimensions : int
            NUMBER OF DIMENSIONS FOR MDS.
        distance_matrix : numpy.array
        
        Returns
        -------
        coordinates : numpy.array
            MDS COORDINATES
        """
        coordinates=None
        if n_dimensions<len(distance_matrix):
            mds=manifold.MDS(n_components=n_dimensions, 
                             dissimilarity='precomputed')
            coordinates=mds.fit_transform(distance_matrix)
        else:
            print("n_dimensions > n_dimensions of distance matrix")
        return coordinates
  
      
    def show_mds_plot(self, coordinates, plot_type="2d", path_to_save=None):
        """
        2d or 3d  multidimensional scaling plot

        Parameters
        ----------
        coordinates : numpy.array
            MDS coordinates after dimensionality reduction.
        plot_type : str, optional
            2d/3d for a 2 or 3 dimensional plot. The default is "2d".
        path_to_save : str, optional
            The default is None.

        Returns
        -------
        None.

        """
        plot_type = plot_type.lower().strip()
        filename = ""
        if plot_type == "2d":
            plt.scatter(coordinates[:,0], coordinates[:,1])
            plt.title("2D MDS PLOT")
            plt.xlabel("MDS1")
            plt.ylabel("MDS2") 
            plt.show()
            filename = "2d_mds_plot.png"
        elif plot_type == "3d":
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection=plot_type)
            ax.scatter(coordinates[:,0], 
                       coordinates[:,1], 
                       coordinates[:,2], 
                       linewidths=1, 
                       alpha=.7,
                       s = 200)
            plt.title("3D MDS PLOT")
            plt.show()
            filename = "3d_mds_plot.png"
        if path_to_save:  
            plt.savefig(os.path.join(path_to_save, filename))
      
            
    @Decorators.timeit
    def __data_sampling(self, title, update, maxval, sample_size, n_quantiles):
        """
        Draws a representative sample from the original dataset.
        It can be used when the dataset is too big.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable
        sample_size [0;1[ : int
            Allows to draw a representative sample from the data.
        n_quantiles : int
            Allows to draw a representative sample from the data when 
            the target variable is numerical. Default value at 0 if the 
            variable is categorical.

        Returns
        -------
        None.

        """
        if sample_size:
            data_sampled = None
            null_checklist = self.__original_data.isnull().any(axis=1)
            data_null = self.__original_data[null_checklist]
            data_no_null = self.__original_data.dropna()
            target_no_null = data_no_null[self.__target_variable_name]
            try:
                train_test = train_test_split(data_no_null, 
                                              test_size=sample_size, 
                                              random_state=42, 
                                              stratify=target_no_null)
                self.__orginal_data_temp = train_test[0]
                data_sampled = train_test[1]
            except ValueError:
                k_bins = KBinsDiscretizer(n_quantiles, "ordinal")
                target_no_null = np.array(target_no_null).reshape((-1, 1))
                y_binned = k_bins.fit_transform(target_no_null)
                train_test = train_test_split(data_no_null, 
                                              test_size=sample_size, 
                                              random_state=42, 
                                              stratify=y_binned) 
                self.__orginal_data_temp = train_test[0]
                data_sampled = train_test[1]
            self.__original_data = pd.concat([data_sampled, data_null])
            self.__original_data = self.__original_data.reset_index() 
            self.__data_null_index = self.__original_data["index"].to_dict()
            self.__original_data = self.__original_data.drop("index", axis=1)
            self.__original_data_sampled = self.__original_data.copy(deep=True)
        update(1)
      
        
    def __reconstruct_original_data(self, final_dataset, sample_size):
        """
        Reconstruct the original dataset with the new values if a sample has
        been drawn.

        Parameters
        ----------
        final_dataset : pandas.core.frame.DataFrame
            
        sample_size : int
          
        Returns
        -------
        final_dataset : pandas.core.frame.DataFrame
        """
        if sample_size:
            final_dataset = final_dataset.rename(self.__data_null_index)
            final_dataset = pd.concat([self.__orginal_data_temp, final_dataset])
            final_dataset.sort_index(inplace=True)
        return final_dataset
    
            
    def __check_variables_name_validity(self):
        """
        1- Verifies whether variables in 'forbidden_features_list' 
            or 'forbidden_features_list' exist in the dataset. 
        2- Verifies whether one variable is not mentioned twice in both lists.

        Raises
        ------
        - customs.VariableNameError
        - customs.VariableNameError
        - customs.VariableNameError

        Returns
        -------
        None.

        """
        columns_names_set = set(self.__original_data.columns.tolist())
        forbidden_variables_set = set(self.__forbidden_features)
        ordinal_variables_set = set(self.__ordinal_vars)
        forbid_inter = forbidden_variables_set.intersection(columns_names_set)
        ordi_inter = ordinal_variables_set.intersection(columns_names_set)   
        unknown_forbidden_set = forbidden_variables_set - forbid_inter
        unknown_ordinal_set = ordinal_variables_set - ordi_inter
        #1
        if unknown_forbidden_set:
            text = (f"Variable(s) {unknown_forbidden_set} in" 
                    " forbidden_features_list not present in the dataset!")
            raise customs.VariableNameError(text)
        if unknown_ordinal_set:
            text = (f"Variable(s) {unknown_ordinal_set} in"
                    " forbidden_features_list not present in the dataset!")
            raise customs.VariableNameError(text)
        #2               
        duplicates_check = np.in1d(self.__ordinal_vars, self.__forbidden_features)
        if duplicates_check:
            duplicates_names = np.where(duplicates_check)[0]
            forbidden_features_list = np.array(self.__ordinal_vars)
            text = (f"Variable(s) {forbidden_features_list[duplicates_names]}"
                    " in forbidden_features_list can't be duplicated in"
                    " forbidden_features_list")
            raise customs.VariableNameError(text)
    
    
    @Decorators.timeit
    def __isolate_samples_with_no_target_value(self, title, update, maxval):
        """
        Separates samples that have a missing target value and one or 
        multiple missing values in their features.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Raises
        ------
        - customs.TargetVariableNameError
        - customs.TrainingSetError

        Returns
        -------
        None.
        """
        diff_idx = None
        try:
            target_variable = self.__original_data[self.__target_variable_name]
            nan_target_check = target_variable.index[target_variable.isnull()]
            nan_target = self.__original_data.loc[nan_target_check]
            features_check = nan_target.columns != self.__target_variable_name
            features = nan_target.loc[: , features_check]  
            nan_features = features.isnull().any(axis=1)
            nan_idx = nan_features.loc[nan_features].index
            samples_nan_target_value = set(nan_idx)
            nan_samples_target_value = set(nan_target_check)
            diff_idx = (nan_samples_target_value
                        .difference(samples_nan_target_value))
            self.__idx_no_target_value = list(nan_idx)
        except KeyError:
            text = (f"Target variable '{self.__target_variable_name}'"
                    " does not exist!")
            raise customs.TargetVariableNameError(text)
        else:
            if diff_idx:
                text = (f"No target value in sample(s) {diff_idx} but no"
                        " missing values in feature(s) as well. Remove"
                        " {index_for_test_set} from this set: that can be"
                        " predicted with another ML algorithm")
                raise customs.TrainingSetError(text)
        update(1)


    def __separate_features_and_target_variable(self):
        """
        Returns
        -------
        None
        """
        self.__features = (self.__original_data
                           .drop(self.__target_variable_name, axis=1))   
        self.__target_variable = (self.__original_data 
                                  .loc[:, self.__target_variable_name] 
                                  .copy(deep=True))
  
    
    @Decorators.timeit
    def __predict_feature_type(self, title, update, maxval):
        """
        Predicts if a feature is either categorical or numerical.
        
        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Returns
        -------
        None.
        """
        self.__features_type_predictions = (self.__data_type_identifier
                                            .predict(self.__features, 0))
        update(1)
     
        
    @Decorators.timeit    
    def __predict_target_variable_type(self, title, update, maxval):
        """
        Predicts if the target variable is either categorical or numerical.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Returns
        -------
        None.

        """
        target_variable = self.__target_variable.to_frame()
        self.__target_variable_type_prediction = (self.__data_type_identifier
                                                  .predict(target_variable, 0))
        update(1)


    @Decorators.timeit
    def __retrieve_nan_coordinates(self, title, update, maxval):
        """
        Gets the coordinates(row and column) of every empty cell in the 
        features dataset.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Raises
        ------
        customs.NoMissingValuesError

        Returns
        -------
        None.
        """
        #Checking if there are any missing values in the dataset.
        if not self.__features.isnull().values.any():
            text = "No missing values were found in the dataset!"
            raise customs.NoMissingValuesError(text)
        
        features_nan_check = self.__features.isnull().any()
        features_nan_name = self.__features.columns[features_nan_check]
        features_nan = self.__features[features_nan_name]
          
        for iterator, feature_nan in enumerate(features_nan):
            empty_cells_checklist = self.__features[feature_nan].isnull()
            row_coordinates = (self.__features[feature_nan]
                               .index[empty_cells_checklist])
            column_coordinate = feature_nan
            col_row_combinations = [column_coordinate]*len(row_coordinates)
            nan_coordinates = list(zip(row_coordinates, col_row_combinations))
            self.__missing_values_coordinates.extend(nan_coordinates)
            update(iterator*(maxval/len(features_nan)))
                      
        #Getting the total number of missing values for future purposes.
        self.__number_of_nan_values = len(self.__missing_values_coordinates)
    
    
    @Decorators.timeit       
    def __make_initial_guesses(self, title, update, maxval):
        """
        Replaces empty cells with initial values in the features dataset:
            - mode for categorical variables 
            - median for numerical variables

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Returns
        -------
        None.
        """
        #Getting features that contains null values only  
        null_columns_checklist = self.__features.isnull().any() 
        null_columns_names = (self.__features
                              .columns[null_columns_checklist])

        #Getting variable type predictions for features containing null values 
        nan_predictions = (self.__features_type_predictions
                           .loc[null_columns_names, "Predictions"])

        #Getting numerical and categorical features' names
        num_check = nan_predictions==const.NUMERICAL
        cat_check = nan_predictions==const.CATEGORICAL
        numerical_variables_names = nan_predictions[num_check].index
        categorical_variables_names = nan_predictions[cat_check].index

        #Calculating medians and modes
        medians = self.__features[numerical_variables_names].median()
        modes = self.__features[categorical_variables_names].mode().iloc[0]
        initial_guesses = pd.concat([medians, modes])

        #Replacing initial_guesses in the dataset
        self.__features.fillna(initial_guesses, inplace=True)
        update(1)
        
            
    def __encode_features(self):
        """
        Encodes every categorical feature the user wants to encode. 
        Any feature mentioned in 'forbidden_features_list' 
        will not be considered.
        1- No numerical variable will be encoded
        2- All categorical variables will be encoded as dummies by default. 
            If one wants to encode ordinal categorical variable, he can do so 
            by adding it to the forbidden_features_list.

        Returns
        -------
        None.

        """
        predictions = self.__features_type_predictions["Predictions"]     
        #Checklists to highlight categorical and numerical variables only.
        categorical_var_check = list(predictions==const.CATEGORICAL)
        numerical_var_check = list(predictions==const.NUMERICAL)
        
        #Getting variables' name that are either categorical or numerical. 
        categorical_vars_names = (predictions.index[categorical_var_check]
                                 .to_list())
        numerical_vars_names = (predictions.index[numerical_var_check]
                               .to_list())
            
        #Retrieving all numerical and categorical variables
        numerical_vars = self.__features[numerical_vars_names]   
        categorical_vars = self.__features[categorical_vars_names]
        
        #Separating nominal and ordinal categorical variables
        if self.__ordinal_vars:
            nominal_cat_vars = (categorical_vars
                                .drop(self.__ordinal_vars, axis=1))
            ordinal_cat_vars = (categorical_vars
                                .loc[:, self.__ordinal_vars])
        
            #Label encoding ordinal categorical variables 
            transform = self.__label_encoder_features.fit_transform
            encoded_ordinal_cat_vars = ordinal_cat_vars.apply(transform)
            
            #One-Hot encoding nominal categorical variables 
            # 'a_c' stands for authorized columns
            a_c = [column_name for column_name in nominal_cat_vars.columns if 
                   column_name not in self.__forbidden_features] 
            encoded_nominal_cat_vars = pd.get_dummies(nominal_cat_vars, 
                                                      columns=a_c)
            
            #Gathering numericals variables and encoded categorical ones
            all_encoded_data  = (numerical_vars, 
                                 encoded_ordinal_cat_vars, 
                                 encoded_nominal_cat_vars)
            self.__encoded_features_model = pd.concat(all_encoded_data, axis=1)    
        elif categorical_vars_names:
            a_c = [column_name for column_name in categorical_vars.columns if 
                   column_name not in self.__forbidden_features] 
            encoded_cat_vars = pd.get_dummies(categorical_vars, 
                                              columns=a_c)
            #Gathering numerical variables and nominal categorical variables
            all_encoded_data = (numerical_vars, encoded_cat_vars)
            self.__encoded_features_model = pd.concat(all_encoded_data, axis=1)
        else:
            self.__encoded_features_model = self.__features.copy(deep=True)
   
        '''
        Creating two separates encoded_features sets 
        if self.__idx_no_target_value is not empty:
        1- One for the ensemble model that have a missing target value
        2- Another for building the proximity matrix and computing the weighted
        averages
        '''    
        if len(self.__idx_no_target_value)!=0:
            self.__encoded_features_pred = (self.__encoded_features_model
                                            .copy(deep=True))
            self.__encoded_features_model.drop(self.__idx_no_target_value, 
                                               inplace=True)
        else:
            self.__encoded_features_pred = (self.__encoded_features_model
                                            .copy(deep=True))
              
                 
    def __encode_target_variable(self):
        """
        Encodes the target variable if it is permitted by the user:
        - If the name of the variable is not in 'forbidden_features_list'
        - If the target variable is numerical it will not be encoded so there's 
            no need to put it in 'forbidden_features_list'.
        The target variable will always be label encoded because the trees in 
        the random forest aren't using it for splitting purposes.

        Returns
        -------
        None.
        """
        prediction = self.__target_variable_type_prediction["Predictions"]
        target_var_cleansed = self.__target_variable.copy(deep=True)
        #Removal of samples having a missing target_value
        if len(self.__idx_no_target_value)!=0:
            target_var_cleansed.drop(self.__idx_no_target_value, inplace=True)
        self.__target_var_encoded = target_var_cleansed
        #We encode it if the variable is categorical
        if (self.__target_variable_name not in self.__forbidden_features and 
            prediction.any()==const.CATEGORICAL):
            self.__target_var_encoded  = (self.__label_encoder_target_vars
                                          .fit_transform(target_var_cleansed))
            
            
    def __retrieve_target_variable_class_mappings(self):
        """
        Returns mappings of our target variable modalities if the latter is 
        categorical.

        Returns
        -------
        None.

        """
        original_values = None
        try:
            original_values = self.__label_encoder_target_vars.classes_ 
            encoded_values = (self.__label_encoder_target_vars
                              .transform(original_values)) 
            all_values = zip(original_values, encoded_values)
            for original_value, encoded_value in all_values:
                self.__mappings_target_variable[encoded_value] = original_value
        except AttributeError:
            pass
          
    @Decorators.timeit         
    def __build_ensemble_model(self, title, update, maxval):
        """
        Builds an ensemble model: random forest classifier or regressor.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Returns
        -------
        None.

        """
        Model = {const.CATEGORICAL:RandomForestClassifier, 
                 const.NUMERICAL:RandomForestRegressor}
        type_ = self.__target_variable_type_prediction["Predictions"].any()
        self.__estimator = Model[type_](n_estimators=self.__n_estimators,
                                        max_depth=self.__max_depth, 
                                        min_samples_split=self.__min_samples_split, 
                                        min_samples_leaf=self.__min_samples_leaf, 
                                        min_weight_fraction_leaf=self.__min_weight_fraction_leaf, 
                                        max_features=self.__max_features, 
                                        max_leaf_nodes=self.__max_leaf_nodes, 
                                        min_impurity_decrease=self.__min_impurity_decrease, 
                                        min_impurity_split=self.__min_impurity_split, 
                                        bootstrap=self.__bootstrap, 
                                        oob_score=self.__oob_score, 
                                        n_jobs=self.__n_jobs, 
                                        random_state=self.__random_state, 
                                        verbose=self.__verbose,
                                        warm_start=self.__warm_start)
        update(1)

           
    @Decorators.timeit 
    def __fit_and_evaluate_ensemble_model(self, title, update, maxval):
        """
        Fits and evaluates the model. 
        1- We compare the out-of-bag score at iteration i-1 with the one at 
        iteration i.
        2- If the latter is lower than the former or equals to it, we stop 
            fitting the model and we keep the one at i-1.
        3- If it's the other way around, we add more estimators to the total 
            number of estimators we currently have.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Returns
        -------
        None.

        """
        precedent_out_of_bag_score = 0
        current_out_of_bag_score = 0
        precedent_estimator = None
        while (current_out_of_bag_score > precedent_out_of_bag_score or not 
               current_out_of_bag_score):
            precedent_estimator = copy(self.__estimator)
            self.__estimator.fit(self.__encoded_features_model, 
                                 self.__target_var_encoded) 
            precedent_out_of_bag_score = current_out_of_bag_score
            current_out_of_bag_score = self.__estimator.oob_score_
            self.__estimator.n_estimators += self.__additional_estimators
            
        #Keeping the configuration of the previous model(i.e the optimal one)
        self.__best_oob_score = np.round(precedent_out_of_bag_score, 2)
        self.__estimator.n_estimators -= self.__additional_estimators
        self.__estimator = precedent_estimator
        update(1)


    def __fill_one_modality(self, 
                            predicted_modality, 
                            prediction_dataframe, 
                            encoded_features):
        """
        Handles every modality separately and construct every ad-hoc proximity 
        matrix.

        Parameters
        ----------
        predicted_modality : int/foat
      
        prediction_dataframe : pandas.core.frame.DataFrame
       
        encoded_features : numpy.array
       

        Returns
        -------
        one_modality_matrix : numpy.array
    
        """
        matrix_shape = (len(encoded_features), len(encoded_features))
        one_modality_matrix = np.zeros(matrix_shape)
        prediction_checklist = prediction_dataframe[0]==predicted_modality
        idx_check = prediction_dataframe.index[prediction_checklist].tolist() 
        idx_check = np.array(idx_check)
        #Using broadcasting to replace null values by 1
        one_modality_matrix[idx_check[:, None], idx_check] = 1
        return one_modality_matrix
        
    
    def __build_prox_matrices(self, 
                              iterator, 
                              update, 
                              maxval, 
                              predictions, 
                              prediction, 
                              encoded_features):
        """
        Builds proximity matrices.
            1- We run all the data down the first tree and output predictions.
            2- If two samples fall in the same node (same predictions) 
                we count it as 1.
            3- We do the same for every single tree, sum up the proximity 
                matrices and divide the total by the number of estimators.

        Parameters
        ----------
        iterator : int
            iterator for progress bar
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable
        predictions : pandas.core.frame.DataFrame
           
        prediction : pandas.core.frame.DataFrame
           
        encoded_features : pandas.core.frame.DataFrame
        
        Returns
        -------
        proximity_matrix : numpy.array
      
        """
        possible_predictions = prediction[0].unique()
        target_type_pred = self.__target_variable_type_prediction.values[0,0]
        if target_type_pred == const.CATEGORICAL:
            array_to_int = np.vectorize(lambda x: np.int(x))
            possible_predictions = array_to_int(possible_predictions)       
        one_modality_matrix = [self.__fill_one_modality(predicted_modality, 
                                                        prediction, 
                                                        encoded_features) 
                               for predicted_modality in possible_predictions]
        proximity_matrix = sum(one_modality_matrix)
        update(iterator*(maxval/len(predictions)))
        return proximity_matrix

    
    def __frame(self, estimator, encoded_features):
        """
        Refactor method

        Parameters
        ----------
        estimator : sklearn.tree

        encoded_features : pandas.core.frame.DataFrame
     
        Returns
        -------
        pandas.core.frame.DataFrame
    
        """
        return pd.DataFrame(estimator.predict(encoded_features))
  
    
    @Decorators.timeit
    def build_proximity_matrix(self, 
                               title, 
                               update, 
                               maxval, 
                               ensemble_estimator, 
                               encoded_features):
        """
        Builds final proximity matrix: sum of all proximity matrices.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable
        ensemble_estimator : sklearn.ensemble._forest
         
        encoded_features : pandas.core.frame.DataFrame
  

        Returns
        -------
        final_proximity_matrix : pandas.core.frame.DataFrame

        """

        all_estimators_list = ensemble_estimator.estimators_
        number_of_estimators = ensemble_estimator.n_estimators  
        predictions = [self.__frame(estimator, encoded_features) 
                       for estimator in all_estimators_list]
        proximity_matrices = [self.__build_prox_matrices(iterator, 
                                                        update, 
                                                        maxval, 
                                                        predictions, 
                                                        prediction, 
                                                        encoded_features) 
                              for iterator, prediction in enumerate(predictions)] 
        final_proximity_matrix = sum(proximity_matrices)/number_of_estimators
        return final_proximity_matrix
     
    
    def __retrieve_combined_predictions(self):
        """
        Predicts new values for the target variable(if there is any nan).

        Returns
        -------
        None.

        """

        combined_pred = self.__estimator.predict(self.__encoded_features_pred)
        for index in self.__idx_no_target_value:
            sample_pred = combined_pred[index]
            if  self.__mappings_target_variable:
                realval = self.__mappings_target_variable[sample_pred]
                self.__nan_target_variable_preds[index].append(realval)
            else:
                self.__nan_target_variable_preds[index].append(sample_pred)
   
             
    @Decorators.timeit    
    def __compute_weighted_averages(self, 
                                    title, 
                                    update, 
                                    maxval, 
                                    decimals):
        """
        Computes weights for every single missing value.
        For categorical variables: 
            Weighted average = (feature value of other samples * proximity value) 
                                 / all proximities values.
        For numerical variables:
            Weighted frequency = (modality proportion * its proximity value) 
                                / all proximities values.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable
        decimals : int
           

        Returns
        -------
        None.

        """     
        for iterator, missing_sample in enumerate(self.__missing_values_coordinates):
            
            #'nan sample number': row that has a missing value.
            #'nan feature name': name of the feature currently selected.
            nan_sample = missing_sample[0]
            nan_feature_name = missing_sample[1]
            target_type = (self.__features_type_predictions
                          .loc[nan_feature_name]
                          .any())
            if target_type == const.NUMERICAL:
                #For every sample with a missing value, we get the proximities
                #We strip the proximity value of the selected sample 
                proximity_vector = self.__proximity_matrix[nan_sample]
                prox_values_of_other_samples = np.delete(proximity_vector, 
                                                         nan_sample)
                #We compute the weight
                prox_values_sum = np.sum(prox_values_of_other_samples)
                weight_vector = prox_values_of_other_samples / prox_values_sum
                                  
                #We get all feature's values for every other sample
                other_samples_check = self.__features.index != nan_sample
                coords = (other_samples_check, nan_feature_name)
                other_features_value = self.__features.loc[coords].values
                
                #Dot product between each feature's value and its weight     
                weighted_average = np.dot(other_features_value, 
                                          weight_vector)
                
                #Round float number if it is required.
                rounded_value = np.around(weighted_average, 
                                          decimals=decimals)
                weighted_average = (int(weighted_average) if not decimals else
                                    rounded_value)
                                     
                #We save each weighted average for each missing value
                self.__all_weighted_averages[missing_sample].append(weighted_average)
            else:
                frequencies_per_modality = (self.__features[nan_feature_name]
                                            .value_counts())
                proportion_per_modality = (frequencies_per_modality /
                                           np.sum(frequencies_per_modality))
                for modality in proportion_per_modality.index.values:
                    #We get all the samples containing the modality.
                    checklist = self.__features[nan_feature_name]==modality
                    samples = (self.__features[nan_feature_name]
                               .index[checklist]
                               .values)
                    
                    #Excluding selected sample
                    if nan_sample in samples:
                        samples_check = np.where(samples == nan_sample)
                        samples = np.delete(samples, samples_check)
                    
                    #Proximity per modality
                    prox_values = self.__proximity_matrix[samples, nan_sample]
                    
                    #We get all other proximities
                    all_prox_values = self.__proximity_matrix[:, nan_sample]
                    
                    #We compute the weight
                    weight = np.sum(prox_values)/np.sum(all_prox_values)
                    
                    #Weighted frequency
                    weighted_freq = proportion_per_modality[modality] * weight
                    proportion_per_modality[modality] = weighted_freq
                    
                #We get the modality that has the biggest weighted frequency.
                optimal_weight = proportion_per_modality.idxmax()
                                                
                #We put every weighted frequency in the group.
                self.__all_weighted_averages[missing_sample].append(optimal_weight) 
            update(iterator*(maxval/len(self.__missing_values_coordinates)))


    def __compute_standard_deviations(self):
        """
        Computes the standard deviation of the last n substitutes for the 
        features.

        Returns
        -------
        None.

        """
        for coord, substitute  in self.__all_weighted_averages.items():
            last_n_substitutes = substitute[-self.__last_n_iterations:]
            try:
                #Standard deviation for last n numerical values for every nan
                self.__standard_deviations[coord] = np.std(last_n_substitutes)
            except TypeError:
                #Checking whether last_n_substitutes are the same (0) or not (1).
                if len(set(last_n_substitutes))==1:
                    self.__standard_deviations[coord] = 0 
                else:
                    self.__standard_deviations[coord] = 1
            
         
    @Decorators.timeit                    
    def __replace_missing_values_in_features_frame(self, title, update, maxval):
        """
        Replaces nan with new values in 'self.__encoded_features'.

        Parameters
        ----------
        title : str
            Progress bar variable
        update : function
            Progress bar variable
        maxval : int
            Progress bar variable

        Returns
        -------
        None.

        """
        weighted_averages_iter = enumerate(self.__all_weighted_averages.items())
        for iterator, weighted_averages in weighted_averages_iter:
            missing_value_coordinates, substitute = weighted_averages
            #Getting the coordinates.
            last_substitute = substitute[-1]
            #Replacing values in the features dataframe.
            self.__features.loc[missing_value_coordinates] = last_substitute       
            update(iterator*(maxval/len(self.__all_weighted_averages)))

  
    def __replace_missing_values_in_target_variable(self):
        """
        Replaces nan values in the target values if they exist:
            - We check at the end of training if the values have converged
            - If they don't, we replace them with the mode or the median

        Returns
        -------
        None.

        """
        for index, predicted_values in self.__nan_target_variable_preds.items():
            self.__target_variable.loc[index] = predicted_values[-1]
            self.__predicted_target_value[index] = predicted_values[-1]
                            

    def __fill_with_nan(self):
        """
        Replaces every divergent value with nan

        Returns
        -------
        None.

        """
        for coordinates in self.__all_weighted_averages.keys():
            self.__features.loc[coordinates] = np.nan
 
        
    def __check_and_remove_convergent_values(self):
        """
        Checks if a given value has converged. If that's the case, the value is 
        removed from the list 'self.__missing_values_coordinates'.

        Returns
        -------
        None.

        """
        missing_value_coordinates = list(self.__standard_deviations.keys())         
        for coordinates in missing_value_coordinates:
            nan_feature_name = coordinates[1]
            standard_deviation = self.__standard_deviations[coordinates]
            feature_type = (self.__features_type_predictions
                            .loc[nan_feature_name]
                            .any())
            if (feature_type==const.NUMERICAL and 0<=standard_deviation<=1)\
            or (feature_type==const.CATEGORICAL and not standard_deviation):
                converged_value = self.__all_weighted_averages[coordinates][-1] 
                self.__converged_values[coordinates] = converged_value
                self.__all_weighted_averages_copy[coordinates] = self.__all_weighted_averages[coordinates]
                #Removing nan values that converged          
                self.__missing_values_coordinates.remove(coordinates)
                self.__all_weighted_averages.pop(coordinates)
                self.__standard_deviations.pop(coordinates)
                
                
    def __check_for_final_convergence(self):
        """
         Checks if all values have converged. If it is the case, training 
        stops. Otherwise it will continue as long as there are improvements. If
        there are no improvements, the resiliency factor will kick in and try 
        for n(training_resilience) more set of iterations. If it happens that 
        some values converged, training will continue. Otherwise, it will stop.

        Returns
        -------
        None.

        """
        #Checking the remaing values and those that converged
        total_nan_values = self.__number_of_nan_values
        nan_values_remaining = len(self.__missing_values_coordinates)
        nan_values_converged = total_nan_values - nan_values_remaining
        text =(f"\n\n- {nan_values_converged} VALUE(S) CONVERGED!\n" 
               f"- {nan_values_remaining} VALUE(S) REMAINING!")
        print(text)
        
        #Checking if there are still values that didn't converge: 
        self.__nan_values_remaining_check.append(nan_values_remaining)
        if (len(set(self.__nan_values_remaining_check))==1 and 
            len(self.__nan_values_remaining_check)==self.__training_resilience):   
            self.__has_converged = True   
            self.__fill_with_nan()
            self.__make_initial_guesses(title="")
            text = (f"- {nan_values_remaining}/{total_nan_values} VALUES UNABLE" 
                    " TO CONVERGE. THE MEDIAN AND/OR THE MODE HAVE BEEN USED AS" 
                    " A REPLACEMENT")
            print(text)              
        elif not self.__missing_values_coordinates:
            self.__has_converged = True
            print("\n- ALL VALUES CONVERGED!") 
        else: 
            text = ("- NOT EVERY VALUE CONVERGED."
                    " ONTO THE NEXT ROUND OF ITERATIONS...\n")
            print(text)
            
                                                 
    def __save_new_dataset(self, final_dataset, path_to_save_dataset):
        """
        Parameters
        ----------
        final_dataset : pandas.core.frame.DataFrame
         
        path_to_save_dataset : str

        Returns
        -------
        None.

        """
        if path_to_save_dataset:
            final_dataset.to_csv(path_or_buf=path_to_save_dataset, index=False)
            print(f"\n- NEW DATASET SAVED in: {path_to_save_dataset}")


    def __reinitialize_key_vars(self):
        """
        Reinitializing vars if (decimals, sample_size, n_quantiles,
        path_to_save_dataset) is modified.

        Returns
        -------
        None.

        """
        self.__has_converged = False
        self.__original_data = self.__original_data_backup.copy(deep=True) 
        self.__missing_values_coordinates = []
        self.__all_weighted_averages = defaultdict(list)
        self.__standard_deviations = defaultdict()
    
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
            The default is None.

        Returns
        -------
        final_dataset : pandas.core.frame.DataFrame

        """
        #Initializing training
        total_iterations = 0
        self.__reinitialize_key_vars()
        self.__data_sampling(title="[DATA SAMPLING]: ", 
                             sample_size=sample_size, 
                             n_quantiles=n_quantiles)
        self.__check_variables_name_validity()
        self.__isolate_samples_with_no_target_value(title="[ISOLATING SAMPLES"\
                                                    "WITH NO TARGET VALUE]: ")
        self.__separate_features_and_target_variable()  
        self.__predict_feature_type(title="[PREDICTING FEATURE TYPE]: ")
        self.__predict_target_variable_type(title="[PREDICTING TARGET VARIABLE"\
                                            "TYPE]: ") 
        self.__retrieve_nan_coordinates(title="[RETRIEVING NAN COORDINATES]: ")
        self.__make_initial_guesses(title="[MAKING INITIAL GUESSES]: ")
        self.__encode_target_variable()
        self.__retrieve_target_variable_class_mappings()
        
        while not self.__has_converged:
            for iteration in range(1, self.__last_n_iterations + 1):
                total_iterations += 1
                self.__encode_features()
                #1- MODEL BULDING
                text = (f"[{iteration}/{total_iterations}-"
                        "BUILDING RANDOM FOREST]: ")
                self.__build_ensemble_model(title=text) 
        
                #2- FITTING AND EVALUATING THE MODEL
                text = (f"[{iteration}-FITTING AND EVALUATING MODEL]: ")
                self.__fit_and_evaluate_ensemble_model(title=text)
                
                #3- BUILDING PROXIMITY MATRIX
                text=(f"[{iteration}-BUILDING PROXIMITY MATRIX TREES/OOB "
                      f"{self.__estimator.n_estimators}"
                      f"/{self.__best_oob_score}]: ")
                self.__proximity_matrix = self.build_proximity_matrix(title=text, 
                                                                      ensemble_estimator=self.__estimator, 
                                                                      encoded_features=self.__encoded_features_pred)
                self.__retrieve_combined_predictions()  
                #4- COMPUTING WEIGHTED AVERAGES
                text = f"[{iteration}-COMPUTING WEIGHTED AVERAGES]: "
                self.__compute_weighted_averages(title=text, 
                                                 decimals=decimals)
                        
                #5- REPLACING NAN VALUES IN ENCODED DATA 
                text = f"[{iteration}-REPLACING MISSING VALUES]: "
                self.__replace_missing_values_in_features_frame(title=text)
            self.__compute_standard_deviations()
            self.__check_and_remove_convergent_values()
            self.__check_for_final_convergence()
        print(f"\n- TOTAL ITERATIONS: {total_iterations}")
        self.__replace_missing_values_in_target_variable()
        #We save the final dataset if a path is given
        all_data = (self.__features, self.__target_variable)
        final_dataset = pd.concat(all_data, axis=1)  
        final_dataset = self.__reconstruct_original_data(final_dataset, 
                                                         sample_size)
        self.__save_new_dataset(final_dataset, path_to_save_dataset)
        return  final_dataset 
    
    
    def __numerical_categorical_plots(self, 
                                      predicted_values, 
                                      variable_type_prediction, 
                                      coordinates, 
                                      iterations, 
                                      std, 
                                      path, 
                                      filename, 
                                      std_str):
        """
        Creates plot for numerical and categorical values.

        Parameters
        ----------
        predicted_values : str/int
            
        variable_type_prediction : str
            
        coordinates : tuple
            
        iterations : int
            
        std : str
            
        path : str
            
        filename : str
            
        std_str : str
            
        Returns
        -------
        None.

        """
        if not os.path.exists(path):
            os.makedirs(path)
        if variable_type_prediction==const.NUMERICAL:
            plt.ioff()
            plt.figure()
            title_text = (f"Evolution of value {coordinates} over {iterations}" 
                          " iterations\nstd on the last {self.__last_n_iterations}" 
                          " iterations:{std}")
            plt.title(title_text)
            plt.plot(np.arange(1,iterations+1), predicted_values)
            plt.xlabel('Iterations')
            plt.ylabel('Values')
            plt.savefig(os.path.join(path, filename+"_"+std_str+".png"))
            plt.close()
        else:
            plt.ioff()
            plt.figure()
            data = Counter(predicted_values)
            names = list(data.keys())
            values = list(data.values())
            percentages = list(map(int, (values/np.sum(values))*100))
            for i in range(len(percentages)):
                plt.annotate(s=percentages[i], 
                             xy=(names[i], percentages[i]+1), 
                             fontsize=10)
                plt.hlines(percentages[i], xmin=0, xmax=0)
            plt.bar(names, percentages, align="center")
            plt.ylabel('Proportion')
            title_text = (f"Proportions of value {coordinates} modalities after"
                          " {iterations} iterations")
            plt.title(title_text)
            plt.savefig(os.path.join(path, filename+".png"))
            plt.close()
                
                
    def create_weighted_averages_plots(self, directory_path, both_graphs=0):
        """
        Creates plots of nan predicted values evolution over n iterations.
        Two type of plots can be generated: for values that diverged and those 
        that converged.
 
        Parameters
        ----------
        directory_path : str
             'directory_path' is set to specify the path for the graphs to be 
             stored into  
        both_graphs : int, optional
            The default is 0. If 'both_graphs' is set to 1, those two type of 
            graph will be generated.

        Returns
        -------
        None.

        """
        graph_choice = (self.__all_weighted_averages, "divergent_graphs")
        convergent_and_divergent = [graph_choice]
        if both_graphs:
            graph_choice = (self.__all_weighted_averages_copy, 
                            "convergent_graphs")
            convergent_and_divergent.append(graph_choice)
        for value in convergent_and_divergent:
            weighted_average_dict = value[0]
            graph_type = value[1]
            std = 0
            std_str = ""
            for coordinates, values in weighted_average_dict.items():
                print(f"-{coordinates} graph created")                       
                try:
                    std = np.std(values[-self.__last_n_iterations:])
                    std = np.round(std, 2)
                    std_str = f"std_{std}"
                except TypeError:
                    pass
                row_number = coordinates[0] 
                variable_name = coordinates[1]
                filename = f"row_{row_number}_column_{variable_name}" 
                iterations = len(values)
                path = os.path.join(directory_path, graph_type, variable_name)
                var_type = self.__features_type_predictions.loc[variable_name].any()
                self.__numerical_categorical_plots(values, 
                                                   var_type,
                                                   coordinates,
                                                   iterations,
                                                   std, 
                                                   path, 
                                                   filename, 
                                                   std_str)
                
                
    def create_target_pred_plot(self, directory_path):
        """
        Creates plots to evaluate missing target values predictions evolution.

        Parameters
        ----------
        directory_path : str
       
        Returns
        -------
        None.

        """
        for index, predicted_values in self.__nan_target_variable_preds.items():
            std = None
            std_str = ""
            filename = f"sample_{index}" 
            iterations = len(predicted_values)
            path = os.path.join(directory_path, "target_values_graphs")
            print(f"graph for sample {index} created")
            try:
                std = np.std(predicted_values[-self.__last_n_iterations:])
                std = np.round(std, 2)
                std_str = f"std_{std}"
            except TypeError:
                pass
            var_type = self.__target_variable_type_prediction["Predictions"].any()
            self.__numerical_categorical_plots(predicted_values, 
                                               var_type, 
                                               index, 
                                               iterations, 
                                               std,
                                               path, 
                                               filename, 
                                               std_str)