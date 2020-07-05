
"""
******************************************************************************
******************************************************************************
******************************************************************************
                        This module contains 3 mixins:
                            - DataPreprocessingMixin
                            - ModelMixin
                            - PlotMixin
******************************************************************************
******************************************************************************
******************************************************************************
"""
from collections import defaultdict, deque, Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from MissingValuesHandler.data_type_identifier import DataTypeIdentifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats as ss
from sklearn import manifold
from copy import copy
import MissingValuesHandler.custom_exceptions as customs
import MissingValuesHandler.constants as const 
import matplotlib.pyplot as plt 
import progressbar as pb
import numpy as np
import pandas as pd
import os

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


"""
##############################################################################
##############################################################################
######################   DataPreprocessingMixin ##############################
##############################################################################
##############################################################################
"""
class DataPreprocessingMixin():
    def __init__(self, 
                 data, 
                 target_variable_name, 
                 ordinal_features_list, 
                 forbidden_features_list):
        """
        Constructor
        
        Parameters
        ----------
        data : pandas.core.frame.DataFrame
        
        target_variable_name : str
        
        ordinal_features_list : list, optional
            The default is None
        forbidden_features_list : list, optional
            The default is None
        
        Returns
        -------
        None
        """
        if ordinal_features_list is None:
            self._ordinal_vars = []
        else:
            self._ordinal_vars = ordinal_features_list
        if forbidden_features_list is None:
            self._forbidden_features = []
        else:
            self._forbidden_features = forbidden_features_list
          
        #Data type identifier object
        self._data_type_identifier = DataTypeIdentifier()
    
    
        #Main variables
        self._original_data = None
        self._original_data_backup = data.copy(deep=True)
        self._original_data_sampled = pd.DataFrame()
        self._orginal_data_temp = pd.DataFrame()
        self._data_null_index = None
        self._idx_no_target_value = None
    
    
        #Features and target variable: original and encoded
        self._features = None
        self._target_variable = None
        self._features_type_predictions = None
        self._target_var_type_prediction = None
        self._encoded_features_model = None
        self._encoded_features_pred = None
        self._target_var_encoded = None
        self._target_variable_name = target_variable_name 
    
        #Label encoder for features and target variable
        self._label_encoder_features = LabelEncoder()
        self._label_encoder_target_vars = LabelEncoder()


    def get_features_type_predictions(self):
        """ 
        Retrieves all features predictions type whether they are numerical 
        or categorical.
        
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self._features_type_predictions
     
     
    def get_sample(self):
        """
        Retrieves sample on which the ensemble model has been trained on.

        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self._original_data_sampled
        
        
    def get_target_variable_type_prediction(self):
        """
        Retrieves prediction about the type of the target variable whether it 
        is numerical or categorical.
        
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self._target_var_type_prediction


    def get_encoded_features(self):
        """
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self._encoded_features_model
    
    
    def get_target_variable_encoded(self):
        """
        Returns
        -------
        pandas.core.frame.DataFrame
        """
        return self._target_var_encoded
      

    @Decorators.timeit
    def _data_sampling(self, title, update, maxval, sample_size, n_quantiles):
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
        None
        """
        if sample_size:
            data_sampled = None
            null_checklist = self._original_data.isnull().any(axis=1)
            data_null = self._original_data[null_checklist]
            data_no_null = self._original_data.dropna()
            target_no_null = data_no_null[self._target_variable_name]
            if not n_quantiles:
                train_test = train_test_split(data_no_null, 
                                              test_size=sample_size, 
                                              random_state=42, 
                                              stratify=target_no_null)
                self._orginal_data_temp = train_test[0]
                data_sampled = train_test[1]
            else:
                k_bins = KBinsDiscretizer(n_quantiles, "ordinal")
                target_no_null = np.array(target_no_null).reshape((-1, 1))
                y_binned = k_bins.fit_transform(target_no_null)
                train_test = train_test_split(data_no_null, 
                                              test_size=sample_size, 
                                              random_state=42, 
                                              stratify=y_binned) 
                self._orginal_data_temp = train_test[0]
                data_sampled = train_test[1]
            self._original_data = pd.concat([data_sampled, data_null])
            self._original_data = self._original_data.reset_index() 
            self._data_null_index = self._original_data["index"].to_dict()
            self._original_data = self._original_data.drop("index", axis=1)
            self._original_data_sampled = self._original_data.copy(deep=True)
      
        
    def _reconstruct_original_data(self, final_dataset, sample_size):
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
            final_dataset = final_dataset.rename(self._data_null_index)
            final_dataset = pd.concat([self._orginal_data_temp, final_dataset])
            final_dataset.sort_index(inplace=True)
        return final_dataset
    
            
    def _check_variables_name_validity(self):
        """
        1- Verifies whether variables in 'forbidden_features_list' 
            or 'ordinal_features_list' exist in the dataset. 
        2- Verifies whether one variable is not mentioned twice in both lists.

        Raises
        ------
         customs.VariableNameError

        Returns
        -------
        None
        """
        columns_names_set = set(self._original_data.columns.tolist())
        forbidden_variables_set = set(self._forbidden_features)
        ordinal_variables_set = set(self._ordinal_vars)
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
                    " ordinal_features_list not present in the dataset!")
            raise customs.VariableNameError(text)
        #2               
        duplicates_check = np.in1d(self._ordinal_vars, self._forbidden_features)
        if duplicates_check:
            duplicates_names = np.where(duplicates_check)[0]
            forbidden_features_list = np.array(self._ordinal_vars)
            text = (f"Variable(s) {forbidden_features_list[duplicates_names]}"
                    " in forbidden_features_list can't be duplicated in"
                    " ordinal_features_list")
            raise customs.VariableNameError(text)
    
    
    @Decorators.timeit
    def _isolate_samples_with_no_target_value(self, title, update, maxval):
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
        None
        """
        diff_idx = None
        try:
            target_variable = self._original_data[self._target_variable_name]
            nan_target_check = target_variable.index[target_variable.isnull()]
            nan_target = self._original_data.loc[nan_target_check]
            features_check = nan_target.columns != self._target_variable_name
            features = nan_target.loc[: , features_check]  
            nan_features = features.isnull().any(axis=1)
            nan_idx = nan_features.loc[nan_features].index
            samples_nan_target_value = set(nan_idx)
            nan_samples_target_value = set(nan_target_check)
            diff_idx = nan_samples_target_value.difference(samples_nan_target_value)
            self._idx_no_target_value = list(nan_idx)
        except KeyError:
            text = (f"Target variable '{self._target_variable_name}'"
                    " does not exist!")
            raise customs.TargetVariableNameError(text)
        else:
            if diff_idx:
                text = (f"No target value in sample(s) {diff_idx} but no"
                        " missing values in feature(s) as well. Remove"
                        f" {diff_idx} from this set: that can be"
                        " predicted with another ML algorithm")
                raise customs.TrainingSetError(text)


    def _separate_features_and_target_variable(self):
        """
        Returns
        -------
        None
        """
        self._features = (self._original_data
                          .drop(self._target_variable_name, axis=1))   
        self._target_variable = (self._original_data 
                                 .loc[:, self._target_variable_name] 
                                 .copy(deep=True))
  
    
    @Decorators.timeit
    def _predict_feature_type(self, title, update, maxval):
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
        None
        """
        self._features_type_predictions = (self._data_type_identifier
                                           .predict(self._features, 0))
     
        
    @Decorators.timeit    
    def _predict_target_variable_type(self, title, update, maxval):
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
        None
        """
        target_variable = self._target_variable.to_frame()
        self._target_var_type_prediction = (self._data_type_identifier
                                            .predict(target_variable, 0))
   

    @Decorators.timeit
    def _retrieve_nan_coordinates(self, title, update, maxval):
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
        None
        """
        #Checking if there are any missing values in the dataset.
        if not self._features.isnull().values.any():
            text = "No missing values were found in the dataset!"
            raise customs.NoMissingValuesError(text)
        
        features_nan_check = self._features.isnull().any()
        features_nan_name = self._features.columns[features_nan_check]
        features_nan = self._features[features_nan_name]
          
        for iterator, feature_nan in enumerate(features_nan):
            empty_cells_checklist = self._features[feature_nan].isnull()
            row_coordinates = (self._features[feature_nan]
                              .index[empty_cells_checklist])
            column_coordinate = feature_nan
            col_row_combinations = [column_coordinate]*len(row_coordinates)
            nan_coordinates = list(zip(row_coordinates, col_row_combinations))
            self._missing_values_coordinates.extend(nan_coordinates)
            update(iterator*(maxval/len(features_nan)))
                      
        #Getting the total number of missing values for future purposes.
        self._number_of_nan_values = len(self._missing_values_coordinates)
    
    
    @Decorators.timeit       
    def _make_initial_guesses(self, title, update, maxval):
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
        None
        """
        #Getting features that contains null values only  
        null_columns_checklist = self._features.isnull().any() 
        null_columns_names = (self._features
                             .columns[null_columns_checklist])

        #Getting variable type predictions for features containing null values 
        nan_predictions = (self._features_type_predictions
                          .loc[null_columns_names, "Predictions"])

        #Getting numerical and categorical features' names
        num_check = nan_predictions==const.NUMERICAL
        cat_check = nan_predictions==const.CATEGORICAL
        numerical_variables_names = nan_predictions[num_check].index
        categorical_variables_names = nan_predictions[cat_check].index

        #Calculating medians and modes
        medians = self._features[numerical_variables_names].median()
        modes = self._features[categorical_variables_names].mode().iloc[0]
        initial_guesses = pd.concat([medians, modes])

        #Replacing initial_guesses in the dataset
        self._features.fillna(initial_guesses, inplace=True)
        
            
    def _encode_features(self):
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
        None
        """
        predictions = self._features_type_predictions["Predictions"] 
    
        #Checklists to highlight categorical and numerical variables only.
        categorical_var_check = list(predictions==const.CATEGORICAL)
        numerical_var_check = list(predictions==const.NUMERICAL)
        
        #Getting variables' name that are either categorical or numerical. 
        categorical_vars_names = (predictions.index[categorical_var_check]
                                  .to_list())
        numerical_vars_names = (predictions.index[numerical_var_check]
                               .to_list())
            
        #Retrieving all numerical and categorical variables
        numerical_vars = self._features[numerical_vars_names]   
        categorical_vars = self._features[categorical_vars_names]
        
        #Separating nominal and ordinal categorical variables
        if self._ordinal_vars:
            nominal_cat_vars = (categorical_vars
                                .drop(self._ordinal_vars, axis=1))
            ordinal_cat_vars = (categorical_vars
                                .loc[:, self._ordinal_vars])
        
            #Label encoding ordinal categorical variables 
            transform = self._label_encoder_features.fit_transform
            encoded_ordinal_cat_vars = ordinal_cat_vars.apply(transform)
            
            #One-Hot encoding nominal categorical variables 
            # 'a_c' stands for authorized columns
            a_c = [column_name for column_name in nominal_cat_vars.columns if 
                    column_name not in self._forbidden_features] 
            encoded_nominal_cat_vars = pd.get_dummies(nominal_cat_vars, columns=a_c)
            
            #Gathering numericals variables and encoded categorical ones
            all_encoded_data  = (numerical_vars, 
                                encoded_ordinal_cat_vars, 
                                encoded_nominal_cat_vars)
            self._encoded_features_model = pd.concat(all_encoded_data, axis=1)    
        elif categorical_vars_names:
            a_c = [column_name for column_name in categorical_vars.columns if 
                    column_name not in self._forbidden_features] 
            encoded_cat_vars = pd.get_dummies(categorical_vars, columns=a_c)
            
            #Gathering numerical variables and nominal categorical variables
            all_encoded_data = (numerical_vars, encoded_cat_vars)
            self._encoded_features_model = pd.concat(all_encoded_data, axis=1)
        else:
            self._encoded_features_model = self._features.copy(deep=True)
   
        '''
        Creating two separates encoded_features sets 
        if self._idx_no_target_value is not empty:
        1- One for the ensemble model that have a missing target value
        2- Another for building the proximity matrix and computing the weighted
        averages
        '''    
        if len(self._idx_no_target_value)!=0:
            self._encoded_features_pred = (self._encoded_features_model
                                            .copy(deep=True))
            self._encoded_features_model.drop(self._idx_no_target_value, 
                                               inplace=True)
        else:
            self._encoded_features_pred = (self._encoded_features_model
                                            .copy(deep=True))
              
                 
    def _encode_target_variable(self):
        """
        Encodes the target variable if it is permitted by the user:
        - If the name of the variable is not in 'forbidden_features_list'
        - If the target variable is numerical it will not be encoded so there's 
            no need to put it in 'forbidden_features_list'.
        The target variable will always be label encoded because the trees in 
        the random forest aren't using it for splitting purposes.

        Returns
        -------
        None
        """
        prediction = self._target_var_type_prediction["Predictions"]
        target_var_cleansed = self._target_variable.copy(deep=True)
        
        #Removal of samples having a missing target_value
        if len(self._idx_no_target_value)!=0:
            target_var_cleansed.drop(self._idx_no_target_value, inplace=True)
        self._target_var_encoded = target_var_cleansed
        
        #We encode it if the variable is categorical
        if (self._target_variable_name not in self._forbidden_features and 
            prediction.any()==const.CATEGORICAL):
            self._target_var_encoded  = (self._label_encoder_target_vars
                                          .fit_transform(target_var_cleansed))
            
            
    def _retrieve_target_variable_class_mappings(self):
        """
        Returns mappings of our target variable modalities if the latter is 
        categorical.

        Returns
        -------
        None
        """
        original_values = None
        try:
            original_values = self._label_encoder_target_vars.classes_ 
            encoded_values = (self._label_encoder_target_vars
                              .transform(original_values)) 
            all_values = zip(original_values, encoded_values)
            for original_value, encoded_value in all_values:
                self._mappings_target_variable[encoded_value] = original_value
        except AttributeError:
            pass


"""
##############################################################################
##############################################################################
##################            ModelMixin      ################################
##############################################################################
##############################################################################
"""
class ModelMixin():
    def __init__(self, training_resilience, n_iterations_for_convergence):
        """
        Constructor 
        
        Parameters
        ----------
        training_resilience : int
        
        n_iterations_for_convergence : int

        Raises
        ------
        customs.TrainingResilienceValueError()
            EXCEPTION RAISED WHEN TRAINING_RESILIENCE LOWER THAN 2

        Returns
        -------
        None
        """
        if training_resilience < 2:
            raise customs.TrainingResilienceValueError()
        #Proximity/distance matrix variables
        self._proximity_matrix = []
        self._distance_matrix = []
        self._divergent_values = defaultdict(list)
        self._all_weighted_averages = defaultdict(list)

        #Weighted averages if sampling is enabled
        self._all_weighted_averages_sample = None
        self._converged_values_sample = None
        self._divergent_values_sample = None
        self._predicted_target_value_sample = None
        self._target_value_predictions_sample = None

        #Features convergence check variables
        self._missing_values_coordinates = []
        self._number_of_nan_values = 0  
        self._std_entropy = defaultdict()
        self._converged_values = defaultdict()
        self._training_resilience = training_resilience
        self._nan_values_remaining_check = deque(maxlen=training_resilience)
        self._last_n_iterations = n_iterations_for_convergence   
        self._has_converged = None
        
        #Target variable predictions(if nan target values exist)
        self._nan_target_variable_preds = defaultdict(list)
        self._predicted_target_value = defaultdict()
        self._mappings_target_variable = defaultdict()
        
        #Random forest(regressor or classifier) variables
        self._estimator = None
        self._n_estimators = None
        self._additional_estimators = None
        self._max_depth = None
        self._min_samples_split = None 
        self._min_samples_leaf = None
        self._min_weight_fraction_leaf = None 
        self._max_features = None
        self._max_leaf_nodes = None
        self._min_impurity_decrease = None
        self._min_impurity_split = None
        self._n_jobs = None 
        self._random_state = None
        self._verbose = None
        self._bootstrap = True
        self._oob_score = True
        self._best_oob_score = 0
        self._warm_start = True
                
     
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
            The default is None
        min_samples_split : int, optional
            The default is 20.
        min_samples_leaf : int, optional
            The default is 20.
        min_weight_fraction_leaf : float, optional
            The default is 0.0.
        max_features : str, optional
            The default is 'auto'.
        max_leaf_nodes : int, optional
            The default is None
        min_impurity_decrease : float, optional
            The default is 0.0.
        min_impurity_split : float, optional
            The default is None
        n_jobs : int, optional
            DESCRIPTION. The default is -1.
        random_state : int, optional
            DESCRIPTION. The default is None
        verbose : int, optional
            The default is 0.

        Returns
        -------
        None
        """
        self._additional_estimators = additional_estimators
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split 
        self._min_samples_leaf = min_samples_leaf
        self._min_weight_fraction_leaf = min_weight_fraction_leaf
        self._max_features = max_features
        self._max_leaf_nodes = max_leaf_nodes
        self._min_impurity_decrease = min_impurity_decrease
        self._min_impurity_split = min_impurity_split
        self._n_jobs = n_jobs 
        self._random_state = random_state
        self._verbose = verbose
 
    
    def get_ensemble_model_parameters(self):
        """
        Retrives random forest regressor or classifier parameters
        
        Returns
        -------
        dict
        """
        return {"n_estimators":self._n_estimators,                
                "additional_estimators":self._additional_estimators,
                "max_depth":self._max_depth,           
                "min_samples_split":self._min_samples_split,                 
                "min_samples_leaf":self._min_samples_leaf,
                "min_weight_fraction_leaf":self._min_weight_fraction_leaf,         
                "max_features":self._max_features,                     
                "max_leaf_nodes":self._max_leaf_nodes,                   
                "min_impurity_decrease":self._min_impurity_decrease,            
                "min_impurity_split":self._min_impurity_split,               
                "n_jobs":self._n_jobs,                           
                "random_state":self._random_state,                     
                "verbose":self._verbose,                          
                "bootstrap":self._bootstrap,                        
                "oob_score":self._oob_score,                        
                "warm_start":self._warm_start}
    
         
        
    def get_ensemble_model(self):
        """
        Random forest model (classifier or regressor)
        
        Returns
        -------
        sklearn.ensemble._forest
        """
        return self._estimator
    
            
    def get_proximity_matrix(self):
        """
        Retrieves the last proximity matrix built with the optimal 
        random forest.
        
        Returns
        -------
        numpy.ndarray
        """
        return self._proximity_matrix
    
    
    def get_distance_matrix(self):
        """
        Retrieves distance matrix which is equals to 1 - proximity matrix.

        Returns
        -------
        numpy.ndarray
        """
        if len(self._distance_matrix) == 0:
            self._distance_matrix = 1-self._proximity_matrix
        return self._distance_matrix
            
    
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
        dict_a_options = {"all":self._all_weighted_averages_sample, 
                          "conv":self._converged_values_sample,
                          "div":self._divergent_values_sample}
        
        dict_b_options = {"all":self._all_weighted_averages,
                          "conv":self._converged_values,
                          "div":self._divergent_values}
        
        if not dict_a_options[option] and self._data_null_index:       
            dict_= {(self._data_null_index[coordinate[0]], coordinate[1]):
                    predicted_value for coordinate, predicted_value 
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
        dict_a_options = {"all":self._nan_target_variable_preds, 
                          "one":self._predicted_target_value}
        
        dict_b_option = {"all": self._target_value_predictions_sample,
                         "one":self._predicted_target_value_sample}
        
        if dict_a_options[option]:
            if not dict_b_option[option] and self._data_null_index:
                dict_ = {(self._data_null_index[coordinate]):predicted_value 
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
  

    @Decorators.timeit         
    def _build_ensemble_model(self, 
                              title, 
                              update, 
                              maxval):
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
        None
        """
        Model = {const.CATEGORICAL:RandomForestClassifier, 
                 const.NUMERICAL:RandomForestRegressor}
        type_ = self._target_var_type_prediction["Predictions"].any()
        self._estimator = Model[type_](n_estimators=self._n_estimators,
                                        max_depth=self._max_depth, 
                                        min_samples_split=self._min_samples_split, 
                                        min_samples_leaf=self._min_samples_leaf, 
                                        min_weight_fraction_leaf=self._min_weight_fraction_leaf, 
                                        max_features=self._max_features, 
                                        max_leaf_nodes=self._max_leaf_nodes, 
                                        min_impurity_decrease=self._min_impurity_decrease, 
                                        min_impurity_split=self._min_impurity_split, 
                                        bootstrap=self._bootstrap, 
                                        oob_score=self._oob_score, 
                                        n_jobs=self._n_jobs, 
                                        random_state=self._random_state, 
                                        verbose=self._verbose,
                                        warm_start=self._warm_start)

           
    @Decorators.timeit 
    def _fit_and_evaluate_ensemble_model(self, 
                                        title,
                                        update, 
                                        maxval):
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
        None

        """
        precedent_out_of_bag_score = 0
        current_out_of_bag_score = 0
        precedent_estimator = None
        while (current_out_of_bag_score > precedent_out_of_bag_score or not 
               current_out_of_bag_score):
            precedent_estimator = copy(self._estimator)
            self._estimator.fit(self._encoded_features_model, 
                                self._target_var_encoded) 
            precedent_out_of_bag_score = current_out_of_bag_score
            current_out_of_bag_score = self._estimator.oob_score_
            self._estimator.n_estimators += self._additional_estimators
            
        #Keeping the configuration of the previous model(i.e the optimal one)
        self._best_oob_score = np.round(precedent_out_of_bag_score, 2)
        self._estimator.n_estimators -= self._additional_estimators
        self._estimator = precedent_estimator


    def _fill_one_modality(self, 
                            predicted_modality, 
                            prediction_dataframe):
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
        matrix_shape = (len(self._encoded_features_pred), 
                        len(self._encoded_features_pred))
        one_modality_matrix = np.zeros(matrix_shape)
        prediction_checklist = prediction_dataframe[0] == predicted_modality
        idx_check = prediction_dataframe.index[prediction_checklist].tolist() 
        idx_check = np.array(idx_check)
        #Using broadcasting to replace null values by 1
        one_modality_matrix[idx_check[:, None], idx_check] = 1
        return one_modality_matrix
        
    
    def _build_prox_matrices(self, 
                              iterator, 
                              update, 
                              maxval, 
                              predictions, 
                              prediction):
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
        if self._target_var_type_prediction.values[0,0] == const.CATEGORICAL:
            array_to_int = np.vectorize(lambda x: np.int(x))
            possible_predictions = array_to_int(possible_predictions)       
        one_modality_matrix = [self._fill_one_modality(predicted_modality, 
                                                        prediction) 
                               for predicted_modality in possible_predictions]
        proximity_matrix = sum(one_modality_matrix)
        update(iterator*(maxval/len(predictions)))
        return proximity_matrix

    
    def _pred_to_frame(self, estimator):
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
        return pd.DataFrame(estimator.predict(self._encoded_features_pred))
  
    
    @Decorators.timeit
    def build_proximity_matrix(self, 
                               title, 
                               update, 
                               maxval):
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

        all_estimators_list =  self._estimator.estimators_
        number_of_estimators =  self._estimator.n_estimators  
        predictions = [self._pred_to_frame(estimator) 
                       for estimator in all_estimators_list]
        proximity_matrices = [self._build_prox_matrices(iterator, 
                                                         update, 
                                                         maxval, 
                                                         predictions, 
                                                         prediction) 
                             for iterator, prediction in enumerate(predictions)] 
        final_proximity_matrix = sum(proximity_matrices)/number_of_estimators
        return final_proximity_matrix
     
    
    def _retrieve_combined_predictions(self):
        """
        Predicts new values for the target variable(if there is any nan).

        Returns
        -------
        None

        """
        combined_pred = self._estimator.predict(self._encoded_features_pred)
        for index in self._idx_no_target_value:
            sample_pred = combined_pred[index]
            if self._mappings_target_variable:
                realval = self._mappings_target_variable[sample_pred]
                self._nan_target_variable_preds[index].append(realval)
            else:
                self._nan_target_variable_preds[index].append(sample_pred)
   
             
    @Decorators.timeit    
    def _compute_weighted_averages(self, 
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
        None
        """     
        for iterator, missing_sample in enumerate(self._missing_values_coordinates):     
            #'nan sample number': row that has a missing value.
            #'nan feature name': name of the feature currently selected.
            nan_sample = missing_sample[0]
            nan_feature_name = missing_sample[1]
            target_type = (self._features_type_predictions
                          .loc[nan_feature_name]
                          .any())
            if target_type == const.NUMERICAL:
                #For every sample with a missing value, we get the proximities
                #We strip the proximity value of the selected sample 
                proximity_vector = self._proximity_matrix[nan_sample]
                prox_values_of_other_samples = np.delete(proximity_vector, 
                                                         nan_sample)
                #We compute the weight
                prox_values_sum = np.sum(prox_values_of_other_samples)
                weight_vector = prox_values_of_other_samples / prox_values_sum
                                  
                #We get all feature's values for every other sample
                other_samples_check = self._features.index != nan_sample
                coords = (other_samples_check, nan_feature_name)
                other_features_value = self._features.loc[coords].values
                
                #Dot product between each feature's value and its weight     
                weighted_average = np.dot(other_features_value, weight_vector)
                
                #Round float number if it is required.
                rounded_value = np.around(weighted_average, decimals=decimals)
                weighted_average = (int(weighted_average) if 
                                    not decimals else rounded_value)
                                     
                #We save each weighted average for each missing value
                self._divergent_values[missing_sample].append(weighted_average)
            else:
                frequencies_per_modality = (self._features[nan_feature_name]
                                            .value_counts())
                proportion_per_modality = (frequencies_per_modality /
                                           np.sum(frequencies_per_modality))
                for modality in proportion_per_modality.index.values:
                    #We get all the samples containing the modality.
                    checklist = self._features[nan_feature_name]==modality
                    samples = (self._features[nan_feature_name]
                               .index[checklist]
                               .values)
                    
                    #Excluding selected sample
                    if nan_sample in samples:
                        samples_check = np.where(samples == nan_sample)
                        samples = np.delete(samples, samples_check)                    
                    #Proximity per modality
                    prox_values = self._proximity_matrix[samples, nan_sample]             
                    #We get all other proximities
                    all_prox_values = self._proximity_matrix[:, nan_sample]                
                    #We compute the weight
                    weight = np.sum(prox_values)/np.sum(all_prox_values)                  
                    #Weighted frequency
                    weighted_freq = proportion_per_modality[modality] * weight
                    proportion_per_modality[modality] = weighted_freq                
                #We get the modality that has the biggest weighted frequency.
                optimal_weight = proportion_per_modality.idxmax()                                             
                #We put every weighted frequency in the group.
                self._divergent_values[missing_sample].append(optimal_weight) 
            update(iterator*(maxval/len(self._missing_values_coordinates)))

    
    def _std_ent(self, option, variable):
        """
        Parameters
        ----------
        option : str
            std: computes standard deviation
            ent: computes entropy
        variable : list

        Returns
        -------
        result : float
        """
        if option == "std":
            result = np.std(variable)
        elif option == "ent": 
            frequencies=Counter(variable)
            prob = [value/len(variable) for _,value in frequencies.items()]
            result = ss.entropy(prob)
        return result 
    
    
    def _compute_std_and_entropy(self):
        """
        Computes the standard deviation or entropy of the last n substitutes 
        for the features.

        Returns
        -------
        None
        """
        for coord, substitute  in self._divergent_values.items():
            last_n_substitutes = substitute[-self._last_n_iterations:]
            try:
                #Standard deviation for last n numerical values for every nan
                self._std_entropy[coord] = self._std_ent("std",
                                                           last_n_substitutes)
            except TypeError:
                self._std_entropy[coord] = self._std_ent("ent", 
                                                           last_n_substitutes)
            
         
    @Decorators.timeit                    
    def _replace_missing_values_in_features_frame(self, title, update, maxval):
        """
        Replaces nan with new values in 'self._encoded_features'.

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
        None
        """
        weighted_averages_iter = enumerate(self._divergent_values.items())
        for iterator, weighted_averages in weighted_averages_iter:
            missing_value_coordinates, substitute = weighted_averages 
            #Getting the coordinates.
            last_substitute = substitute[-1]
            
            #Replacing values in the features dataframe.
            self._features.loc[missing_value_coordinates] = last_substitute       
            update(iterator*(maxval/len(self._divergent_values)))

  
    def _replace_missing_values_in_target_variable(self):
        """
        Replaces nan values in the target values if they exist:
            - We check at the end of training if the values have converged
            - If they don't, we replace them with the mode or the median

        Returns
        -------
        None
        """
        for index, predicted_values in self._nan_target_variable_preds.items():
            self._target_variable.loc[index] = predicted_values[-1]
            self._predicted_target_value[index] = predicted_values[-1]
                            

    def _fill_with_nan(self):
        """
        Replaces every divergent value with nan

        Returns
        -------
        None

        """
        for coordinates in self._divergent_values.keys():
            self._features.loc[coordinates] = np.nan
 
        
    def _check_and_remove_convergent_values(self):
        """
        Checks if a given value has converged. If that's the case, the value is 
        removed from the list 'self._missing_values_coordinates'.

        Returns
        -------
        None

        """
        missing_value_coordinates = list(self._std_entropy.keys())         
        for coordinates in missing_value_coordinates:
            nan_feature_name = coordinates[1]
            standard_deviation = self._std_entropy[coordinates]
            feature_type = (self._features_type_predictions
                            .loc[nan_feature_name]
                            .any())
            if (feature_type==const.NUMERICAL and 0<=standard_deviation<=1)\
            or (feature_type==const.CATEGORICAL and not standard_deviation):
                converged_value = self._divergent_values[coordinates][-1] 
                self._converged_values[coordinates] = converged_value
                self._all_weighted_averages[coordinates] = self._divergent_values[coordinates]
                #Removing nan values that converged          
                self._missing_values_coordinates.remove(coordinates)
                self._divergent_values.pop(coordinates)
                self._std_entropy.pop(coordinates)
                
                
    def _check_for_final_convergence(self):
        """
         Checks if all values have converged. If it is the case, training 
        stops. Otherwise it will continue as long as there are improvements. If
        there are no improvements, the resiliency factor will kick in and try 
        for n(training_resilience) more set of iterations. If it happens that 
        some values converged, training will continue. Otherwise, it will stop.

        Returns
        -------
        None
        """
        #Checking the remaing values and those that converged
        total_nan_values = self._number_of_nan_values
        nan_values_remaining = len(self._missing_values_coordinates)
        nan_values_converged = total_nan_values - nan_values_remaining
        text =(f"\n\n- {nan_values_converged} VALUE(S) CONVERGED!\n" 
               f"- {nan_values_remaining} VALUE(S) REMAINING!")
        print(text)
        
        #Checking if there are still values that didn't converge: 
        self._nan_values_remaining_check.append(nan_values_remaining)
        if (len(set(self._nan_values_remaining_check))==1 and 
            len(self._nan_values_remaining_check)==self._training_resilience):   
            self._has_converged = True   
            self._fill_with_nan()
            self._make_initial_guesses(title="")
            text = (f"- {nan_values_remaining}/{total_nan_values} VALUES UNABLE" 
                    " TO CONVERGE. THE MEDIAN AND/OR THE MODE HAVE BEEN USED AS" 
                    " A REPLACEMENT")
            print(text)              
        elif not self._missing_values_coordinates:
            self._has_converged = True
            print("\n- ALL VALUES CONVERGED!") 
        else: 
            text = ("- NOT EVERY VALUE CONVERGED."
                    " ONTO THE NEXT ROUND OF ITERATIONS...\n")
            print(text)
            
                                                 
    def _save_new_dataset(self, final_dataset, path_to_save_dataset):
        """
        Parameters
        ----------
        final_dataset : pandas.core.frame.DataFrame
         
        path_to_save_dataset : str

        Returns
        -------
        None

        """
        if path_to_save_dataset:
            final_dataset.to_csv(path_or_buf=path_to_save_dataset, index=False)
            print(f"\n- NEW DATASET SAVED in: {path_to_save_dataset}")


    def _reinitialize_key_vars(self):
        """
        Reinitializing vars if (decimals, sample_size, n_quantiles,
        path_to_save_dataset) is modified.

        Returns
        -------
        None

        """
        self._has_converged = False
        self._original_data = self._original_data_backup.copy(deep=True) 
        self._missing_values_coordinates = []
        self._divergent_values = defaultdict(list)
        self._all_weighted_averages = defaultdict(list) 
        self._std_entropy = defaultdict()
    
    

"""
##############################################################################
##############################################################################
##################               PlotMixin       #############################
##############################################################################
##############################################################################
"""  
class PlotMixin():
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
            The default is None

        Returns
        -------
        None
        """
        plot_type = plot_type.lower().strip()
        filename = ""
        if plot_type == "2d":
            plt.scatter(coordinates[:,0], coordinates[:,1])
            plt.title("2D MDS PLOT")
            plt.xlabel("MDS1")
            plt.ylabel("MDS2") 
            plt.show()
            filename = "2d_mds_plot"+const.IMG_EXTENSION
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
            filename = "3d_mds_plot"+const.IMG_EXTENSION
        if path_to_save:  
            plt.savefig(os.path.join(path_to_save, filename))
      

    def _numerical_categorical_plots(self, 
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
        None
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if variable_type_prediction==const.NUMERICAL:
            plt.ioff()
            plt.figure()
            title_text = (f"Evolution of value {coordinates} over {iterations}" 
                          f" iterations\nstd on the last {self._last_n_iterations}" 
                          f" iterations:{std}")
            plt.title(title_text)
            plt.plot(np.arange(1,iterations+1), predicted_values)
            plt.xlabel('Iterations')
            plt.ylabel('Values')
            plt.savefig(os.path.join(path, filename+"_"+std_str+const.IMG_EXTENSION))
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
                          f" {iterations} iterations")
            plt.title(title_text)
            plt.savefig(os.path.join(path, filename+const.IMG_EXTENSION))
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
        None
        """
        convergent_and_divergent = [(self._divergent_values, "divergent_graphs")]
        if both_graphs:
            convergent_and_divergent.append((self._all_weighted_averages, 
                                             "convergent_graphs"))
        for value in convergent_and_divergent:
            weighted_average_dict = value[0]
            graph_type = value[1]
            std = 0
            std_str = ""
            for coordinates, values in weighted_average_dict.items():
                print(f"-{coordinates} graph created")                       
                try:
                    std = np.std(values[-self._last_n_iterations:])
                    std = np.round(std, 2)
                    std_str = f"std_{std}"
                except TypeError:
                    pass
                row_number = coordinates[0] 
                variable_name = coordinates[1]
                filename = f"row_{row_number}_column_{variable_name}" 
                iterations = len(values)
                path = os.path.join(directory_path, graph_type, variable_name)
                var_type = self._features_type_predictions.loc[variable_name].any()
                self._numerical_categorical_plots(values, 
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
        None
        """
        for index, predicted_values in self._nan_target_variable_preds.items():
            std = None
            std_str = ""
            filename = f"sample_{index}" 
            iterations = len(predicted_values)
            path = os.path.join(directory_path, "target_values_graphs")
            print(f"graph for sample {index} created")
            try:
                std = np.std(predicted_values[-self._last_n_iterations:])
                std = np.round(std, 2)
                std_str = f"std_{std}"
            except TypeError:
                pass
            var_type = self._target_var_type_prediction["Predictions"].any()
            self._numerical_categorical_plots(predicted_values, 
                                               var_type, 
                                               index, 
                                               iterations, 
                                               std,
                                               path, 
                                               filename, 
                                               std_str)