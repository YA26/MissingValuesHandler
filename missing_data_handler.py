# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:46:50 2019

@author: Yann Avok
"""
from custom_exceptions import VariableNameError, TargetVariableNameError, NoMissingValuesError
from joblib import Parallel, delayed
from collections import defaultdict
from colorama import Back, Style
from copy import copy
import numpy as np
import pandas as pd
import multiprocessing


class MissingDataHandler():
    '''
    This class uses a random forest to replace missing values in a dataset for samples that have a target value. It doesn't tackle missing values in a new sample.
    The main idea is to use the random forest's definition of proximity to find the values that are best fit to replace the missing ones.
    
    We have the following parts: 

    1- We get the dataset, separates the features and the target variable and predict their types:
        - private method: __check_variables_name_validity
        - private method: __separate_features_and_target_variable
        - private method: __predict_feature_type
        - private method: __predict_target_variable_type
        
    2- We retrieve the missing values coordinates(row and column) and fill in the nan cells with initial values:
        - private method: __retrieve_nan_coordinates
        - private method: __make_initial_guesses
        
    3- We encode the features and the target variables if they need to be encoded:
        - private method: __encode_features
        - private method: __encode_target_variable
    
    4- We build our model, fit it and evaluate it (we keep the model having the best out of bag score). We then use it to build the proximity matrix:
        - private method: __build_ensemble_model
        - private method: __fit_and_evaluate_ensemble_model
        - private method: __build_proximity_matrix
    
    5- We use the proximity matrix to compute weighted averages for all missing values(both in categorical and numerical variables):
        - private method: __compute_weighted_averages
        - private method:__replace_missing_values_in_encoded_dataframe
        
    6- We check if every value that has been replaced has converged after n iterations. If that's not the case, we go back to step 4 and go for another round of n iterations:
        - private method: __compute_standard_deviations
        - private method: __check_for_convergence
        - private method: __replace_missing_values_in_original_dataframe
        
    7- If all values have converged, we stop everything and save the dataset if a path has been given. 'train' is the main function:
         - private method: __save_new_dataset
         - public  method: train
    '''
    
    def __init__(self,
                 data_type_identifier_object,
                 data_type_identifier_model,
                 mappings,
                 encoder):
        self.__data_type_identifier_object      = data_type_identifier_object
        self.__data_type_identifier_model       = data_type_identifier_model
        self.__mappings                         = mappings
        self.__original_data                    = None 
        self.__features                         = None
        self.__target_variable                  = None
        self.__target_variable_name             = None
        self.__features_type_predictions        = None
        self.__target_variable_type_prediction  = None
        self.__encoded_features                 = None
        self.__target_variable_encoded          = None
        self.__target_variable_mappings         = None
        self.__proximity_matrix                 = None
        self.__distance_matrix                  = None
        self.__number_of_nan_values             = None
        self.__missing_values_coordinates       = []
        self.__encoder                          = encoder
        self.__categorical_features_mappings    = defaultdict()
        self.__standard_deviations              = defaultdict()
        self.__converged_values                 = defaultdict()
        self.__all_weighted_averages            = defaultdict(list)
        self.__all_weighted_averages_copy       = defaultdict(list)
        self.__has_converged                    = False
        
        #Parallelization variables 
        self.__parrallel = Parallel(n_jobs=multiprocessing.cpu_count())
        
        #Random forest variables
        self.__estimator                        = None
        self.__n_estimators                     = None
        self.__criterion                        = None
        self.__max_depth                        = None
        self.__min_samples_split                = None 
        self.__min_samples_leaf                 = None
        self.__min_weight_fraction_leaf         = None 
        self.__max_features                     = None
        self.__max_leaf_nodes                   = None
        self.__min_impurity_decrease            = None
        self.__min_impurity_split               = None
        self.__bootstrap                        = None
        self.__oob_score                        = None
        self.__n_jobs                           = None 
        self.__random_state                     = None
        self.__verbose                          = None
        self.__warm_start                       = None
        
        
    def get_features_type_predictions(self):
        '''
        Retrieves all features predictions type whether they are numerical or categorical.
        'data_type_identifer_model' and 'data_type_identifier_object' are used to predict each type.
        '''
        return self.__features_type_predictions
     
    def get_target_variable_type_prediction(self):
        '''
        Retrieves prediction about the type of the target variable whether it is numerical or categorical.
        'data_type_identifer_model' and 'data_type_identifier_object' are used to predict each type.
        '''
        return self.__target_variable_type_prediction
         
    def get_ensemble_model(self):
        '''
        Used to get a random forest model whether it is a classifier or a regressor
        '''
        return self.__estimator
    
    def get_encoded_features(self):
        return self.__encoded_features
    
    def get_target_variable_encoded(self):
        return self.__target_variable_encoded
         
    def get_categorical_features_mappings(self):
        '''
        Retrieves mappings of all categorical variables if they're not in the 'forbidden_variables_list'(i.e: not already encoded)
        '''
        return self.__categorical_features_mappings
    
    def get_target_variable_mappings(self):
        '''
        Retrieves mappings of the target variable if it is not in 'forbidden_variables_list'(i.e: not already encoded).
        '''
        return self.__target_variable_mappings
        
    def get_proximity_matrix(self):
        '''
        Retrieves the last proximity matrix built with the last random forest(the most optimal one)
        '''
        return self.__proximity_matrix
    
    def get_distance_matrix(self):
        '''
        Retrieves distance matrix which is equals to 1 - proximity matrix.
        '''
        if self.__distance_matrix==None:
            self.__distance_matrix=1-self.__proximity_matrix
        return self.__distance_matrix
     
    def get_all_weighted_averages(self):
        '''
        Retrieves all weighted averages that are used to replace nan values whether those come from categorical or numerical variables.
        '''
        return self.__all_weighted_averages_copy
    
    def get_converged_values(self):
        '''
        Retrieves all nan values and their last calculated values.
        '''
        return self.__converged_values
    
    def __check_variables_name_validity(self, forbidden_variables_list):
        '''
        Verifies whether variables in 'forbidden_variables_list' exist in the dataset. 
        If that's not the case, an exception is raised.
        '''
        for variable_name in forbidden_variables_list:
            if variable_name not in self.__original_data.columns.tolist():
                raise VariableNameError("Variable '{}' in forbidden_variables_list is not present in the dataset!".format(variable_name))
             
    def __separate_features_and_target_variable(self, data, target_variable_name):
        try:    
            self.__features             = data.drop(target_variable_name, axis=1)
            self.__target_variable      = data.loc[:, target_variable_name]
            self.__target_variable_name = target_variable_name
        except KeyError:
            #We raise an exception if the name of the target variable given by the user is not found.
            raise TargetVariableNameError("Target variable '{}' does not exist!".format(target_variable_name))
        
    def __predict_feature_type(self):
        '''
        Predicts if a feature is either categorical or numerical.
        '''
        #We make a copy of the features and the target variable because the method "predict" deletes all nan values.
        features                               = self.__features.copy()
        self.__features_type_predictions       = self.__data_type_identifier_object.predict(features, self.__mappings, self.__data_type_identifier_model)
        
    def __predict_target_variable_type(self):
        '''
        Predicts if the target variable is either categorical or numerical.
        '''
        target_variable                        = self.__target_variable.to_frame()
        self.__target_variable_type_prediction = self.__data_type_identifier_object.predict(target_variable, self.__mappings, self.__data_type_identifier_model) 

    def __retrieve_nan_coordinates(self):
        '''
        Gets the coordinates(row and column) of every empty cell in the features dataset.
        '''
        for feature in self.__features.columns:  
            if self.__features[feature].isnull().values.any():
                #We use the index to get the row coordinate of every empty cell for a given column(feature)  
                empty_cells_checklist = self.__features[feature].isnull()
                row_coordinates       = self.__features[feature].index[empty_cells_checklist]
                column_coordinate     = feature
                for row_coordinate in row_coordinates:
                    self.__missing_values_coordinates.append((row_coordinate, column_coordinate))
                    
        #We check if any nan values was retrieved. If that's not the case, an exception is raised.
        if not self.__missing_values_coordinates:
            raise NoMissingValuesError("No missing values were found in the dataset!")
        
        #We don't forget to get the total number of missing values for future purposes.
        self.__number_of_nan_values = len(self.__missing_values_coordinates)
                      
    def __make_initial_guesses(self):
        '''
        Replaces empty cells with initial values in the features dataset:
            - mode for categorical variables 
            - median for numerical variables
        '''
        for feature in self.__features.columns:
            if self.__features_type_predictions.loc[feature].any()=="numerical":   
                feature_median = self.__features[feature].median()
                self.__features[feature].fillna(feature_median, inplace=True)
            else:
                feature_mode = self.__features[feature].mode().iloc[0]
                self.__features[feature].fillna(feature_mode, inplace=True)
                
    def __encode_features(self, forbidden_variables_list):
        '''
        Encodes every categorical feature the user wants to encode. Any feature mentioned in 'forbidden_variables_list' will not be considered.
        '''
        
        #Creating checklists to highlight categorical and numerical variables only.
        #Example: in 'categorical_variables_checklist', we will get 'True' if a given variable happens to be categorical.
        categorical_variables_checklist = list(self.__features_type_predictions["Predictions"]=="categorical")
        numerical_variables_checklist   = list(self.__features_type_predictions["Predictions"]=="numerical")
        
        #With the right checklist we get the name of the variables that are either categorical or numerical. 
        #The idea here is to separate the two kind of variables and to focus on categorical ones only.
        categorical_variables_names     = self.__features_type_predictions["Predictions"].index[categorical_variables_checklist].to_list()
        numerical_variables_names       = self.__features_type_predictions["Predictions"].index[numerical_variables_checklist].to_list()
            
        #Retrieving all numerical variables.
        numerical_variables             = self.__features[numerical_variables_names]         
        categorical_variables_encoded   = pd.DataFrame()
        
        #Encoding categorical variables and getting their mappings.
        for name in categorical_variables_names:     
            if name not in forbidden_variables_list:
                feature_encoded                             = self.__encoder.fit_transform(self.__features[name])
                categorical_variables_encoded[name]         = feature_encoded
                #In 'feature_string_values' we get the original modalities labels.
                feature_string_values                       = self.__encoder.classes_
                #In 'feature_integer_values' we get the encoded modalities.
                feature_integer_values                      = self.__encoder.transform(feature_string_values)
                #We get the ad hoc mappings
                feature_mappings                            = pd.Series(feature_string_values, index=feature_integer_values)
                self.__categorical_features_mappings[name]  = feature_mappings
            else:
                #We keep the variable as it is and don't encode it if the user requires not to. 
                #Example: Variables that are already encoded will belong in that list.
                categorical_variables_encoded[name]        = self.__features[name]
        #Finally we put all variables together: encoded ones and numerical ones.
        self.__encoded_features = pd.concat((numerical_variables, categorical_variables_encoded), axis=1)
    
    def __encode_target_variable(self, forbidden_variables_list):
        '''
        Encodes the target variable if it is permitted by the user(i.e if the name of the variable is not in 'forbidden_variables_list').  
        If the target variable is numerical it will not be encoded so there's no need to put it in 'forbidden_variables_list'.
        '''
        if self.__target_variable_name not in forbidden_variables_list and self.__target_variable_type_prediction["Predictions"].any()=="categorical":
            self.__target_variable_encoded  = self.__encoder.fit_transform(self.__target_variable)
            target_variable_string_values   = self.__encoder.classes_
            target_variable_integer_values  = self.__encoder.transform(target_variable_string_values)
            self.__target_variable_mappings = pd.Series(target_variable_string_values, index=target_variable_integer_values)
        else:
            #We keep the target variable unchanged if the user requires it or if it is numerical
            self.__target_variable_encoded  = self.__target_variable                 

    def __build_ensemble_model(self, base_estimator):
            '''Builds an ensemble model like a random forest'''
            self.__estimator = base_estimator(n_estimators=self.__n_estimators, 
                                              criterion=self.__criterion, 
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
                                   
    def __fit_and_evaluate_ensemble_model(self, additional_estimators):
            '''
            fits and evaluates the model. 
            1- We compare the out of bag score at time t-1 with the one at time t.
            2- If the latter is lower than the former or equals to it, we stop fitting the model and we keep the one at t-1.
            3- If it's the other way around, we add more estimators to the total number of estimators we currently have.
            '''  
            #Those are kick start values. '10**-5' was chosen for current_out_of_bag_score. But why?
            #Because we want to make sure that no model is bad enough to output a score lower than that.
            precedent_out_of_bag_score  = 0
            current_out_of_bag_score    = 10**-5
            precedent_estimator         = None
            while current_out_of_bag_score > precedent_out_of_bag_score:
                precedent_estimator = copy(self.__estimator)
                self.__estimator.fit(self.__encoded_features, self.__target_variable_encoded) 
                precedent_out_of_bag_score     = current_out_of_bag_score
                current_out_of_bag_score       = self.__estimator.oob_score_
                self.__estimator.n_estimators  += additional_estimators
                print("- Former out of bag score: {}".format(precedent_out_of_bag_score))
                print("- Current out of bag score: {}".format(current_out_of_bag_score))
            #We subtract the additional_estimators because we want to keep the configuration of the previous model(i.e the optimal one)
            self.__estimator.n_estimators   -= additional_estimators
            self.__estimator                = precedent_estimator
            print("\nThe model with score {} has been kept\n".format(precedent_out_of_bag_score))
                    
    def __build_proximity_matrix(estimator, encoded_features):
        '''
            builds a scaled proximity matrix.
            1- We run all the data down the first tree and output predictions.
            2- If two samples fall in the same node (same predictions) we count it as 1.
            3- We do the same for every single tree, sum up the proximity matrix and divide by the total number of estimators.
        '''
        proximity_matrix = np.zeros((len(encoded_features), len(encoded_features)))
        predictions = estimator.predict(encoded_features)

        for row, column in np.ndindex(proximity_matrix.shape):
            #row and column each represents a specific observation.
            #Example: observation number 7 equals row 7 and observation number 5 equals column 5.
            #So at position (7, 5), we will have a number giving an information about how close those two observations are.
            if predictions[row]==predictions[column]:
                proximity_matrix[row, column]=1        
        return proximity_matrix
               
    def __compute_weighted_averages(self, numerical_features_decimals):
        '''
        Computes weights for every single missing value.
        For categorical variables: Weighted average for the sample that has a missing value = (feature's value of every other sample * its proximity value) / all proximities values in the proximity_vector.
        For numerical variables: Weighted average for the sample that has a missing value = (modality proportion * its proximity value) / all proximities values in the proximity_vector.
        '''     
        for missing_sample in self.__missing_values_coordinates:
            #We handle samples that contain missing values one after another. We get the coordinates of the missing sample from the encoded features.
            #'nan sample number' is the row of the sample that has a missing value.
            #'nan feature name' is the name of the feature we are currently working on.
            nan_sample_number   = missing_sample[0]
            nan_feature_name    = missing_sample[1]
            
            if  self.__features_type_predictions.loc[nan_feature_name].any()=="numerical":
                '''
                For every single numerical feature:
                '''
                #For every sample having a missing value, we get all proximities values(with other samples) and put them in the 'proximity_vector'. 
                #We do not forget to strip the proximity value of the sample with itself(because it is always 1). This is the 'prox_values_of_other_samples'
                proximity_vector                = self.__proximity_matrix[nan_sample_number]
                row_to_be_stripped              = nan_sample_number
                prox_values_of_other_samples    = np.delete(proximity_vector, row_to_be_stripped)
                
                #We compute the weight.
                weight_vector = prox_values_of_other_samples / np.sum(prox_values_of_other_samples)
                
                #We get all feature's values for every other sample
                all_other_samples_checklist = self.__encoded_features.index != nan_sample_number
                all_other_feature_values    = self.__encoded_features.loc[all_other_samples_checklist, nan_feature_name].values
                
                #We compute the dot product between each feature's value and its weight
                weighted_average = np.dot(all_other_feature_values, weight_vector)
                
                #If the values were originally integers, we could keep them that way. Otherwise, we can still choose the number of decimals.
                weighted_average = int(weighted_average) if numerical_features_decimals == 0 else np.around(weighted_average, decimals=numerical_features_decimals)
                
                #We save each weighted average for each missing value
                self.__all_weighted_averages[(nan_sample_number, nan_feature_name)].append(weighted_average)
            else:
                '''
                For every single categorical feature:
                '''
                #For categorical variables, we're going to take into account the frequency of every value per feature
                frequencies_per_modality    = self.__encoded_features[nan_feature_name].value_counts()
                proportion_per_modality     = frequencies_per_modality/np.sum(frequencies_per_modality)
                
                #We iterate over the values Ex: 0 and 1 if the categorical variable is binary AND 0,1,2... for a multinomial one
                for modality in proportion_per_modality.index.values:
                    #We get all the samples presenting the modality.
                    checklist   = self.__encoded_features[nan_feature_name]==modality
                    samples     = self.__encoded_features[nan_feature_name].index[checklist].values
                    
                    #We don't want to include the sample we want to predict the value for(i.e sample with missing value)
                    if nan_sample_number in samples:
                        samples = np.delete(samples, np.where(samples == nan_sample_number))
                    
                    #For each modality, we compute the weights
                    prox_values_for_a_modality  = self.__proximity_matrix[samples, nan_sample_number]
                    all_prox_values             = self.__proximity_matrix[:, nan_sample_number]
                    weight                      = np.sum(prox_values_for_a_modality)/np.sum(all_prox_values)
                    
                    #Weighted frequency
                    proportion_per_modality[modality] = proportion_per_modality[modality] * weight
                    
                #We get the modality that has the biggest weighted frequency.
                modality_with_max_weight = int(proportion_per_modality.idxmax())
                                                
                #We put every weighted frequency in the group.
                self.__all_weighted_averages[(nan_sample_number, nan_feature_name)].append(modality_with_max_weight) 
                                
    def __replace_missing_values_in_encoded_dataframe(self):
        '''
        replaces nan with new values in 'self.__encoded_features'
        '''
        for missing_value_coordinates, substitute in self.__all_weighted_averages.items():
            #Getting the coordinates.
            row             = missing_value_coordinates[0]
            column          = missing_value_coordinates[1]
            last_substitute = substitute[-1]
            #Replacing values in the encoded features dataframe.
            self.__encoded_features.loc[row, column] = last_substitute
             
    def __compute_standard_deviations(self, n_iterations_for_convergence):
        '''
        computes the standard deviation of the last n substitutes
        '''
        for missing_value_coordinates, substitute in self.__all_weighted_averages.items():
            last_n_substitutes                                      = substitute[-n_iterations_for_convergence:]
            self.__standard_deviations[missing_value_coordinates]   = np.std(last_n_substitutes)
                
    def __check_for_convergence(self):
        '''
        checks if a given value has converged. If that's the case, the value is removed from the list 'self.__missing_values_coordinates'
        '''
        missing_value_coordinates = list(self.__standard_deviations.keys()) 
        
        for coordinates in missing_value_coordinates:
            nan_feature_name = coordinates[1]
            standard_deviation = self.__standard_deviations[coordinates]
            if (self.__features_type_predictions.loc[nan_feature_name].any()=="numerical" and standard_deviation>=0 and standard_deviation<1)\
            or (self.__features_type_predictions.loc[nan_feature_name].any()=="categorical" and standard_deviation==0):
                converged_value                                 = self.__all_weighted_averages[coordinates][-1] 
                self.__converged_values[coordinates]            = converged_value
                self.__all_weighted_averages_copy[coordinates]  = self.__all_weighted_averages[coordinates]
                #Removing nan values that converged
                self.__missing_values_coordinates.remove(coordinates)
                self.__all_weighted_averages.pop(coordinates)
                self.__standard_deviations.pop(coordinates)
               
        #checking the remaing values and those that converged
        total_nan_values        = self.__number_of_nan_values
        nan_values_remaining    = len(self.__missing_values_coordinates)
        nan_values_converged    = total_nan_values - nan_values_remaining
        print("{}- {} VALUE(S) CONVERGED!\n- {} VALUE(S) REMAINING!{}".format(Back.GREEN, nan_values_converged, nan_values_remaining, Style.RESET_ALL))
                           
        #Setting 'has_converged' flag to True if all values converged.
        if not self.__missing_values_coordinates:
            self.__has_converged = True
        else:
            print("\n-NOT EVERY VALUE CONVERGED. ONTO THE NEXT ROUND OF ITERATIONS...\n")
                              
    def __replace_missing_values_in_original_dataframe(self, forbidden_variables_list):
        '''
        replaces nan with new values in 'self.__original_data'
        '''           
        for missing_value_coordinates, converged_value in self.__converged_values.items():
            #Getting the coordinates.
            row             = missing_value_coordinates[0]
            column          = missing_value_coordinates[1]
            #Replacing values in the encoded features dataframe:
            if self.__features_type_predictions.loc[column].any()=="categorical" and column not in forbidden_variables_list:
                self.__original_data.loc[row, column] = self.__categorical_features_mappings[column][converged_value]
            else:
                self.__original_data.loc[row, column] = converged_value
                     
    def __save_new_dataset(self, path_to_save_dataset):
        if path_to_save_dataset is not None:
            self.__original_data.to_csv(path_or_buf=path_to_save_dataset, index=False)
            print("\n- New dataset saved in: {}".format(path_to_save_dataset))
            
    def train(self, 
              data, 
              target_variable_name, 
              base_estimator,
              numerical_features_decimals=0, 
              n_iterations_for_convergence=3, 
              additional_estimators=5, 
              estimator=None,
              n_estimators=25,
              criterion='gini',
              max_depth=None,
              min_samples_split=20,
              min_samples_leaf=20,
              min_weight_fraction_leaf=0.0, 
              max_features='auto',
              max_leaf_nodes=None,
              min_impurity_decrease=0.0, 
              min_impurity_split=None,
              bootstrap=True,
              oob_score=True,
              n_jobs=-1,
              random_state=None,
              verbose=0,
              warm_start=True,
              path_to_save_dataset=None,
              forbidden_variables_list=[],
              __build_proximity_matrix=__build_proximity_matrix):
        '''
        This is the main function. At run time, every other private functions will be exceuted one after another.
        '''
 
        #Updating Random forest variables
        self.__estimator                = estimator
        self.__n_estimators             = n_estimators
        self.__criterion                = criterion
        self.__max_depth                = max_depth
        self.__min_samples_split        = min_samples_split 
        self.__min_samples_leaf         = min_samples_leaf
        self.__min_weight_fraction_leaf = min_weight_fraction_leaf
        self.__max_features             = max_features
        self.__max_leaf_nodes           = max_leaf_nodes
        self.__min_impurity_decrease    = min_impurity_decrease 
        self.__min_impurity_split       = min_impurity_split
        self.__bootstrap                = bootstrap
        self.__oob_score                = oob_score
        self.__n_jobs                   = n_jobs 
        self.__random_state             = random_state
        self.__verbose                  = verbose
        self.__warm_start               = warm_start
        self.__has_converged            = False
        total_iterations                = 0
        
        print("Getting ready...\nMaking initial guesses\nEncoding the features...")
        self.__original_data = data.copy()
        self.__check_variables_name_validity(forbidden_variables_list)
        self.__separate_features_and_target_variable(data, target_variable_name)  
        self.__predict_feature_type()
        self.__predict_target_variable_type()
        self.__retrieve_nan_coordinates()
        self.__make_initial_guesses()
        self.__encode_features(forbidden_variables_list)
        self.__encode_target_variable(forbidden_variables_list)
        
        #Every value has to converge. Otherwise we will be here for another round of n iterations.
        while self.__has_converged==False:
            for iteration in range(1, n_iterations_for_convergence + 1):
                total_iterations += iteration 
                
                print("\n\n⚽ ⚽️ ⚽️ ⚽️ ⚽️ ITERATION NUMBER :{} ⚽️ ⚽️ ⚽️ ⚽️ ⚽️\n\n".format(iteration))
                
                print("\n1- BUILDING THE MODEL...\n")
                self.__build_ensemble_model(base_estimator)
                print("\nMODEL BUILT...\n")
                
                print("\n2- FITTING AND EVALUATING THE MODEL...\n")
                self.__fit_and_evaluate_ensemble_model(additional_estimators)
                print("MODEL FITTED AND EVALUATED!")
                
                print("\n3- BUILDING PROXIMITY MATRIX...")  
                print("\n{} {} estimators have been counted {}\nEach estimator is being used for predictions...".format(Back.GREEN, self.__estimator.n_estimators, Style.RESET_ALL))
                all_estimators_list     = self.__estimator.estimators_
                number_of_estimators    = self.__estimator.n_estimators
                proximity_matrices      = self.__parrallel(delayed(__build_proximity_matrix)(estimator, self.__encoded_features) for estimator in all_estimators_list)
                sum_proximity_matrices  = sum(proximity_matrices)
                self.__proximity_matrix = sum_proximity_matrices/number_of_estimators
                print("\nPROXIMITY MATRIX BUILT!\n")
                      
                print("\n4- COMPUTING WEIGHT AVERAGES...")
                self.__compute_weighted_averages(numerical_features_decimals)
                print("\nWEIGHT AVERAGES COMPUTED!\n")
                        
                print("\n5- REPLACING NAN VALUES IN ENCODED DATA...")           
                self.__replace_missing_values_in_encoded_dataframe()
                print("\nNAN VALUES REPLACED!\n")
            
            print("\n\n⚾ ⚾ ⚾ ⚾ ⚾ CHECKING FOR CONVERGENCE...⚾ ⚾ ⚾ ⚾ ⚾ ⚾\n\n")
            self.__compute_standard_deviations(n_iterations_for_convergence)
            self.__check_for_convergence()
        print("\nALL VALUES CONVERGED!\n")
        
        print("\n⚾ ⚾ ⚾ ⚾ ⚾ REPLACING NAN VALUES IN ORIGINAL DATA...⚾ ⚾ ⚾ ⚾ ⚾") 
        self.__replace_missing_values_in_original_dataframe(forbidden_variables_list)
        print("\n- ALL VALUES HAVE BEEN REPLACED!")   
        print("\n- TOTAL ITERATIONS: {}".format(total_iterations))
        #We save the dataset if a path is given
        self.__save_new_dataset(path_to_save_dataset)
        return  self.__original_data 
    
