from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from inspect import getsourcefile
from os.path import join, dirname
from numpy import sort, array
import pandas as pd
import pickle

class DataTypeIdentifier(object):
        
    def __init__(self):
        self.__encoder          = LabelEncoder()
        self.__print_verbose    = None
                     
    def get_encoder(self):
        return self.__encoder

    def __print(self, string):
        if self.__print_verbose:
            print(string) 
    
    def __keep_initial_data_types(self, original_data):
        '''
        This function helps us render immuable the data type of every feature(column) that was set before the data was imported.
        The basic idea here is to avoid having integers being transformed into float.
        We are using a special pandas datatype (Int64Dtype) to achieve it.
        '''
        data = original_data.copy(deep=True)
        for column in data.columns:
            try:    
                self.__print(f"- Feature *{column}* treated")
                data=data.astype({column: pd.Int64Dtype()})
            except TypeError:
                pass     
        return data
    
    def __build_final_set(self, data):
        '''
        Function to build training/final set. 
        We create our final features (is_float and unique_values) to help predict if a feature is numerical or categorical 
        We also encode our target variable
        '''
        data.dropna(inplace=True) 
        new_features_list = []
        for feature_name in data:
            feature = data[feature_name]
            #Checking feature data type: we create a variable "is_float" to check if the data type of the feature is float. The default value is "False" or 0
            is_float = 0 
            if feature.dtype == float:
                is_float = 1       
            
            #Computing steps between one modality and the one following it: if steps are consstant and equal to 1, there's a good chance that the distribution is categorical
            feature_encoded=feature
            if feature.dtype == object:
                feature_encoded=self.__encoder.fit_transform(feature)    
            feature_encoded = sort(list(set(feature_encoded)))
            
            #Step for two successive modalities
            index=0
            step=1     
            while step == 1 and index<len(feature_encoded)-1:
                step = feature_encoded[index+1] - feature_encoded[index]
                index+=1
            
            #if every step is constant and equals to 1, we affect 1 to "one_step"
            one_step = 0
            if step==1:
                one_step = 1   
             
            #Counting unique values per feature
            unique_values = feature.nunique()
            #Summarizing everything in a list for every single feature
            new_features_list.append([is_float, unique_values, one_step])
        new_features_list = array(new_features_list)
        #We scale(mean:0 and std:1) the values in order to keep the big modalities within a certain range: we don't want "unique_values" modalities to be dominant but we also want to keep the outliers.
        new_features_list[:,1] = scale(new_features_list[:,1])
        #Dataframe depicting our new features
        new_features = pd.DataFrame(new_features_list, columns=["is_float", "unique_values", "one_step"])
        return new_features
       
    def __label_predictions(self, predictions, mappings):
       '''
       This functions labels our predictions according to the mappings.
       For example: 0 for "categorical" and 1 for "numerical".
       '''
       labeled_predictions = []
       for prediction in predictions:
           prediction = prediction[0]
           labeled_predictions.append(mappings[prediction])
       return labeled_predictions
       
    def __save_variables(self, path, variable):
        '''
        Save variables with pickle.
        '''
        with open(path, "wb") as file:
            pickle.dump(variable, file)
    
    def __load_variables(self, path):
        '''
        Load variables with pickle.
        '''
        loaded_variable = None
        with open(path, "rb") as file:
            loaded_variable = pickle.load(file)
        return loaded_variable
    
    def predict(self, original_data, verbose=1):
        '''
        We finally get our predictions by doing some data preprocessing first(counting unique values and checking if a feature has the data type float).
        '''
        self.__print_verbose    = verbose
        # 1- We keep the initial data types
        accurately_typed_data   = self.__keep_initial_data_types(original_data)
        # 2- We build our final_set for our model.
        features                = self.__build_final_set(accurately_typed_data)
        # 3- We get our predictions 
        model                   = load_model(join(dirname(getsourcefile(DataTypeIdentifier)), "data_type_identifier_model", "data_type_identifier.h5"))
        predictions             = model.predict_classes(features)
        # 4- We label our predictions. For instance 0 represents "categorical" and 1 represents "numerical"
        mappings                = self.__load_variables(join(dirname(getsourcefile(DataTypeIdentifier)), "saved_variables", "mappings.pickle"))
        labeled_predictions     = self.__label_predictions(predictions, mappings)
        # 5- We finally summarize everything in a dataframe
        final_predictions       = pd.DataFrame(labeled_predictions, columns=["Predictions"], index=original_data.columns)
        return final_predictions

    
   
