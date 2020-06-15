from tensorflow.keras import layers
from sklearn.preprocessing import scale
from numpy import sort, array
import tensorflow as tf
import pandas as pd
import pickle

class DataTypeIdentifier(object):
        
    def __init__(self, encoder=None):
        """
        The encoder is used to encode the target variable
        The default value of the encoder is None because we will not always need to encode the target variable. Especially if we only want to use the "predict method"
        """
        if encoder is not None:
            self.__encoder = encoder()
            self.__encoder_2 = encoder()
        
    def get_encoder(self):
        return self.__encoder
        
    def keep_initial_data_types(self, data):
        '''
        This function helps us render immuable the data type of every feature(column) that was set before the data was imported.
        The basic idea here is to avoid having integers being transformed into float.
        We are using a special pandas datatype (Int64Dtype) to achieve it.
        '''
        for column in data.columns:
            try:    
                print("- Feature *{}* treated".format(column))
                data=data.astype({column: pd.Int64Dtype()})
            except TypeError:
                pass     
        return data
    
    def build_final_set(self, original_correctly_typed_data, target_variable=None):
        '''
        Function to build training/final set. 
        We create our final features (is_float and unique_values) to help predict if a feature is numerical or categorical 
        We also encode our target variable
        '''
        correctly_typed_data = original_correctly_typed_data.copy()
        correctly_typed_data.dropna(inplace=True) 
        new_features_list = []
        for feature_name in correctly_typed_data:
            feature = correctly_typed_data[feature_name]
            #Checking feature data type: we create a variable "is_float" to check if the data type of the feature is float. The default value is "False" or 0
            is_float = 0 
            if feature.dtype == float:
                is_float = 1       
            
            #Computing steps between one modality and the one following it: if steps are consstant and equal to 1, there's a good chance that the distribution is categorical
            feature_encoded=feature
            if feature.dtype == object:
                feature_encoded=self.__encoder_2.fit_transform(feature)    
            feature_encoded = sort(list(set(feature_encoded)))
            
            #Step for two succesive modalities
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
        new_features_list       = array(new_features_list)
        #We scale(mean:0 and std:1) the values in order to keep the big modalities within a certain range: we don't want "unique_values" modalities to be dominant but we also want to keep the outliers.
        new_features_list[:,1]  = scale(new_features_list[:,1])
        #Dataframe depicting our new features
        new_features = pd.DataFrame(new_features_list, columns=["is_float", "unique_values", "one_step"])
        
        #Encoding our target variable
        target_variable_encoded = None
        if target_variable is not None: 
            target_variable_encoded = pd.DataFrame(self.__encoder.fit_transform(target_variable))
        
        #Putting our features and our target variable in one dictionary
        features_target_dict = {'new_features': new_features, 'target_variable_encoded':target_variable_encoded}
        return features_target_dict
       
    def get_target_variable_class_mappings(self):
        '''
        Returns mappings of our target variable modalities
        '''
        original_values  = self.__encoder.classes_ 
        encoded_values   = self.__encoder.transform(original_values)
        mappings         = {encoded_values[0]: original_values[0], encoded_values[1]: original_values[1]}
        return mappings
    
    def sigmoid_neuron(self, X, y,path, epoch, validation_split, batch_size):
        model = tf.keras.Sequential()
        model.add(layers.Dense(units=X.shape[1] , input_dim=X.shape[1]))
        model.add(layers.Dense(units=1, activation='sigmoid'))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) 
        model.fit(X, y, epochs=epoch, validation_split=validation_split, batch_size=batch_size)
        model.save(path)
        print("model saved to "+str(path))
        return model
    
    def label_predictions(self, predictions, mappings):
       '''
       This functions labels our predictions according to the mappings.
       For example: 0 for "categorical" and 1 for "numerical".
       '''
       labeled_predictions = []
       for prediction in predictions:
           prediction = prediction[0]
           labeled_predictions.append(mappings[prediction])
       return labeled_predictions
       
    def predict(self, data, mappings, model):
        '''
        We finally get our predictions by doing some data preprocessing first(counting unique values and checking if a feature has the data type float).
        '''

        # 1- We keep the initial data types
        accurately_typed_data = self.keep_initial_data_types(data)
        # 2- We build our final_set for our model. !no target variable is given because it is what we are trying to predict
        final_set             = self.build_final_set(original_correctly_typed_data=accurately_typed_data)
        # 3- We get our two features: "is_float" and "unique_values" 
        features              = final_set["new_features"]
        # 4- We get our predictions 
        predictions           = model.predict_classes(features)
        # 5- We label our predictions. For instance 0 represents "categorical" and 1 represents "numerical"
        labeled_predictions   = self.label_predictions(predictions, mappings)
        # 6- We finally summarize everything in a dataframe
        final_predictions     = pd.DataFrame(labeled_predictions, columns=["Predictions"], index=data.columns)
        return final_predictions

    def save_variables(self, path, variable):
        '''
        Save variables with pickle.
        '''
        with open(path, "wb") as file:
            pickle.dump(variable, file)
    
    def load_variables(self, path):
        '''
        Load variables with pickle.
        '''
        loaded_variable = None
        with open(path, "rb") as file:
            loaded_variable = pickle.load(file)
        return loaded_variable
    
    
   
