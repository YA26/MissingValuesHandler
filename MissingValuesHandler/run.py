from MissingValuesHandler.missing_data_handler import RandomForestImputer
from os.path import join
from pandas import read_csv

"""
############################################
############# IMPORT DATA  #################
############################################
"""
data = read_csv(join("data","scoring.csv"), sep=",", index_col=False)

"""
############################################
############### RUN TIME ###################
############################################
"""
#Main object
random_forest_imputer = RandomForestImputer(data=data,
                                            target_variable_name="Status",
                                            training_resilience=3, 
                                            n_iterations_for_convergence=5,
                                            forbidden_features_list=["Age"],
                                            ordinal_features_list=[])

#Setting the ensemble model parameters: it could be a random forest regressor or classifier
random_forest_imputer.set_ensemble_model_parameters(n_estimators=40, additional_estimators=10)

#Launching training and getting our new dataset
new_data = random_forest_imputer.train(sample_size=0.3)
"""
############################################
########## DATA RETRIEVAL ##################
############################################
"""
sample_used                         = random_forest_imputer.get_sample()
features_type_prediction            = random_forest_imputer.get_features_type_predictions()
target_variable_type_prediction     = random_forest_imputer.get_target_variable_type_prediction()
encoded_features                    = random_forest_imputer.get_encoded_features()
encoded_target_variable             = random_forest_imputer.get_target_variable_encoded()
final_proximity_matrix              = random_forest_imputer.get_proximity_matrix()
final_distance_matrix               = random_forest_imputer.get_distance_matrix()
weighted_averages                   = random_forest_imputer.get_nan_features_predictions(option="all")
convergent_values                   = random_forest_imputer.get_nan_features_predictions(option="conv")
divergent_values                    = random_forest_imputer.get_nan_features_predictions(option="div")
ensemble_model_parameters           = random_forest_imputer.get_ensemble_model_parameters()
all_target_value_predictions        = random_forest_imputer.get_nan_target_values_predictions(option="all")
target_value_predictions            = random_forest_imputer.get_nan_target_values_predictions(option="one")


"""
############################################
######## WEIGHTED AVERAGES PLOT ############
############################################
"""
random_forest_imputer.create_weighted_averages_plots(directory_path="graphs", both_graphs=1)

"""
############################################
######## TARGET VALUE(S) PLOT ##############
############################################
"""
random_forest_imputer.create_target_pred_plot(directory_path="graphs")

"""
############################################
##########      MDS PLOT    ################
############################################
"""
mds_coordinates = random_forest_imputer.get_mds_coordinates(n_dimensions=3, distance_matrix=final_distance_matrix)
random_forest_imputer.show_mds_plot(mds_coordinates, plot_type="3d")
