from missing_data_handler import RandomForestImputer
from os.path import join
from pandas import read_csv

"""
############################################
############# MAIN OBJECT ##################
############################################
"""
random_forest_imputer = RandomForestImputer(training_resilience=3)

"""
############################################
############### RUN TIME ###################
############################################
"""
data = read_csv(join("data","Loan_approval.csv"), sep=",", index_col=False)


#Setting the ensemble model parameters: it could be a random forest regressor or classifier
random_forest_imputer.set_ensemble_model_parameters(n_estimators=80, additional_estimators=20)

#Launching training and getting our new dataset
new_data = random_forest_imputer.train(data=data, 
                                      target_variable_name="Loan_Status",  
                                      n_iterations_for_convergence=5,
                                      verbose=1,
                                      path_to_save_dataset=join("data", "Loan_approval_no_nan.csv"),
                                      forbidden_variables_list=["Credit_History"])

"""
############################################
########## DATA RETRIEVAL ##################
############################################
"""
features_type_prediction            = random_forest_imputer.get_features_type_predictions()
target_variable_type_prediction     = random_forest_imputer.get_target_variable_type_prediction()
encoded_features                    = random_forest_imputer.get_encoded_features()
encoded_target_variable             = random_forest_imputer.get_target_variable_encoded()
final_proximity_matrix              = random_forest_imputer.get_proximity_matrix()
final_distance_matrix               = random_forest_imputer.get_distance_matrix()
weighted_averages                   = random_forest_imputer.get_all_weighted_averages()
convergent_values                   = random_forest_imputer.get_convergent_values()
divergent_values                    = random_forest_imputer.get_divergent_values()
ensemble_model_parameters           = random_forest_imputer.get_ensemble_model_parameters()
target_value_predictions            = random_forest_imputer.get_target_values_predictions()

"""
############################################
######## WEIGHTED AVERAGES PLOT ############
############################################
"""
random_forest_imputer.create_weighted_averages_plots(directory_path="graphs", both_graphs=1, verbose=1)


"""
############################################
######## TARGET VALUE(S) PLOT ##############
############################################
"""
random_forest_imputer.create_target_pred_plot(directory_path="graphs", verbose=1)


"""
############################################
##########      MDS PLOT    ################
############################################
"""
mds_coordinates = random_forest_imputer.get_mds_coordinates(n_dimensions=3)
random_forest_imputer.show_mds_plot(mds_coordinates, plot_type="3d")
