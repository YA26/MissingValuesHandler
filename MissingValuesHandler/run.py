from missing_data_handler import MissingDataHandler
from os.path import join
from pandas import read_csv

"""
############################################
############# MAIN OBJECT ##################
############################################
"""
missing_data_handler = MissingDataHandler(training_resilience=3)


"""
############################################
############### RUN TIME ###################
############################################
"""
data = read_csv(join("data","Loan_approval.csv"), sep=",", index_col=False)
#Setting the ensemble model parameters: it could be a random forest regressor or classifier
missing_data_handler.set_ensemble_model_parameters(n_estimators=80, additional_estimators=20)

#Launching training and getting our new dataset
new_data = missing_data_handler.train(data=data, 
                                      target_variable_name="Loan_Status",  
                                      n_iterations_for_convergence=5,
                                      verbose=1,
                                      path_to_save_dataset=join("data", "Loan_approval_no_nan.csv"),
                                      forbidden_variables_list=[])


"""
############################################
########## DATA RETRIEVAL ##################
############################################
"""
features_type_prediction            = missing_data_handler.get_features_type_predictions()
target_variable_type_prediction     = missing_data_handler.get_target_variable_type_prediction()
encoded_features                    = missing_data_handler.get_encoded_features()
encoded_target_variable             = missing_data_handler.get_target_variable_encoded()
final_proximity_matrix              = missing_data_handler.get_proximity_matrix()
final_distance_matrix               = missing_data_handler.get_distance_matrix()
weighted_averages                   = missing_data_handler.get_all_weighted_averages()
convergent_values                   = missing_data_handler.get_convergent_values()
divergent_values                    = missing_data_handler.get_divergent_values()
ensemble_model_parameters           = missing_data_handler.get_ensemble_model_parameters()


"""
############################################
######## WEIGHTED AVERAGES PLOT ############
############################################
"""
#missing_data_handler.create_weighted_averages_plots(directory_path="img", both_graphs=1, verbose=0)


"""
############################################
##########      MDS PLOT    ################
############################################
"""
mds_coordinates = missing_data_handler.get_mds_coordinates(n_dimensions=3)
missing_data_handler.show_mds_pot(mds_coordinates, plot_type="3d")
