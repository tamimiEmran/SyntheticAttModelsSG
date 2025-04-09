

"""
The goal of this experiment is to see if a good model performance is indicative
 of a good model performance in the real world attacks
 

This code does three things:
    1. load the data.
    2. set up the training validation and testing data of synthetic attacks.
    3. set up the training validation and testing data of real attacks.
    4. There are 10 folds each. 
    
    
    abstraction required: Write a function that takes the fold id and if the 
    dataset is real or synthetic and return the three tuples. One for 
    training (x,y), validation (x,y), and testing (x,y)
    
    
    
    Step 2:
    1. initialize the five models
    2. build an objective function for each for hypertuning
    3. save the parameters for synthetic and real datasets.
    (for 5 fold for each model)
    
    abstractions required: Write a function that takes the model name and the 
    validation tuple and returns the parameters. If the model where already 
    saved load them. if not use the objective function to get the parameters 
    and save them.
    
    step3:
    1. Test each model on the same testing test (once real and once synthetic).
    2. Save the results as a dictionary, one for synthetic and one for real. 
    template = {
              'modelName1': {
                          'F1 score': [a list for the results of each fold]
                          'AUC': [a list for the results of each fold]
                          'precision: [A list for the results of each fold]
                          'recall: [A list for the results of each fold]'
                          }
              'modelName2': {
                          'F1 score': [a list for the results of each fold]
                          'AUC': [a list for the results of each fold]
                          'precision: [A list for the results of each fold]
                          'recall: [A list for the results of each fold]'
                          }
            }
    
    3. save the results locally
    step 4: 
    1. Plot the auc for the real and synthetic data of all five models with 
    error bars. The plot will have 10 error bars. Each two adjacent bars will
    belong to the same model, one for real and one for synthetic.
    The bars will visulize the average and std of the 10 folds.
    
    2. visualize the trade off between the precision and recall for all models 
    in a single figure. Two figures in total one for real and one for synthetic.
    
    
    
"""


import exp3_abstraction_results as r



results = r.resutls()

results_real, results_synthetic, results_synth_onReal = results

import exp3_abstraction_visualizing as v
v.plot_auc(results_real, results_synthetic, results_synth_onReal)

# v.plot_precision_recall(results_real, title= 'Precision-Recall Curve for real Data')
# v.plot_precision_recall(results_synthetic, title= 'Precision-Recall Curve for synthetic Data')



# v.plot_model_performance(results_real, title= 'Precision-Recall Curve for real Data')
# v.plot_model_performance(results_synthetic, title= 'Precision-Recall Curve for synthetic Data')

####

    