# lets see how much is classifies and what it is like
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
plt.ioff()

# import class
from precision_recall_evaluation import MulticlassDecisionBoundary
from sklearn.ensemble import RandomForestClassifier

# so the function predict of MulticlassDecisionBoundary classifies
# if the returned probability by the model is higher than the boundary
# if not it leaves that sample as np.nan to be classified by a human
# this is messy data, 2 people may not agree on edge cases so we only want to
# classify where high probability to get good precision but also fair recall

# parameters for two trees
rf_params = [{'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 10, 'bootstrap': True},
             {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 80, 'bootstrap': True}]
# decision boundaries of whether to classify as predicted class or 
# leave it to experimenter chosen on basis of getting precision as close to 1
# but also having recall as high as possible
decision_bounds = [[0.9, 0.9, 0.9, 0.6], 
                   [0.8, 0.9, 0.8, 0.9]]
# two datasets
csvs = ['dataset 1', 'dataset 2']
# label classification replacement
repl_dict = {0 : 'No CR', 1 : 'CR', 2 : 'Not Flat', 3 : 'Alpha'}

for i in range(0, len(csvs)):
    # read in dataset.
    dataset = pd.read_csv('{}.csv'.format(csvs[i]))
    
    # time values, most columns are voltage value at that time point
    time_values = np.array([], dtype=float)
    for cols in dataset.columns:
        if 'timepoint' in cols:
            time_values = np.append(time_values, np.float(cols.strip('timepoint_')))
    
    # drop the unlabelled (nan rows), to leave the labelled dataset
    labelled_data = dataset.dropna(axis=0)
        
    # get X and y
    # np arrays for algos
    X = labelled_data.iloc[:, 5:labelled_data.shape[1] - 1].values
    y = labelled_data.iloc[:, labelled_data.shape[1] - 1].values
    
    # get this unlabelled
    X_unlabelled = dataset[np.isnan(dataset['Label'])]
    X_unlabelled = X_unlabelled.iloc[:, 5:X_unlabelled.shape[1] - 1].values
    
    # classify at the decision bounds
    rf_clf = RandomForestClassifier(**rf_params[i])
    
    # create Multiclass
    multiclass_decision = MulticlassDecisionBoundary(rf_clf, X, y)
    
    # predictions
    predictions = multiclass_decision.predictions(X=X_unlabelled, decision_boundary=decision_bounds[i])
    
    # numbers classified
    n_classified = np.invert(np.isnan(predictions)).sum()
    total_samples = len(predictions)
    perc_classified = n_classified / total_samples * 100

    
    # make a directory
    os.makedirs('Random Forest Classifier/{}'.format(csvs[i]))
    
    # stick numbers into file
    with open('Random Forest Classifier/{}/quantification of classification of unlabelled data.txt'.format(csvs[i]), 'w') as f:
        f.write('NUMBER CLASSIFIED:\n')
        f.write(str(n_classified) + '\n')
        f.write('TOTAL NUMBER OF SAMPLES:\n')
        f.write(str(total_samples) + '\n')
        f.write('PERCENT CLASSIFIED:\n')
        f.write('{:.2f}%\n'.format(perc_classified))
        response_type = []
        response_counts = np.empty(len(decision_bounds[i]))
        for k in range(0, len(decision_bounds[i])):    
            f.write('{} {}s classified\n'.format(len(predictions[predictions == k]), repl_dict.get(k)))
            response_type.append(repl_dict.get(k))
            response_counts[k] = len(predictions[predictions == k])
    
    # make a plot of counts and save
    counts_df = pd.DataFrame(data = {'type' : response_type,
                                     'counts' : response_counts})
    counts_df.plot('type', 'counts', kind='bar')
    plt.savefig('Random Forest Classifier/{}/counts of labels by type.png'.format(csvs[i]))
    plt.close()
    
    # make a random set of plots of the newly labelled data
    for k in range(0, len(decision_bounds[i])):
        indexes = predictions == k
        
        # make a directory for that label
        os.makedirs('Random Forest Classifier/{}/{}/'.format(csvs[i], repl_dict.get(k)))
        
        predictedk_sample = X_unlabelled[indexes]
        
        if predictedk_sample.shape[0] != 0:
            # lets 20 random rows from each class and plot
            for j in range(1, 51):
                row_n = np.random.randint(0, predictedk_sample.shape[0])
                
                plt.plot(time_values, predictedk_sample[row_n, :])
                plt.xlabel('time value (secs)')
                plt.ylabel('eyelid closure (arbitary voltage)')
                ymin, ymax = plt.ylim()
                if (ymax - ymin) < 1:
                    plt.ylim(ymax = ymin + 1)
                plt.savefig('Random Forest Classifier/{}/{}/{} plot {}.png'.format(csvs[i], repl_dict.get(k), repl_dict.get(k), j))
                plt.close()
            
    
    # reassemble dataset and save
    predicted_data = dataset[np.isnan(dataset['Label'])].copy()
    predicted_data['Label'] = predictions
    
    # write to csvs
    predicted_data.to_csv('Random Forest Classifier/{}/predicted_data.csv'.format(csvs[i]), index=False)
    # training data
    labelled_data.to_csv('Random Forest Classifier/{}/training_data.csv'.format(csvs[i]), index=False)
            
            
    
    
    
    