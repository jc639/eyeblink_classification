# Classifying eyeblink conditioning behvioural data
# Problem:
#   - Data is classified by hand into Conditional Response, No Conditional response
#   Alpha response (startle eyelid closure) or Not flat in baseline period.
#   - For analysis we only care about trials that are CR or not CR
#   - Data is messy, so machine learning will likely not be able to get extremely 
#   high accuracy, but what if we use model with probabilities to classify those that
#   which it is highly confident about and thus reduce workload

# datasets - timeseries of trial recordings in rows
# each individual row is one trial and label of
# 0 : 'No CR', 
# 1 : 'CR',
# 2 : 'Not Flat',
# 3 : 'Alpha'
#
# Previous analysis showed that MLPClassifier and Random Forest worked
# fairly well, >80% accuracy for both datasets
# Will tune hyperparameters and test performance/amount classified (workload reduced)
# at different decision boundaries
  
# getting around joblib protecting main loop
 
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import sklearn libraries, on basis of prev analysis RF and MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from precision_recall_evaluation import MulticlassDecisionBoundary      
import matplotlib.patches as patches  

# joblib in main, for when run as script
# this has to be run in python, not Spyder/IPython
# get joblib error if in IPython
if __name__ == '__main__':
    
    # iterate through the datasets    
    csvs = ['dataset 1', 'dataset 2']
    for csv in csvs:
        # read in dataset.
        dataset = pd.read_csv('{}.csv'.format(csv))
        
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
        
        # process class labels
        from sklearn.preprocessing import LabelEncoder
        y_class = LabelEncoder()
        y = y_class.fit_transform(y)
        
        
        # get a small holdout test set for later validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        
         
        # losing most of the data didn't seem to effect - smaller input without loss of accuracy
        # lets plot the change in accuracy over steps
        
        # classifier objects
        # step up iter to prevent non-convergence 
        # n_estimators = 100, will gridCV in a bit to find best
        nn_clf = MLPClassifier(max_iter=1000)
        rf_clf = RandomForestClassifier(n_estimators = 100)
        steps = np.array([1, 5, 10, 20, 30, 50], dtype='int')
        
        # lets do a 10-fold for each stepsize
        kfolds = KFold(n_splits=10)
        
        # make an empty dataframe for plotting the results
        step_df = pd.DataFrame(data=[], columns = ['Algorithm', 'Step Size', 'Accuracy'])
        
        # steps through
        for step in steps:
            reduced_X = X_train[:, ::step]
            
            # do the k folds
            for train, test in kfolds.split(reduced_X):
                X_tr = reduced_X[train]
                X_te = reduced_X[test]
                y_tr = y_train[train]
                y_te = y_train[test]
                
                # mlp classifier
                nn_clf.fit(X_tr, y_tr)
                        
                # random forest classifier
                rf_clf.fit(X_tr, y_tr)
                
                # make a temporary df and bind it to step_df
                tmp_df = pd.DataFrame(data=np.array([['MLP Classifier', step, nn_clf.score(X_te, y_te)],
                                                      ['Random Forest', step, rf_clf.score(X_te, y_te)]], dtype='object'), columns = ['Algorithm', 'Step Size', 'Accuracy'])
                
                step_df = pd.concat([step_df, tmp_df], ignore_index = True)
                
        step_df['Algorithm'] = step_df['Algorithm'].astype('str')
        step_df['Step Size'] = step_df['Step Size'].astype('int')
        step_df['Accuracy'] = step_df['Accuracy'].astype('float')
        
        # get means and SEM
        gr = step_df.groupby(['Algorithm', 'Step Size'])
        mean_df = gr.mean().reset_index()
        sem_df = gr.sem().reset_index()
        
        # plot
        MLP_scores = mean_df['Accuracy'][mean_df['Algorithm'] == 'MLP Classifier'].values
        MLP_sem = sem_df['Accuracy'][sem_df['Algorithm'] == 'MLP Classifier'].values
        RF_scores = mean_df['Accuracy'][mean_df['Algorithm'] == 'Random Forest'].values
        RF_sem = sem_df['Accuracy'][sem_df['Algorithm'] == 'Random Forest'].values
        
        os.makedirs('RF and MLP evaluation/{}'.format(csv))
            
        # plot with sem errorbar
        plt.plot(steps, MLP_scores, 'b-', label='MLP Classifier')
        plt.plot(steps, RF_scores, 'r-', label='Random Forest')
        plt.errorbar(steps, MLP_scores, yerr=MLP_sem, ecolor='b', fmt='none')
        plt.errorbar(steps, RF_scores, yerr=RF_sem, ecolor='r', fmt='none')
        plt.xlabel('Step Size for Slice over X columns')
        plt.ylabel('Mean accuracy (+-SEM)')
        plt.legend()
        plt.savefig(filename='RF and MLP evaluation/{}/reduction in features.png'.format(csv))
        plt.close()
        
        # mainly looks like downsampling has a general trend to worse accuracy
        # will work on full dataset, gridsearch to see if accuracy can be increased
        
        # first up random forests
        # randomsearch first then refine if increase in accuracy
        # drop cv to increase n_iter
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        max_feat = ['auto', 'log2', 0.25, 0.5, 0.75]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        parameters = {'n_estimators': n_estimators,
                      'max_features': max_feat,
                      'max_depth' : max_depth,
                      'min_samples_split' : min_samples_split,
                      'min_samples_leaf' : min_samples_leaf,
                      'bootstrap': bootstrap}
        rf_clf = RandomForestClassifier()
        rf_random_search = RandomizedSearchCV(estimator = rf_clf,
                                   param_distributions = parameters,
                                   cv = 5,
                                   n_iter = 30,
                                   n_jobs=-1)
        rf_random_search = rf_random_search.fit(X_train, y_train)
        rf_best_accuracy = rf_random_search.best_score_
        rf_best_parameters = rf_random_search.best_params_
        rf_test_score = rf_random_search.score(X_test, y_test)
        
        
        # get the scores, i know this should be cv_results but find this more readable
        rf_random_scores = rf_random_search.grid_scores_
        rf_random_scores = [str(sc) + '\n' for sc in rf_random_scores]
        
        # write to file for inspection
        with open('RF and MLP evaluation/{}/RF random search cv.txt'.format(csv), 'w') as f:
            f.write('BEST ACCURACY SCORE:\n')
            f.write(str(rf_best_accuracy) + '\n')
            f.write('BEST PARAMETERS:\n')
            f.write(str(rf_best_parameters) + '\n\n')
            f.write('ACCURACY SCORE ON HOLDOUT TEST SET:\n')
            f.write(str(rf_test_score) + '\n\n\n')
            
            f.writelines(rf_random_scores)
            
    
    
        # won't proceed wiht gridsearch but will use best params
           
        
        # grid search nn - 1 hidden layer, more is not usually necessary
        # nodes rule of thumb: somewhere between input size and output size
        hidden_nodes = [int(x) for x in np.linspace(1, X_train.shape[1], 5)]
        hidden_nodes = [(x,) for x in hidden_nodes]
        
        # try some different activation function, currently rectified linear
        activation_func = ['relu', 'logistic', 'tanh']
        
        # L2 penalty
        alpha = [0.001, 0.01, 0.1, 1, 10, 100]
        
        # solver
        solver = ['lbfgs', 'adam']
        
        # mlp
        nn_clf = MLPClassifier(max_iter=1000)
        parameters = {'hidden_layer_sizes' : hidden_nodes,
                      'activation': activation_func,
                      'alpha': alpha,
                      'solver' : solver}
        mlp_grid_search = GridSearchCV(nn_clf, 
                                       param_grid = parameters,
                                       cv = 3,
                                       n_jobs=-1)
        
        mlp_grid_search = mlp_grid_search.fit(X_train, y_train)
        mlp_best_accuracy = mlp_grid_search.best_score_
        mlp_best_parameters = mlp_grid_search.best_params_
        mlp_test_score = mlp_grid_search.score(X_test, y_test)
        
        
        # get the scores to write to file
        mlp_grid_scores = mlp_grid_search.grid_scores_
        mlp_grid_scores = [str(sc) + '\n' for sc in mlp_grid_scores]
        
        with open('RF and MLP evaluation/{}/MLP grid search cv.txt'.format(csv), 'w') as f:
            f.write('BEST ACCURACY SCORE:\n')
            f.write(str(mlp_best_accuracy) + '\n')
            f.write('BEST PARAMETERS:\n')
            f.write(str(mlp_best_parameters) + '\n')
            f.write('ACCURACY SCORE ON HOLDOUT TEST SET:\n')
            f.write(str(mlp_test_score) + '\n\n\n')
            
            f.writelines(mlp_grid_scores)
        
        # determine how decision function boundary affects precision and recall
        # really we want to keep precision as high as possible while
        # still classifying a good chunk of the data
        
        # initialise models
        # stratified kfold = 4
        nn_clf = MLPClassifier(**mlp_best_parameters)
        rf_clf = RandomForestClassifier(**rf_best_parameters)
        
        # make an empty dataframe, will concat the results of each fold
        # then later will make a sns pointplot
        mlp_df = pd.DataFrame(data = [], columns = ['decision_boundary', 
                              'label', 'score_type', 'scores'])
        rf_df = pd.DataFrame(data = [], columns = ['decision_boundary', 'label',
                             'score_type', 'scores'])
        
        # stratified k fold, preserves proportions of labels
        # important as we are assessing curves for each individual label
        # want at least some of each class in the test fold
        strat_k = StratifiedKFold(n_splits = 4)
        
        # label them predictors and labels (train/test) to avoid confusion
        # with X_train, X_test ...
        for train, test in strat_k.split(X_train, y_train):
            train_predictors = X_train[train]
            train_labels = y_train[train]
            test_predictors = X_train[test]
            test_labels = y_train[test]
            
            # init the Decision Boundary Object 
            mlp_decision_curves = MulticlassDecisionBoundary(nn_clf, 
                                                             train_predictors,
                                                             train_labels)
            mlp_decision_curves = mlp_decision_curves.pre_rec_curves(test_predictors,
                                                                     test_labels,
                                                                     step = 0.1)
            mlp_df = pd.concat([mlp_df, mlp_decision_curves], ignore_index = True)
            
            rf_decision_curves = MulticlassDecisionBoundary(rf_clf,
                                                            train_predictors,
                                                            train_labels)
            rf_decision_curves = rf_decision_curves.pre_rec_curves(test_predictors,
                                                                   test_labels,
                                                                   step=0.1)
            rf_df = pd.concat([rf_df, rf_decision_curves], ignore_index=True)
            
            
        # makes some plots
        mlp_grid = sns.FacetGrid(mlp_df, col='label', hue='score_type',
                                 col_wrap = 2)
        mlp_grid = mlp_grid.map(sns.pointplot, 'decision_boundary', 'scores')
        mlp_grid.axes[2].set_xlabel('decision boundary')
        mlp_grid.axes[3].set_xlabel('decision boundary')
        # colours dont show when you call add_legend(), work around
        lab = mlp_grid.hue_names
        col = sns.color_palette('deep').as_hex()[:len(lab)]
        handles = [patches.Patch(color=col, label=lab) for col, lab in zip(col, lab)]
        plt.legend(handles = handles, title='score type', loc = 'center left', bbox_to_anchor=(1, 1))
        mlp_grid.savefig('RF and MLP evaluation/{}/MLP decision boundary precision recall kfold.png'.format(csv))
        plt.close()
        
        # same again for RF
        rf_grid = sns.FacetGrid(rf_df, col='label', hue='score_type',
                                 col_wrap = 2)
        rf_grid = rf_grid.map(sns.pointplot, 'decision_boundary', 'scores')
        rf_grid.axes[2].set_xlabel('decision boundary')
        rf_grid.axes[3].set_xlabel('decision boundary')
        # colours dont show when you call add_legend(), work around
        lab = rf_grid.hue_names
        col = sns.color_palette('deep').as_hex()[:len(lab)]
        handles = [patches.Patch(color=col, label=lab) for col, lab in zip(col, lab)]
        plt.legend(handles = handles, title='score type', loc = 'center left', bbox_to_anchor=(1, 1))
        rf_grid.savefig('RF and MLP evaluation/{}/RF decision boundary precision recall kfold.png'.format(csv))
        plt.close()
        
        # on the test data
        test_mlp_curve = MulticlassDecisionBoundary(nn_clf, X_train, y_train)
        test_mlp_curve = test_mlp_curve.pre_rec_curves(X_test, y_test, step=0.1)
        
        # plot
        test_mlp_grid = sns.FacetGrid(test_mlp_curve, col='label', hue='score_type',
                                 col_wrap = 2)
        test_mlp_grid = test_mlp_grid.map(plt.plot, 'decision_boundary', 'scores')
        plt.legend(title='score type', loc = 'center left', bbox_to_anchor=(1, 1))
        plt.savefig('RF and MLP evaluation/{}/MLP decision boundary precision recall validation set.png'.format(csv))
        plt.close()
        
        # same again rf
        # on the test data
        test_rf_curve = MulticlassDecisionBoundary(rf_clf, X_train, y_train)
        test_rf_curve = test_rf_curve.pre_rec_curves(X_test, y_test, step=0.1)
        
        # plot
        test_rf_grid = sns.FacetGrid(test_rf_curve, col='label', hue='score_type',
                                 col_wrap = 2)
        test_rf_grid = test_rf_grid.map(plt.plot, 'decision_boundary', 'scores')
        plt.legend(title='score type', loc = 'center left', bbox_to_anchor=(1, 1))
        plt.savefig('RF and MLP evaluation/{}/RF decision boundary precision recall validation set.png'.format(csv))
        plt.close()
        
        
        

