# -*- coding: utf-8 -*-
# Classifying eyeblink conditioning behvioural data
# Problem:
#   - Data is classified by hand into Conditional Response, No Conditional response
#   Alpha response (startle eyelid closure) or Not flat in baseline period.
#   - For analysis we only care about trials that are CR or not CR
#
# quick exploratory analysis
# which algorithm performs best: SVM, logisitic regression, conv neural net
from scipy.stats import sem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 2 datasets, slightly different trial lengths and stimulus types, probably need two models
# will test over both of them.
csvs = ['dataset 1', 'dataset 2']

for csv in csvs:
    # import the dataset
    dataset = pd.read_csv('{}.csv'.format(csv))
    
    # time values
    time_values = np.array([], dtype=float)
    for cols in dataset.columns:
        if 'timepoint' in cols:
            time_values = np.append(time_values, np.float(cols.strip('timepoint_')))
    
    # drop the unlabelled (nan rows), to leave the labelled dataset
    labelled_data = dataset.dropna(axis=0)
    
    # plots of average response by group
    counts = labelled_data.groupby(['trial','Label']).size().reset_index(name='counts')
    
    # # "No Conditional Response",0, | "Conditional Response",1,
    #                | "Not Flat Baseline",2 | "Alpha Response",3
    # relabel the counts
    counts['Label'] = counts['Label'].astype(int)
    repl_dict = {0 : 'No CR', 1 : 'CR', 2 : 'Not Flat', 3 : 'Alpha'}
    counts['Label'] = counts['Label'].replace(repl_dict)
     
    # CR and No CR are most common, but there is a lot of not flat too
    # create plot, first create folder to store
    os.makedirs('Exploratory plots/{}'.format(csv))
    
    facet = sns.FacetGrid(counts, col='trial')
    facet.map(sns.barplot, 'Label', 'counts')
    facet.set_xticklabels(rotation=20)
    facet.savefig(filename='Exploratory plots/{}/count plot.png'.format(csv))
    plt.close()
    
    # lets have a look at the average waveform for the 
    # response types
    avg_responses = labelled_data.iloc[:, 5:]
    avg_responses = avg_responses.groupby(['Label']).apply(np.mean, axis=0)
    max_y = avg_responses.iloc[:, 5:avg_responses.shape[1]-1].max().max() + 0.3
    min_y = avg_responses.iloc[:, 5:avg_responses.shape[1]-1].min().min() - 0.1
    
    # get standard error to plot a ribbon around
    sem_responses = labelled_data.iloc[:, 5:]
    sem_responses = sem_responses.groupby(['Label']).apply(sem, axis=0)
    
    
    # lets plot these with ribbons for SEM
    fig = plt.figure()
    for i in range(0, 4):
        fig.add_subplot(2, 2, i + 1)
        y_vals = avg_responses.iloc[i,:avg_responses.shape[1] - 1].values
        y_upper = y_vals + sem_responses[i][ :len(sem_responses[i]) - 1]
        y_lower = y_vals - sem_responses[i][ :len(sem_responses[i]) - 1]
        plt.plot(time_values, y_vals)
        plt.fill_between(time_values, y_lower, y_upper, alpha=0.4)
        plt.ylim([min_y, max_y])
        plt.xlabel('Time values relative to stimulus (S)')
        plt.ylabel('Eyelid closure')
        plt.title('{}'.format(repl_dict.get(i)))
    plt.tight_layout()
    plt.savefig('Exploratory plots/{}/average responses for each class.png'.format(csv))
    plt.close()
    
    # shows that in the average response profile there is difference
    # alpha is messy - this basically means anything that starts too early
    # but there are low numbers of this --> Alphas and not flat are excluded
    # from analysis. We want to classify CR and no CR as best as possible
    # probabilist models will allow us to change decision func
    # probably can't classify with no errors but might be able
    # to reduce workload. Currently 15,000 trials need to be classified
    # by hand...
    
    # import the sklearn libraries
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    
    # np arrays for algos
    X = labelled_data.iloc[:, 5:labelled_data.shape[1] - 1].values
    y = labelled_data.iloc[:, labelled_data.shape[1] - 1].values
    
    # process class labels
    from sklearn.preprocessing import LabelEncoder
    y_class = LabelEncoder()
    y = y_class.fit_transform(y)
    
    # most likely non-linear problem
    svm_clf = SVC(kernel = 'rbf')
    log_reg = LogisticRegression()
    nn_clf = MLPClassifier(max_iter=1000)
    naive_clf = GaussianNB()
    forest_clf = RandomForestClassifier(n_estimators=100)
    x_scaler = StandardScaler()
    
    kfold = KFold(n_splits = 10)
    
    # SVC, logistic regression, MLP, NB and random forest scores in that row order
    # think scaling might make this worse
    # data is already trial voltage readings minus mean(baseline)
    # therefore units at each timepoint have some meaning --- will test
    # secondly think data is oversampled, 1kHz voltage readings
    # make nice plots but for determining CR / no CR -  a lot is redundant
    # will test against downsample simply by taking every 10th point
    # potential for PCA, but quickly explore raw input
    # makes m >> p, at the moment m < p
    
    # unscaled scores
    scores = np.zeros((10, 5))
    # scaled
    sc_scores = np.zeros((10, 5))
    # downsampled
    dwn_scores = np.zeros((10, 5))
    
    split_num = 0
    for train, test in kfold.split(X):
        # split
        X_train = X[train]
        X_test = X[test]
        y_train = y[train]
        y_test = y[test]
        
        # scale - fit/transform to train and transform test
        X_train_scaled = x_scaler.fit_transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)
        
        # down sample -> 100Hz
        X_train_down = X_train[:, ::10]
        X_test_down = X_test[:, ::10]
        
        # fit models - first SVC
        svm_clf.fit(X_train, y_train)
        scores[split_num, 0] = svm_clf.score(X_test, y_test)
        
        # scaled scores
        svm_clf.fit(X_train_scaled, y_train)
        sc_scores[split_num, 0] = svm_clf.score(X_test_scaled, y_test)
        
        # downsampled scores
        svm_clf.fit(X_train_down, y_train)
        dwn_scores[split_num, 0] = svm_clf.score(X_test_down, y_test)
        
        # logistic regression
        log_reg.fit(X_train, y_train)
        scores[split_num, 1] = log_reg.score(X_test, y_test)
        
        # scaled scores
        log_reg.fit(X_train_scaled, y_train)
        sc_scores[split_num, 1] = log_reg.score(X_test_scaled, y_test)
        
        # logistic regression - down scores
        log_reg.fit(X_train_down, y_train)
        dwn_scores[split_num, 1] = log_reg.score(X_test_down, y_test)
        
        # MLP classifier - non-scaled
        nn_clf.fit(X_train, y_train)
        scores[split_num, 2] = nn_clf.score(X_test, y_test)
        
        # scaled scores
        nn_clf.fit(X_train_scaled, y_train)
        sc_scores[split_num, 2] = nn_clf.score(X_test_scaled, y_test)
        
        # downsampled scores
        nn_clf.fit(X_train_down, y_train)
        dwn_scores[split_num, 2] = nn_clf.score(X_test_down, y_test)
        
        # Naive bayes
        naive_clf.fit(X_train, y_train)
        scores[split_num, 3] = naive_clf.score(X_test, y_test)
        
        naive_clf.fit(X_train_scaled, y_train)
        sc_scores[split_num, 3] = naive_clf.score(X_test_scaled, y_test)
    
        naive_clf.fit(X_train_down, y_train)
        dwn_scores[split_num, 3] = naive_clf.score(X_test_down, y_test)
    
        # random forest - ensemble of decision trees
        forest_clf.fit(X_train, y_train)
        scores[split_num, 4] = forest_clf.score(X_test, y_test)
        
        forest_clf.fit(X_train_scaled, y_train)
        sc_scores[split_num, 4] = forest_clf.score(X_test_scaled, y_test)
        
        forest_clf.fit(X_train_down, y_train)
        dwn_scores[split_num, 4] = forest_clf.score(X_test_down, y_test)        
        
        # iterate split number
        split_num += 1
        
    # lets do some plots of scores
    # get it into a tidy format and plot with sns
    # MLPClassifier does the best by far
    # create label column in arrays
    scores = scores.astype('object')
    scores = np.hstack((scores, np.array(['no preprocessing' for i in range(0, len(scores))], dtype='object').reshape(-1, 1)))
    
    dwn_scores = dwn_scores.astype('object')
    dwn_scores = np.hstack((dwn_scores, np.array(['downsampled' for i in range(0, len(dwn_scores))], dtype='object').reshape(-1, 1)))
    
    sc_scores = sc_scores.astype('object')
    sc_scores = np.hstack((sc_scores, np.array(['scaled' for i in range(0, len(dwn_scores))], dtype='object').reshape(-1, 1)))
    
    score_array = [scores, dwn_scores, sc_scores]
    
    # make a tidy dataframe for seaborn
    algorithm_names = ['SVM', 'Logistic Regression', 'MLP Classifier', 'Naive Bayes', 'Random Forest']
    algo_comparison = pd.DataFrame(data=[], columns = ['Accuracy Score', 'X Processing', 'Algorithm'])
    
    # stack the scores
    for i in np.arange(0, len(score_array)):
        sc_ar = score_array[i]
        for k in np.arange(0, len(algorithm_names)):
            dat = np.hstack((sc_ar[:, [k, 5]], np.array([algorithm_names[k] for j in range(0, len(sc_ar))]).reshape(-1, 1)))
            tmp_df = pd.DataFrame(data= dat, columns = ['Accuracy Score', 'X Processing', 'Algorithm', ])
            algo_comparison = pd.concat([algo_comparison, tmp_df])
    
    # currently objects, convert to float and str    
    algo_comparison['Accuracy Score'] = algo_comparison['Accuracy Score'].astype('float') 
    algo_comparison['X Processing'] = algo_comparison['X Processing'].astype('str')
    algo_comparison['Algorithm'] = algo_comparison['Algorithm'].astype('str')
    
    # pointplot
    sns.pointplot(x='X Processing', y='Accuracy Score', hue='Algorithm', data=algo_comparison, dodge = .3, join = False,
                  hue_order = ['MLP Classifier', 'Random Forest', 'SVM', 'Logistic Regression', 'Naive Bayes'])
    plt.legend(bbox_to_anchor=(.86, .9), loc=2, borderaxespad=0.)
    plt.xlabel('Processing of X array')
    plt.ylabel('Accuracy Scores')
    plt.savefig(filename='Exploratory plots/{}/Algorithm comparison.png'.format(csv))
    plt.close()
    
    # MLP and random forests do pretty well
    # get ~85% with basically default hyperparams
    # will continue with these and then GridSearchCV to see if it can be improved
    # then will change decision boundaries to see if we can get high accuracy on
    # what we classify (precision), whilst still classify a fair chunk of the data (recall)
    # if can reduce workload by 50% thats 000's of trials that we no longer have
    # to classify by hand
    # will carry on in evaluation_of_classifier.py for clarity
