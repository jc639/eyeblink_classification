# precision recall evaluation class object
import numpy as np
import pandas as pd

class MulticlassDecisionBoundary:
    
    def __init__(self, classifier, X_train, y_train):        
        self.X_train = X_train
        self.y_train = y_train
        self.classifier = classifier.fit(self.X_train, self.y_train)
    
    
    def pre_rec_curves(self, X_test=None, y_test=None, step=0.1):
        '''
        Returns precision and recall at different
        boundaries of decision function, for each class
        Precision - how many selected items correct
        Recall - how many relevant items are selected
        '''
        # get classes
        classes = self.classifier.classes_
        
        # get probability array, probability of class (cols) for each
        # training sample (rows), and max value
        prob_array = self.classifier.predict_proba(X_test)
        max_prob = np.amax(prob_array, axis = 1)
               
        # class prediction is selected by highest probability
        predicted_class = self.classifier.predict(X_test)
        
        # decision boundaries to test over
        boundary = np.arange(step, 1.01, step)
        # smallest boundary is 1/len(classes), as classified by highest
        # no point checking decision below that threshold
        boundary = boundary[boundary >= 1/len(classes)]
        
        # array lengths (*2, one for precision one for recall)
        n_estimations = len(boundary) * len(classes)
                
        # initialize empty array
        precision = np.zeros(n_estimations)
        recall = np.zeros(n_estimations)
        class_list = np.zeros(n_estimations)
        decision_boundary = np.zeros(n_estimations)
        
        # array indexs
        index = np.arange(0, n_estimations, len(boundary))
        # iterate through the classes
        for i in range(0, len(classes)):
            n_true_cl = len(y_test[y_test == classes[i]])
            
            # list where current predicted is current class
            pred_class = predicted_class[predicted_class == classes[i]]
            # probability given by model of that class
            pred_prob = max_prob[predicted_class == classes[i]]
            # true y
            y_true = y_test[predicted_class == classes[i]]
            
            # iterate through the boundaries
            for k in range(0, len(boundary)):
                ind = index[i] + k                
                
                # update class 
                class_list[ind] = classes[i]
                
                # decision boundary
                decision_boundary[ind] = boundary[k]
                
                # only classify those above the decision bound
                pr_class = pred_class[pred_prob > boundary[k]]
                true_class = y_true[pred_prob > boundary[k]]
                
                precision[ind] = (pr_class == true_class).sum() / len(pr_class)
                recall[ind] = (pr_class == true_class).sum() / n_true_cl
                
        # lets double up and put precision and recall in single array
        # add a column for precision/recall str
        # makes tidy data for seaborn facet later
        data = {'label' : np.concatenate((class_list, class_list)),
                'decision_boundary' : np.concatenate((decision_boundary, decision_boundary)),
                'scores' : np.concatenate((precision, recall)),
                'score_type' : np.array(['precision', 'recall']).repeat(n_estimations)}
        
        score_df = pd.DataFrame(data=data)
        
        return(score_df)
    
    def predictions(self, X=None, decision_boundary=None):
        '''
        Return predictions if the probability is higher than the decision
        boundary, otherwise return nan for that samples. NANs will be
        reviewed by experimenter for classification
        '''
        # get probability array, probability of class (cols) for each
        # training sample (rows), and max value
        prob_array = self.classifier.predict_proba(X)
        max_prob = np.amax(prob_array, axis = 1)
               
        # class prediction is selected by highest probability
        predicted_class = self.classifier.predict(X)
        predicted_class = predicted_class.astype('float')
        
        for i in range(0, len(decision_boundary)):
            temp = predicted_class[predicted_class == i]
            temp[max_prob[predicted_class == i] < decision_boundary[i]] = np.nan
            predicted_class[predicted_class == i] = temp
        
        return(predicted_class)
        