# %%
from sklearn import svm
from Preprocessing import preprocess
import numpy as np
from utils import *
import itertools as it
from datetime import datetime
import copy
from sklearn.naive_bayes import MultinomialNB


def enforce_equal_opportunity(categorical_results, epsilon):
    #print("Hello")
    thresholds = {}
    equal_opportunity_data = {}

    keys = []
    for i in categorical_results.keys():
        keys.append(i)
    
    max_predictions = max([x[0] for x in categorical_results[keys[0]]])
    min_predictions = min([x[0] for x in categorical_results[keys[0]]])
    threshold = np.arange(min_predictions, max_predictions, 0.01)
    threshed = {}
    
    max_accuracy = 0
    
    for j in threshold:
        keys_0 = []
        thresholds_pre = {}
        threshed[keys[0]] = apply_threshold(categorical_results[keys[0]], j)
        accuracy = get_true_positive_rate(threshed[keys[0]])
        keys_0.append(str(j))
        thresholds_pre[keys[0]] = keys_0
        for i in keys[1:]: 
            keys_1 = []
            max_predictions = max([x[0] for x in categorical_results[i]])
            min_predictions = min([x[0] for x in categorical_results[i]])
            threshold_1 = np.arange(min_predictions, max_predictions, 0.01)
            for k in threshold_1:
                threshed[i] = apply_threshold(categorical_results[i], k)
                accuracy_1 = get_true_positive_rate(threshed[i])
                if accuracy-epsilon <= accuracy_1 <= accuracy+epsilon:
                    keys_1.append(str(k))
                    thresholds_pre[i] = keys_1
            if i not in thresholds_pre.keys():
                break
                
        if len(thresholds_pre) == len(keys):
            combinations = it.product(*(thresholds_pre[key] for key in keys))
            for combination in list(combinations):
                total_accuracy = 0
                threshed = {}
                thresholds_pre_1 = {}
                combination_1 = []
                for j in combination:
                    combination_1.append(float(j))
                for i in range(len(keys)):
                    key = keys[i]
                    thresholds_pre_1[key] = combination_1[i]
                    threshed[key] = apply_threshold(categorical_results[key], thresholds_pre_1[key])

                total_accuracy = get_total_accuracy(threshed)
                if total_accuracy > max_accuracy:
                    max_accuracy = total_accuracy
                    for i in keys:
                        thresholds[i] = thresholds_pre_1[i]
                        equal_opportunity_data[i] = threshed[i]

    return equal_opportunity_data, thresholds
# %%


def report_results(data):

    begin = datetime.now()
    print("Attempting to enforce equal opportunity...")
    equal_opportunity_data, equal_opportunity_thresholds = enforce_equal_opportunity(copy.deepcopy(data), 0.01)
    if equal_opportunity_data is not None:
        print("--------------------EQUAL OPPORTUNITY RESULTS--------------------")
        print("")
        for group in equal_opportunity_data.keys():
            accuracy = get_num_correct(equal_opportunity_data[group]) / len(equal_opportunity_data[group])
            print("Accuracy for " + group + ": " + str(accuracy))

        print("")
        for group in equal_opportunity_data.keys():
            FPR = get_false_positive_rate(equal_opportunity_data[group])
            print("FPR for " + group + ": " + str(FPR))

        print("")
        for group in equal_opportunity_data.keys():
            FNR = get_false_negative_rate(equal_opportunity_data[group])
            print("FNR for " + group + ": " + str(FNR))

        print("")
        for group in equal_opportunity_data.keys():
            TPR = get_true_positive_rate(equal_opportunity_data[group])
            print("TPR for " + group + ": " + str(TPR))

        print("")
        for group in equal_opportunity_data.keys():
            TNR = get_true_negative_rate(equal_opportunity_data[group])
            print("TNR for " + group + ": " + str(TNR))

        print("")
        for group in equal_opportunity_thresholds.keys():
            print("Threshold for " + group + ": " + str(equal_opportunity_thresholds[group]))

        print("")
        total_cost = apply_financials(equal_opportunity_data)
        print("Total cost: ")
        print('${:,.0f}'.format(total_cost))
        total_accuracy = get_total_accuracy(equal_opportunity_data)
        print("Total accuracy: " + str(total_accuracy))
        print("-----------------------------------------------------------------")
        print("")

        end = datetime.now()

        seconds = end-begin
        print("Postprocessing took approximately: " + str(seconds) + " seconds")


metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
#data, predictions, labels, categories, mappings = naive_bayes_classification(metrics)
#race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)

training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = enforce_equal_opportunity(training_race_cases, 0.01)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED

#report_results(training_race_cases)

print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")


print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

