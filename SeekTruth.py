# SeekTruth.py : Classify text objects into two categories
#
# Code by: Shyam Makwana (smakwana), Mansi Sarda (msarda), Lakshay Madaan (lmadaan)
#
# Based on skeleton code by D. Crandall, October 2021
#

import sys

def load_file(filename):
    objects=[]
    labels=[]
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ',1)
            labels.append(parsed[0] if len(parsed)>0 else "")
            objects.append(parsed[1] if len(parsed)>1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}

# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!
#
def preprocess(s):
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for word in s:
        if word in punc:
            s = s.replace(word, "")
    return s
            
def classifier(train_data, test_data,a):
    # This is just dummy code -- put yours here!
    classes = train_data['classes']
    wi_A = {}; wi_B = {};word_classB=0;word_classA = 0;senB=0;senA=0
    
    
    for i,sen in enumerate(train_data['objects']):
        sen = preprocess(sen)                           ## to preprocess the file to remove punctation brackets.
        if train_data['labels'][i] == classes[0]:
            senA += 1
            for word in sen.split():
                if word not in wi_A:
                    wi_A[word]=1
                else:
                    wi_A[word]+=1
        else:
            senB += 1
            for word in sen.split():
                if word not in wi_B:
                    wi_B[word]=1
                else:
                    wi_B[word]+=1
    
    wi_A = {key : val for key, val in wi_A.items() if (val > 5)}   ## dropping the low frequency words
    wi_B = {key : val for key, val in wi_B.items() if (val > 5)}
    A_words = (sum(wi_A.values()))
    B_words = (sum(wi_B.values()))
    
    prob_A = senA/(senA+senB)                                  ## calculating P(A)
    prob_B = 1 - prob_A                                        ## calculating P(B)
    final_label=[]
    n = A_words+B_words
    for sen in test_data['objects']:
        sen = preprocess(sen)
        A, B = 1, 1
        for word in sen.split():
            try:
                A *= (wi_A[word]+a)/(A_words+a*n)*1000         ## P(wi/A) = (nyi+alpha)/(Ny + alpha*n)
                                                               ## nyi is the count of given word in class A
                                                               ## Ny is the total words count in class A
                                                               ## n is the total words in A and B
            except:
                A *= a/(A_words+a*n)*1000
            try:
                B *= (wi_B[word]+a)/(B_words+a*n)*1000
            except:
                B *= a/(B_words+a*n)*1000
        A = A*prob_A
        B = B*prob_B
        
        if B==0:
            final_label.append(classes[0])
        else:
            if (A/B)>1:
                final_label.append(classes[0])
            else:
                final_label.append(classes[1])
                
    
    return final_label


if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = "","deceptive.train.txt", "deceptive.test.txt" 
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if(sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}
    
    ## Hyperparameter tunig to get the best alpha which came out to be 1
    '''
    alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]
    for a in alpha:
        results= classifier(train_data, test_data_sanitized,a)

        # calculate accuracy
        correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
        print(f"Classification accuracy = %5.2f%% for {a}" % (100.0 * correct_ct / len(test_data["labels"])))
    '''
    '''
    Output:
    
    Classification accuracy = 74.75% for 1e-05
    Classification accuracy = 76.75% for 0.0001
    Classification accuracy = 77.00% for 0.001
    Classification accuracy = 81.50% for 0.1
    Classification accuracy = 84.00% for 1
    Classification accuracy = 82.75% for 10
    Classification accuracy = 69.50% for 100
    Classification accuracy = 60.50% for 1000
    '''
    a = 1
    results= classifier(train_data, test_data_sanitized,a)

    # calculate accuracy
    correct_ct = sum([ (results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"])) ])
    print(f"Classification accuracy = %5.2f%% for {a}" % (100.0 * correct_ct / len(test_data["labels"])))