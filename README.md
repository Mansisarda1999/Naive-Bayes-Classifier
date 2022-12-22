# Naive-Bayes-Classifier

**Description:** For a given textual object D consisting of words w1, w2, w3,..., wn, we need to evaluate whether it belongs to class A or class B by computing the odds and comparing it to a threshold using the Bayesian classifier.

**Various functions and steps that we have used and considered to solve this problem are:**

**Data Cleaning:** To make the use of meaningful words for our Bayesian Classifier, we have removed unnecessary punctuation marks : [!()-[]{};:'"\,<>./?@#$%^&*_~]
The words with a frequency of less than five has been dropped from the training data.

**Estimaters of the Classifier**
1. P(A) = Total sentences in Class A / Total sentences in the training data
2. P(B) = 1-P(A)
3. Priors:
    - P(wi/A) = frequency of wi in Class A /  total words in Class A  (the formula is updated to solve **Zero Frequency Problem** which is explained further in the report)
    - P(wi/B) = frequency of wi in Class B /  total words in Class B

<br/> Testing :
For each word in a given object we are computing : 
1. P(A|w1,w2,..., wn) = [P(w1|A)* P(w2|A)... ]* P(A) and 
2. P(B|w1,w2,..., wn) = [P(w1|B)* P(w2|B)... ]* P(B)
3. we are calculating the odds by dividing the value of P(A|w1,w2,..., wn) and P(B|w1,w2,..., wn)

**Optimization:**  To prevent the P(A|w1,w2,..., wn)/P(A|w1,w2,..., wn) becoming zero if a probability for a word is not there in the training data, we have modified the above formula:
<br/> P(wi/A) = (frequency of wi in Class A + alpha) /  (total words in Class A + alpha*n) * 1000
<br/> where, n is the total words in both the classes.
<br/> Similarly, for class B the probability is calculated.
<br/> We have multiplied the probability by 1000 because it was becoming too small. At the end since we are calculating odds 1000 will cancel out.

<br/> For the best results, we implemented **Hyperparameter Tuning** and tested our code with various alpha values for our Naive Bayes Model and found that using **Laplace smoothing** i.e. setting alpha = 1, gave us the best outcome.
