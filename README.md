# Feature Selection with Nearest Neighbor Classifier

## Introduction

This project applies **supervised learning** to the **classification problem**, using previously labeled data to classify new, unseen instances.
Specifically, this project implements the **Nearest Neighbor Classifier** with **Euclidean distance** and performs **feature selection** using two greedy search methods:
- **Forward Selection**
- **Backward Elimination**

Both methods aim to find the subset of features that maximizes classification accuracy under **leave-one-out cross-validation**.

---

## Algorithm Description

### **Nearest Neighbor Classification**
- Each instance is classified based on the label of its **closest** neighbor in feature space.  
- Distance is computed using the **Euclidean formula**.  
- The model construction requires no training time, but classification has **O(n)** complexity since each instance must be compared against all others.

---

## Feature Selection Methods

### **1. Forward Selection**
- Starts with no features (baseline is the default rate).  
- Adds one feature at a time, choosing the one that most improves accuracy.  
- Uses greedy search to approximate the best subset.

*Small Dataset:*  
- Best subset: **{6, 5, 2}**  
- Accuracy: **94.8%**  
- Default rate: **84%**
<img width="420" alt="SmallForward" src="https://github.com/user-attachments/assets/78a133de-02be-49eb-8e6d-c0477e16bcf6" />


<br />*Large Dataset:*  
- Best subset: **{7, 14}**  
- Accuracy: **96.4%**  
- Default rate: **83%**
<img width="420" alt="LargeForward" src="https://github.com/user-attachments/assets/25ebcc5a-21eb-4943-8329-e966e8b53cc1" />

<br />
Accuracy rises with the inclusion of relevant features and drops again once irrelevant ones are added.  
For example, feature 2 in the small dataset only increased accuracy by 0.4%, suggesting marginal significance.

---

### **2. Backward Elimination**
- Begins with all features.  
- Removes one feature at a time, eliminating those whose removal least harms accuracy.  
- More effective for highly correlated data, since it evaluates in full feature space first.

*Small Dataset:*  
- Best subset: **{6, 5, 2}** (same as forward selection)  
- Accuracy: **94.8%**
<img width="420" alt="SmallBackward" src="https://github.com/user-attachments/assets/b546716f-83e9-4f07-815e-dfb33b6eb0bb" />

<br />*Large Dataset:*  
- Best subset returned by the search had **18 features** with **78.2% accuracy**, never surpassing the default rate (83%).  
- Key features (7 and 14) were removed early, leading to suboptimal results.
<img width="420" alt="LargeBackward" src="https://github.com/user-attachments/assets/9eadc310-c340-44d7-964b-285f5cb85798" />

<br />
Small fluctuations can steer the algorithm away from the optimal path, which is the main disadvantage of greedy search.

---

## Example Program Execution

Below is a trace from the **Forward Selection** algorithm on the small dataset.

<pre>
Welcome to the Feature Selection Algorithm.
Type in the name of the file to test: CS170_Small_Data_87.txt
Type the number of the algorithm you want to run.

1) Forward Selection
2) Backward Elimination

This dataset has 6 features (not including the class attribute), with 500 instances.
Running nearest neighbor with all 6 features, using "leave-one-out" evaluation, I get an accuracy of 84.4%

Beginning search.
Using feature(s) {1} accuracy is 72.6%
Using feature(s) {2} accuracy is 72.4%
Using feature(s) {3} accuracy is 74.8%
Using feature(s) {4} accuracy is 70.8%
Using feature(s) {5} accuracy is 75.2%
Using feature(s) {6} accuracy is 87.6%

Feature set {6} was best, accuracy is 87.6%
...
Feature set {6, 5, 2} was best, accuracy is 94.8%
Finished search! The best feature subset is {6, 5, 2}, which has an accuracy of 94.8%
</pre>

---
## Performance Summary

| Dataset          | Method                         | Best Subset      | Accuracy   | Default Rate  | Runtime  |
|------------------|--------------------------------|------------------|------------|---------------|-----------
| Small Dataset    | Forward Selection              | {6, 5, 2}        | 94.8%      | 84%           | 2.88 s   |
| Small Dataset    | Backward Elimination           | {6, 5, 2}        | 94.8%      | 84%           | 3.17 s   |
| Large Dataset    | Forward Selection              | {7, 14}          | 96.4%      | 83%           | 72.2 m   |
| Large Dataset    | Backward Elimination           | 18 features      | 78.2%      | 83%           | 89.6 m   |


---
## Implementation Details

**Language** C++ <br />
**Key Components**
- euclidean_distance(): computes distance between two vectors
- forward_selection(): greedy addition of features
- backward_elimination(): greedy removal of features
- leave_one_out_cross_validation(): evaluates accuracy
- default_rate(): computes the baseline accuracy
- hide_features(): temporarily prunes unused features

---
## Acknowledgments
This project was part of CS170 taught by professor Eamonn Keogh. Project guidelines and datasets were provided by the course instructor.<br />
References to course material:
- Eamonn Keogh. Machine Learning 001, 2025.
- Eamonn Keogh. Machine Learning 002, 2025.
- Eamonn Keogh. Project 2 Briefing Video, 2025.
