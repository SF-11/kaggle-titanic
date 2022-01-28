# kaggle-titanic

https://www.kaggle.com/c/titanic/overview


Current implementation leverages an sklearn decision tree classifier, with the following confusion matrix and classification report for the test data:
```
[[56  1]
 [12 21]]
              precision    recall  f1-score   support

           0       0.82      0.98      0.90        57
           1       0.95      0.64      0.76        33

    accuracy                           0.86        90
   macro avg       0.89      0.81      0.83        90
weighted avg       0.87      0.86      0.85        90
```

The submission scored ~78%. Currently, NaN values are just treated as 0s because the DecisionTreeClassifier isn't able to handle NaN.
