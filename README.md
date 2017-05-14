# Merchant Fraud Detection: Exploring Imbalanced Classification Techniques  
This is a one off project with a presentation to try and improve the _recall_ scores of a classifier by exploring Cost Based optimizations. There is also some data exploration to try and extract meaningful features for the prediction. Most of the research exists in Jupyter Notebooks linked below:  
  
* Overall [Project Presentation](./visuals/fraud_detection.pdf)  
* [Feature Exploration](./Data_Explore/readme.md) where I create an address similarity signature for each merchant as one of the features using the _Jaccard Similarity_ score.  
* [Cost Based Learning - Tuning the Decision Threshold](./CostBasedLearning_TuningDecisionThreshold/readme.md) to improve classifier _Recall_.  
* [Cost Based Learning - Balanced Subsampling](./CostBasedLearning_BalancedSubsampling/readme.md) to improve classifier _Recall_ and compare with the previous method.  
* To-dos for future iterations:  
    * Explore synthetic oversampling (SMOTE)  
    * Use other indicators of imbalance such as _Kohen's Kappa_  
    * Incorporate a feature that captures the prior probability of Fraud at certain transaction amounts.  
