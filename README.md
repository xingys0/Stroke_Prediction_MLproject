# Stroke Prediction🧠:EDA 📊|Random Forest&KNN🔄|XGBoost🚀|
               OVERVIEW

       Part1 :   Exploratory Data Analysis(EDA)

       Part2 :  Data Collection&Pre-processing 

       Part3 :  Model Selection and Implementation

       Part4 :  Model Evaluation and Interpretation

![image](https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/e9616415-e552-4344-9f19-d9b7b0c3afa9)






<img width="500" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/bfc40610-e666-4015-bc87-4d96fad177ff">

Target variable: Stroke
Important Feature: highly imbalanced with far more instances of class 0 (no stroke) than class 1 (stroke).

<img width="500" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/f4ace693-2c71-4c66-8c81-cb21e2c80da6">

CategoryVairables:  gender, hypertension, heart_disease, ever_married, work_type.ect 
Important Features:1. various distributions;
2. AGE,IBM,Avg_Glucose_Level have more noticeable impact 

<img width="572" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/1a8636ee-f3a6-435a-a693-2bb530d67b82">

continuous variable：
1.age: The age of the patients varies from young to old, with the majority of patients being in the range of 40-80 years. 
2.avg_glucose_level: Most patients have an average glucose level in the range of 50-125, but there are also many patients with higher levels. The distribution is right-skewed. 
3.bmi: The majority of patients have a BMI in the range of 20-40, which is considered normal to overweight. There are some outliers with extremely high BMI values.

                                          Summary:

Imbalanced Target Distribution: The prediction of strokes shows a significant imbalance. This points towards the necessity of adopting algorithms adept at handling imbalanced data. Random Forest emerges as a pertinent choice given its proficiency in adjusting class weights effectively to address such imbalances.
Complexity with Categorical Variables: The dataset possesses multiple categorical variables. For such scenarios, tree-based algorithms like Random Forest and XGBoost exhibit a clear edge.
Non-linear Relationships: Further exploration reveals that the relationship between the features and the incidence of strokes is non-linear. This underscores the need for non-linear models.

Through the analysis of the datset, we have identified the following characteristics: complex relationships, multidimensional features, nonlinear associations, binary classification. As a result, it is suitable to use models such as Random Forest, K-Nearest Neighbors (KNN), and XGBoost for stroke prediction modeling. These models can handle complex relationships, multi-feature inputs, nonlinear patterns, and are well-suited for binary classification tasks, contributing to accurate stroke prediction.

Lastly, considering that this research pertains to the medical domain, the interpretability of the model becomes paramount. I propose to utilize the LIME tool to delve deep into and elucidate the predictions of the top-performing model, ensuring transparency and reliability in its forecasts.

 ![image](https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/9c7a9899-8fe8-455d-803a-607975012c67)

 <img width="521" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/a74d0797-3539-4ee1-abf8-a3158c30ccba">

 The original data underwent a series of preprocessing steps, which are crucial for the training and evaluation of machine learning models.

 <img width="559" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/e5512118-e15e-4443-a71f-e15291af53c1">

According to data analysis, the avg_glucose_level shows the strongest relationship with stroke, followed by age and BMI. These three indicators are the most important factors in predicting the risk of stroke.

![image](https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/ca917b8d-17ea-4716-ac22-8132e0fd3545)
 

                                         3.1 Prediction with Random Forest Model

 
<img width="600" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/01efee45-a08c-4904-a85e-08654e7fd0ba">

The model's overall performance is commendable, boasting high accuracy and recall. However, given that the precision is somewhat lower than the other metrics, further investigation might be required to determine if the model occasionally produces false positives (i.e., misclassifying negative samples as positive).

<img width="422" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/d367be1c-d797-4dff-bf83-97517d032d58">

Through GridSearchCV, we've obtained an optimized combination of hyperparameters that offer the best performance for our model. Overall, the model performs well on the training set and we can say that this model is highly reliable in predicting the risk of stroke.

                                      Bar Chart& ROC Curve comparison about the metric 

<img width="602" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/d3465bda-0aac-417c-87bb-4a06f6087750">

  1.The optimized Random Forest model exhibits improved performance in precision without compromising much on other metrics. This fine-tuning has enabled the model to make more accurate positive predictions.
  2.While the accuracy and recall witnessed minor improvements, the considerable boost in precision without a significant drop in the F1 score is noteworthy.
  3.The optimized hyperparameters, particularly the increase in n_estimators and the conditions for splitting and leaf nodes, contributed to this enhanced performance.
  In summary, the hyperparameter optimization process has refined the Random Forest model, making it more precise and slightly improving its overall prediction capabilities. 

                                               3.2 prediction with KNN
 <img width="600" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/e3031503-fd49-40bc-9bd5-a9c78b0b4047">

 From these metrics, we can see that the overall performance of the model is quite good, with accuracy and recall exceeding 90%. However, precision is relatively low, meaning that the model may produce more false positives, which could reduce the model's usability in some application scenarios. Therefore, depending on the specific needs of the application, we may need to further optimize the model. 

 <img width="600" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/976f3651-214d-4ea2-9f85-39410a5308ee">

 Through GridSearchCV, we've obtained the best performance for our model. Overall, the model performs well on the training set and we can say that this model is highly reliable in predicting the risk of stroke.

                                                ROC curve comparison

 <img width="633" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/a2fdc4a6-1ae1-490e-baeb-3667dc957fa4">


![image](https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/a3fab290-b3f6-4e2f-b7a4-90204a1c91d0)

                                              3.3 prediction with XGBoost

<img width="600" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/a5f5f4e9-4f18-4249-afb0-5098bdf70bda">

From these metrics, we can see that the overall performance of the model is quite good, with accuracy and recall exceeding 90%. 

<img width="455" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/f8898b76-660a-4a92-8ace-018d1d79011b">


Through GridSearchCV, we've obtained the best performance for our model. Overall, the model performs well on the training set and we can say that this model is highly reliable in predicting the risk of stroke.

                            Comparison of evaluation metrics before and after optimization


<img width="700" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/82ac2766-b699-427e-92c0-31ca00807c10">

The tuned XGBoost model improved on accuracy, precision, and recall, which means it became better at predicting strokes, could identify more individuals who truly had strokes, and reduced false alarms. Therefore, the model, after being tuned, is more likely to provide better predictive results and corresponding treatment suggestions for patients in real-life scenarios.
![image](https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/66aa20e5-b447-40dc-99dc-b7c7fcdef04d)


                            The results of the three models are compared visually with the bar chart


<img width="600" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/a316c48b-158f-46d1-bc78-76e6a2469c33">

                            The results of the three models are compared visually with the ROC curves
<img width="600" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/716f2085-7dc3-4abe-a5c9-8c2f884fd624">

                                       Summary: 
1.The AUC for the XGBoost model is 0.83. This means that the model has a high accuracy in predicting stroke risk, with an 83% chance of correctly distinguishing between stroke patients and non-stroke patients. This performance is much better than random guessing, so we can consider the XGBoost model to have the best performance among the three models in predicting stroke risk.
2.The AUC for the Random Forest model is 0.76, meaning that its accuracy in predicting stroke risk is slightly lower than that of XGBoost, but still better than random guessing. It has a 76% chance of correctly distinguishing between stroke patients and non-stroke patients.
3.The AUC for the KNN model is 0.64, which means that the model's performance in predicting stroke risk is relatively poor. Although it still performs better than random guessing, it has only a 64% chance of correctly distinguishing between stroke patients and non-stroke patients.

 Overall, among these three models, the XGBoost model performs the best.Therefore, if we had to choose one of these three models to predict stroke risk, the XGBoost model should be chosen.


                      Implement LIME model interpretation of XGBoost model


 <img width="562" alt="image" src="https://github.com/xingys0/Stroke_Prediction_MLproject/assets/130510998/924c0fc7-2f85-44d6-af4c-72e1d72a6a69">




                                             Conclusion

 After analyzing this dataset and applying different prediction models, we can arrive at the following conclusions:
1.Data Feature Analysis: Age, average glucose level, and BMI are critical factors in predicting stroke risk. Typically, stroke patients tend to be older and have higher glucose levels. This is consistent with our earlier findings that these three indicators are strongly related to stroke risk.
2.Model Selection: Among the three models we used, the XGBoost model performs the best. It has higher accuracy and recall rates, and while its precision is slightly lower, it's still acceptable. The XGBoost model has an AUC of 0.83, indicating that it has high accuracy in predicting stroke risk.
3. Model Optimization: Through GridSearchCV, we found a set of optimal hyperparameters that further enhance the model's performance. The optimized model improved in terms of precision without sacrificing other metrics.
4. Model Interpretability: The optimized XGBoost model not only improved prediction results but also allowed us to better interpret the model's prediction process. This is of great practical value in explaining prediction results and their potential health impacts to patients in real-world applications.
In summary, this analysis underscores the importance of age, glucose levels, and BMI in predicting stroke risk and provides an effective model (i.e., the optimized XGBoost model) for predicting stroke risk. This model has high accuracy in prediction results and can provide useful references for practical medical applications.






















 

 



 





                                               























