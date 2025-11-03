Machine Learning Model for Precision Water Requirement Prediction in Irrigation

1. Description
This is the Machine Learning Project for the Water Requirement Prediction for making the irrigation systems smart. The aim of this project is to find that is to predict the optimal amount of water required or needed by the crops based on the real time environmental conditions.

This project addresses two significant challenges:
a. Optimization of the crop yield: This system takes care that right amount of water is given at the right time to the crops, therefore improving and enhancing the health of the crop yield.


b. Scarcity of Water: This optimal irrigation system minimizes the wastage of excess water, leading to conservation of water, which was otherwise wasted in the traditional irrigation methods.

2. Dataset Used (Source of the dataset)
This project uses two dataset, one is synthetic dataset and the other is real-world dataset taken from kaggle (Smart_irrigation_dataset.csv).

work on datasets:
a. The models uses the real-world dataset for the prediction of the continuous value of irrigation_amount_m3.
b. The categorical variables were converted into numerical values.
c. Standardization of the column heads was done.

3. Methods used
Two models were used for the predcition. Random forest and the second one is Decision tree. The Random forest model was used as the ensemble leader to improve the accuracy of the prediction. Decision tree was used as a baseline model.

4.Results 

Performance Table

Sr.No.      Dataset     Model       MAE     RMSE        R2
1           synthetic   RF          3.3902  5.4874      0.7252
2           synthetic   DT          4.0839  7.2875      0.5154
3           Real World  RF          155.80  215.87      0.3861
4           Real World  DT          202.05  305.32      -0.2279

5. Conclusion
This project was successfuly to implement the Machine Learning Model For precision irrigation.
The model implements two Machine learning model that is decision tree and random forest and the negative r2 value demonstrates effectively that the decision tree model is not good enough for agricultural predictions.

6. Steps to run the code
a. Clone the repository: git clone [your repo url]
b. Install the dependencies: pip install -r requirements.txt
c. Run the application: streamlit run app.py

7. References
smart_irrigation_dataset.csv ([text](https://www.kaggle.com/datasets/mahiruddin/smart-irrigation-dataset))

8. Live streamlit app url here: [text](https://smart-irrigation-ml-project-hfnr4eeubnn9b22ukdyhsj.streamlit.app/)