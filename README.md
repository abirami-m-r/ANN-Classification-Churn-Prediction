## ANN Implementation

**Classification**: Predict the 'Exited' column based on the features related to a customer like - Has Creditcard, EstimatedSalary, Age, CreditScore, balance etc.

Dataset: Churn_Modelling.csv

Feature Engineering: Convert categorical columns like Gender and Geography using LabelEncoder, OneHotEncoder to numerical values.
     Use StandardScaler and standardise all the values.

Create an ANN model using Sequential, Dense and EarlyStopping with input layer, 2 Hidden layers(ReLu activation functions) and one output layer(Sigmoid) with adam optimiser and 
binary_crossentropy Loss.

Perform Hyperparameter tuning using KerasClassifier for params like = neurons, Layers and epochs.

Use tensorBoard Viz to view the performance of the model during training.

Create app.py and run Streamlit to get the input from the user(from a webpage) and display the prediction output.


##Change is the activation function at op layer of ANN and loss function, Rest are same as Classification.

**Regression**: Predict the 'EstimatedSalary' column based on the features related to a customer like - Has Creditcard, Age, CreditScore, balance etc.

Feature Engineering: Convert categorical columns like Gender and Geography using LabelEncoder, OneHotEncoder to numerical values.
     Use StandardScaler and standardise all the values.

Create an ANN model using Sequential, Dense and EarlyStopping with input layer, 2 Hidden layers(ReLu activation functions) and one output layer(Linear Activation) with adam optimiser and 
mean_absolute_error Loss.

Use tensorBoard Viz to view the performance of the model during training.

Create regression_streamlit.py and run Streamlit to get the input from the user(from a webpage) and display the prediction output.
