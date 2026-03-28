Loan Prediction App

This project is part of my learning journey in machine learning and deployment. It predicts whether a loan application will be approved based on applicant details such as income, dependents, loan amount, and CIBIL score. I wanted to go beyond training a model in Jupyter Notebook, so I built a Streamlit app to make the predictions interactive and easy to use.

Why this matters
Loan approvals are important in real life. Banks use models and rules to decide who qualifies. By recreating a simplified version, I learned how data, features, and algorithms come together to make decisions that impact people’s lives.

What you will find here

app.py: Streamlit app for loan prediction

loan_prediction_model.pkl: Trained Random Forest model

loan_approval_dataset_ml.ipynb: Jupyter Notebook with data exploration, training, and evaluation

requirements.txt: Dependencies needed to run the app

How to run it

Clone the repository:
git clone https://github.com/janudhanyavinothsingh-bit/loan-prediction-app.git (github.com in Bing)  
cd loan-prediction-app

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

Dataset
The dataset includes applicant demographics, income, loan details, and asset values. It was used to train a Random Forest model that balances accuracy with interpretability.

Deployment
This app can be deployed on Streamlit Cloud so anyone can try it without installing Python. That is my next step — making it accessible with just a link.

Author
Janu Dhanya Vinoth Singh
Exploring machine learning, deployment, and how to turn notebooks into real apps.
