# import packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st

# Read in the data, call the dataframe "s"  and check the dimensions of the dataframe
s = pd.read_csv("social_media_usage.csv") 
#s.shape


# ***

# #### Q2

# In[27]:


# Define a function called clean_sm that takes one input, x, and uses `np.where` to check 
# whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. 
def clean_sm(x):
    result = np.where(np.array([x]) == 1, 1, 0)
   # print(f"Input: {x}, Output: {result}")
    return result[0]    


# ***

# #### Q3

# In[28]:


# Create a new dataframe called "ss". The new dataframe should contain a target column 
# called sm_li which should be a binary variable ( that takes the value of 1 if it is 
# 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the 
# individual uses LinkedIn.

ss = s[['web1h', 'income','educ2', 'par', 'marital', 'gender', 'age']].copy(deep=True)
ss.rename(columns={'web1h': 'sm_li', 'educ2': 'education', 'par': 'parent', 'marital': 'married'}, inplace=True)

# With the following features: income (ordered numeric from 
# 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 
# considered missing), parent (binary), married (binary), female (binary), and age (numeric, 
# above 98 considered missing). Drop any missing values.

ss.loc[ss['income'] > 9, 'income'] = np.nan
ss.loc[ss['education'] >8, 'education'] = np.nan
ss.loc[ss['age']>98, 'age'] = np.nan

ss = ss.dropna()

ss['sm_li'] = ss['sm_li'].apply(clean_sm)
ss['parent'] = ss['parent'].apply(clean_sm)
ss['married'] = ss['married'].apply(clean_sm)
ss['gender'] = ss['gender'].apply(clean_sm)



# ***

# #### Q4

# In[29]:


# Create a target vector (y) and feature set (X)
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "gender", "age"]]


# ***

# #### Q5

# In[30]:


# Split the data into training and test sets. Hold out 20% of the data for testing. 
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# Explain what each new object contains and how it is used in machine learning

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# ***

# #### Q6

# In[31]:


# Instantiate a logistic regression model and set class_weight to balanced. 
lr = LogisticRegression(class_weight='balanced')

# Fit the model with the training data.
lr.fit(X_train, y_train)



# ***

# #### Q7

# In[33]:


# Evaluate the model using the testing data. What is the model accuracy for the model? 
# Use the model to make predictions and then generate a confusion matrix from the model. 
y_pred = lr.predict(X_test)


# ***

# ***

# #### Input

# In[40]:


# 
st.markdown("Weldcome to my app!")
st.markdown("Would you like to play a game...")
st.markdown("...what is the probability of being a LinkedIn user based on the following: ")
#select income options
i_income = 0
income_dd = ["0 -$9,999","$10,000 - $19,999", "$20,000 - $29,999","$30,000 - $39,999","$40,000 - $49,999", 
             "$50,000 - $74,999", "$75,000 - $99,999", "$100,000 - $149,999", "$150,000 +" ]
income_selected_option = st.selectbox("Choose an income:", income_dd)
if income_selected_option == "0 -$9,999":
    i_income = 1
elif income_selected_option == "$10,000 - $19,999":
    i_income = 2
elif income_selected_option == "$20,000 - $29,999":
    i_income = 3
elif income_selected_option == "$30,000 - $39,999":
    i_income = 4
elif income_selected_option == "$40,000 - $49,999":
    i_income = 5
elif income_selected_option == "$50,000 - $74,999":
    i_income = 6
elif income_selected_option == "$75,000 - $99,999":
    i_income = 7
elif income_selected_option == "$100,000 - $149,999":
    i_income = 8
elif income_selected_option == "$150,000 +":
    i_income = 9

#select education options
i_education = 0
education_dd = ["Less than high school","High school incomplete", "High school graduate","Some college, no degree",
                "Two-year associate degree from a college or university","Four-year college or university degree/Bachelors",
                "Some postgraduate or professional schooling, no postgraduate degree",
                "Postgraduate or professional degree, including masters,doctorate, medical or law degree" ]
education_selected_option = st.selectbox("Highest level of school/degree completed:", education_dd)
if education_selected_option == "Less than high school":
    i_education = 1
elif education_selected_option == "High school incomplete":
    i_education = 2
elif education_selected_option == "High school graduate":
    i_education = 3
elif education_selected_option == "Some college, no degree":
    i_education = 4
elif education_selected_option == "Two-year associate degree from a college or university":
    i_education = 5
elif education_selected_option == "Four-year college or university degree/Bachelors":
    i_education = 6
elif education_selected_option == "Some postgraduate or professional schooling, no postgraduate degree":
    i_education = 7
elif education_selected_option == "Postgraduate or professional degree, including masters,doctorate, medical or law degree":
    i_education = 8

#select parent options
i_parent = 0
parent_dd = ["Yes", "No" ]
parent_selected_option = st.selectbox("Do you have children:", parent_dd)
if parent_selected_option == "Yes":
    i_parent = 1
elif parent_selected_option == "No":
    i_parent = 2

    
#select married options
i_married = 0
married_dd = ["Yes", "No" ]
married_selected_option = st.selectbox("Are you married:", married_dd)
if married_selected_option == "Yes":
    i_married = 1
elif married_selected_option == "No":
    i_married = 2

#select gender options
i_gender = 0
gender_dd = ["Male", "Female" ]
gender_selected_option = st.selectbox("What is your gender:", gender_dd)
if gender_selected_option == "Male":
    i_gender = 1
elif gender_selected_option == "Female":
    i_gender = 2
    
#select age options
i_age = 0
age_selected_option = st.number_input("Please enter an age:", min_value=0, max_value=100, value=25, step=1)
i_age = age_selected_option


# #### Q10

# In[43]:


# Use the model to make predictions. For instance, what is the probability that a 
# high income (e.g. income=8), with a high level of education (e.g. 7), non-parent 
# who is married female and 42 years old uses LinkedIn? 
# How does the probability change 
# if another person is 82 years old, but otherwise the same?

# i_income = 8
# i_education = 7
# i_parent = 0
# i_married = 1
# i_gender = 0
# i_age = 67

newdata = pd.DataFrame({
    "income": [i_income],  # higher income
    "education": [i_education], # higher education
    "parent": [i_parent], # 0 for non-parent 
    "married": [i_married], # 1 for married
    "gender": [i_gender], # 0 for female
    "age": [i_age] # two different ages 
})

# print(newdata)

probabilities = lr.predict_proba(newdata)
linkedin_probabilities = probabilities[:, 1]

st.markdown(f"Probability of being a LinkedIn user based on the information provided is: {round(linkedin_probabilities[0], 2)}")
