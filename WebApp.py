# Description: This program detects if someone has diabetes


# Imports
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st 


# Create a title and a sub-title 
st.write("""
# Diabetes Detection
Detect if someone has diabetes 
""")

# Open and display an Image 

image = Image.open('C:/Users/Bernardo/Desktop/diabetes.jpg')
st.image(image, caption='', use_column_width=True)

# Get the data
df = pd.read_csv('C:/Users/Bernardo/Desktop/py projects/diabetes.csv')

# Set a subheader
st.subheader('Information: ')

# Show the data as a table 
st.dataframe(df)

# Set a subheader
st.subheader('Statistics: ')

# Show the statistics on the data
st.write(df.describe())

# Set a subheader
st.subheader('Bar Chart: ')

# show a data as a chart
chart = st.bar_chart(df)

# Split the data into independent and dependent variables (x and y)
x = df.iloc[:, 0:8].values
y = df.iloc[:, -1].values

# Split the data set into 70% training and 30% testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Get the feature input from the user 
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'BMI': BMI,
                'DPF': DPF,
                'age': age}

    # Transform the data into a dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store he user input into a variable
user_input = get_user_input()

# Set a subheader and display the user input 
st.subheader('User Input: ')
st.write(user_input)

# Create and train the model 
RandomForestClassifier = RandomForestClassifier() 
RandomForestClassifier.fit(x_train, y_train)

# Show the models metrics 
st.subheader('Model Test Accuracy Score: ')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%')

# Store the models predictions in a variable 
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader an display the classification
st.subheader('Classification: ')
st.write(prediction)

