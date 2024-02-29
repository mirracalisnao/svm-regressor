import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Define the Streamlit app
def app():
    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    st.title('Predicting Housing Cost using the SVM Regressor')
    # Use session state to track the current form
    if "current_form" not in st.session_state:
        st.session_state["current_form"] = 1    

    if "scaler" not in st.session_state:
        st.session_state["scaler"] = StandardScaler()

    if "svm_reg" not in st.session_state:
        st.session_state["svm_reg"] = SVR()
    
    if "input_array" not in st.session_state:
        st.session_state['input_array'] = [[]]

    # Display the appropriate form based on the current form state
    if st.session_state["current_form"] == 1:
        display_form1()
    elif st.session_state["current_form"] == 2:
        display_form2()
    elif st.session_state["current_form"] == 3:
        display_form3()

def display_form1():
    st.session_state["current_form"] = 1
    form1 = st.form("intro")
    form1.subheader('About the Classifier')
    form1.write("""
        (c) 2024 Louie F. Cervantes
        Department of Computer Science
        College of Information and Communications Technology
        West Visayas state University
    """)
                
    form1.write('Replace with the actual description')        
    #insert the rest of the information here

    submit1 = form1.form_submit_button("Start")

    if submit1:
        form1 = [];
        # Go to the next form        
        display_form2()

def display_form2():
    st.session_state["current_form"] = 2
    form2 = st.form("training")
    form2.subheader('Classifier Training')        

    # Load the California housing data
    data = fetch_california_housing()

    # Convert data features to a DataFrame
    feature_names = data.feature_names
    df = pd.DataFrame(data.data, columns=feature_names)
    df['target'] = data.target
    
    form2.write('The housing dataset')
    form2.write(df)


    submit2 = form2.form_submit_button("Train")
    if submit2:        
        # Separate features and target variable
        X = df.drop('target', axis=1)  # Target variable column name
        y = df['target']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features using StandardScaler (recommended)
        scaler = st.session_state["scaler"] 
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state["scaler"] = scaler

        #This part uses the linear regressor
        #from sklearn.linear_model import LinearRegression
        #svm_reg = LinearRegression()
        #svm_reg.fit(X_train_scaled, y_train)

        # Create and train the SVM regressor     
        svm_reg = st.session_state["svm_reg"]
        svm_reg.fit(X_train_scaled, y_train)
        st.session_state["svm_reg"] = svm_reg

        # Make predictions on the test set
        y_test_pred = st.session_state["svm_reg"].predict(X_test_scaled)

        # Evaluate performance using appropriate metrics (e.g., mean squared error, R-squared)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)

        form2.text("Mean squared error: " + f"{mse:,.2f}")
        form2.text("R-squared: " + f"{r2:,.2f}")

        # Create a figure and an axes object
        fig, ax = plt.subplots()

        # Scatter plot using the axes object
        ax.scatter(y_test, y_test_pred, s=5)

        # Set labels and title using the axes object
        ax.set_xlabel("Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Real vs Predicted Housing Prices")

        # Display the plot
        form2.pyplot(fig)

        display_form3()

def display_form3():    
    st.session_state["current_form"] = 3
    form3 = st.form("prediction")
    form3.subheader('Prediction')
    form3.text('The trained model can now predict the property value based on the user inputs.')

    #initialize the slider variables
    if "medinc" not in st.session_state:        
        st.session_state['medinc'] = 0
    if "houseage" not in st.session_state:
        st.session_state['houseage'] = 0
    if "averooms" not in st.session_state:
        st.session_state['averooms'] = 0
    if "avebedrms" not in st.session_state:
        st.session_state['avebedrms'] = 0
    if "population" not in st.session_state:
        st.session_state['population'] = 0
    if "aveoccup" not in st.session_state:
        st.session_state['aveoccup'] = 0
    if "latitude" not in st.session_state:
        st.session_state['latitude'] = 0
    if "longitude" not in st.session_state:
        st.session_state['longitude'] = 0

    medinc = form3.slider(
        label="MedInc (Median Income):",
        min_value=0.49,
        max_value=15.0,
        step=0.1,
        value=5.0,  # Initial value
        on_change=update_values(),
        key="medinc"
    )

    houseage = form3.slider(
        label="HouseAge (Median Age):",
        min_value=1,
        max_value=52,
        step=1,
        value=21,  # Initial value
        on_change=update_values(),
        key="houseage"

    )

    averooms = form3.slider(
        label="AveRooms (Average Rooms):",
        min_value=1,
        max_value=141,
        step=1,
        value=10,  # Initial value
        on_change=update_values(),
        key="averooms"
    )

    avebedrms = form3.slider(
        label="AveBedrms (Average Bedrooms):",
        min_value=1,
        max_value=141,
        step=1,
        value=10,  # Initial value
        on_change=update_values(),
        key="avebedrms"

    )

    population = form3.slider(
        label="Population:",
        min_value=3,
        max_value=35000,
        step=1,
        value=100,  # Initial value
        on_change=update_values(),
        key="population"

    )

    aveoccup = form3.slider(
        label="AveOccup (Average Occupancy):",
        min_value=1,
        max_value=1243,
        step=1,
        value=100,  # Initial value
        on_change=update_values(),
        key="aveoccup"

    )

    latitude = form3.slider(
        label="Latitude:",
        min_value=32.5,
        max_value=41.95,
        step=0.1,
        value=37.0,  # Initial value
        on_change=update_values(),
        key="latitude"

    )

    longitude = form3.slider(
        label="Longitude:",
        min_value=-124.5,
        max_value=-114.0,
        step=0.1,
        value=-120.0,  # Initial value
        on_change=update_values(),
        key="longitude"
    )

    form3.text("Click the Predict button to generate the predicted price.")

    predictbn = form3.form_submit_button("Predict")    
    if predictbn:     
        testdata = st.session_state['input_array']
        scaler = st.session_state["scaler"]
        test_data_scaled =scaler.transform(testdata)

        test_data_scaled = np.array(test_data_scaled)

        form3.text('Test data = ' + str(testdata))
        form3.text('Test data scaled = ' + str(test_data_scaled))

        predicted =  st.session_state["svm_reg"].predict(test_data_scaled)
        predvalue = predicted * 100000
        form3.subheader("Predicted Property Value = $ " + f"{predvalue[0]:,.2f}")
    
    submit3 = form3.form_submit_button("Reset")
    if submit3:
        st.session_state.reset_app = True
        st.session_state.clear()

def update_values():
    """Get the updated values from the sliders."""
    medinc = st.session_state['medinc']
    houseage = st.session_state['houseage']
    averooms = st.session_state['averooms']
    avebedrms = st.session_state['avebedrms']
    population = st.session_state['population']
    aveoccup = st.session_state['aveoccup']
    latitude = st.session_state['latitude']
    longitude = st.session_state['longitude']
    
    #update the input array
    st.session_state['input_array'] = [[medinc, houseage, averooms, avebedrms, 
                    population, aveoccup, latitude, longitude]]
    
if __name__ == "__main__":
    app()
