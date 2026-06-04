from flask import Flask, render_template, request
import pymysql
import pandas as pd
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Database connection
mydb = pymysql.connect(
    host="localhost",
    user="root",
    password="vinay@201",
    port=3306,
    database='phishing'
)

mycursor = mydb.cursor()
mycursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    password VARCHAR(255)
)
""")
mydb.commit()
print("Users table created successfully")


def executionquery(query, values):
    mycursor.execute(query, values)
    mydb.commit()

def retrivequery1(query, values):
    mycursor.execute(query, values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data

# Define the improved_roberts_similarity function for the TensorFlow model
def improved_roberts_similarity(x):
    """Enhanced version for time series data"""
    vertical_diff = x[:, 1:, :] - x[:, :-1, :]  # shape: (batch, time_steps-1, features)
    horizontal_diff = x[:, :, 1:] - x[:, :, :-1]  # shape: (batch, time_steps, features-1)
    horizontal_diff = tf.pad(horizontal_diff, [(0, 0), (0, 0), (0, 1)])
    horizontal_diff = horizontal_diff[:, :-1, :]  # shape: (batch, time_steps-1, features)
    magnitude = tf.sqrt(tf.square(vertical_diff) + tf.square(horizontal_diff))
    magnitude = tf.math.l2_normalize(magnitude, axis=1)
    return magnitude

# Load the TensorFlow model
try:
    print("Current folder:", os.getcwd())
    print("Files in folder:", os.listdir())
    model = tf.keras.models.load_model(
        'eeg_cnn_lstm_roberts.h5',
        custom_objects={'improved_roberts_similarity': improved_roberts_similarity}
    )
except Exception as e:
    print(f"Error loading TensorFlow model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/algorithm')
def algorithm():
    return render_template('algorithm.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = [i[0] for i in email_data]
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (email, password) VALUES (%s, %s)"
                values = (email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID already exists!")
        return render_template('register.html', message="Passwords do not match!")
    return render_template('register.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = [i[0] for i in email_data]
        if email.upper() in email_data_list:
            query = "SELECT password FROM users WHERE email = %s"
            values = (email,)
            password_data = retrivequery1(query, values)
            if password == password_data[0][0]:
                global user_email
                user_email = email
                return render_template('home.html')
            return render_template('login.html', message="Invalid Password!")
        return render_template('login.html', message="This email ID does not exist!")
    return render_template('login.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        try:
            # Define the feature order
            feature_order = [
                'entropy3_b', 'entropy0_a', 'entropy0_b', 'entropy3_a', 'min_q_7_b',
                'mean_3_a', 'mean_3_b', 'mean_d_3_b2', 'max_q_8_b', 'min_q_15_a',
                'mean_d_15_a', 'min_0_a', 'fft_91_b', 'min_q_12_a', 'min_q_5_a'
            ]

            # Collect input features
            input_features = {}
            for feature in feature_order:
                try:
                    input_features[feature] = float(request.form[feature])
                except (KeyError, ValueError):
                    return render_template('prediction.html', error_message=f"Invalid or missing value for {feature}")

            # Create a DataFrame to ensure correct feature order
            input_df = pd.DataFrame([input_features], columns=feature_order)

            # Convert to numpy array and reshape to (1, 15, 1)
            X_input = input_df.values
            X_input_3d = X_input.reshape(X_input.shape[0], X_input.shape[1], 1)

            # Make prediction
            
           
            y_pred_prob = model.predict(X_input_3d, verbose=0)
            print("Raw prediction:", y_pred_prob) 
            predicted_class = np.argmax(y_pred_prob, axis=1)[0]
            probabilities = y_pred_prob[0]
            confidence = probabilities[predicted_class] * 100

            # Map class index to label
            class_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
            predicted_label = class_names.get(predicted_class, "Unknown")

            # Format probabilities for display
            prob_display = {class_names[i]: f"{prob:.4f}" for i, prob in enumerate(probabilities)}

            return render_template('prediction.html',
                                prediction=f"Prediction: {predicted_label} (Confidence: {confidence:.2f}%)",
                                probabilities=prob_display)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('prediction.html', error_message=f"An error occurred during prediction: {str(e)}")

    # For GET request, show the form
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)