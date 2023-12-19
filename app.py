from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Initialize the threshold value (can be changed easily as needed)
THRESHOLD_STRENGTH = 31

# Function to load data
# Function to load data
def load_data(file_path):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        return pd.DataFrame(columns=['Date', 'AverageStrength', 'Days'])
    else:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        return df

# Function to add new data
def add_new_data(df, week, average_strength):
    start_date = df['Date'].min() if not df.empty else datetime.now()
    new_date = start_date + timedelta(weeks=week)
    new_data = pd.DataFrame([{'Date': new_date, 'AverageStrength': average_strength}])
    df = pd.concat([df, new_data], ignore_index=True)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    return df

# Function to predict and evaluate
def predict_and_evaluate(df, weeks_ahead):
    if df.empty:
        return 0, False

    model = LinearRegression()
    model.fit(df[['Days']], df['AverageStrength'])

    # Predicting for future weeks
    future_days = df['Days'].max() + np.array([7 * i for i in range(1, weeks_ahead + 1)])
    future_predictions = model.predict(future_days.reshape(-1, 1))

    # Determine the safety status
    is_safe = all(p >= THRESHOLD_STRENGTH for p in future_predictions)
    return future_predictions[0], is_safe

@app.route('/', methods=['GET', 'POST'])
def index():
    message = ""
    latest_strength = 0
    if request.method == 'POST':
        df = load_data('data.csv')
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filepath = os.path.join('uploads', file.filename)
                file.save(filepath)
                df = load_data(filepath)

        week = int(request.form['week'])
        average_strength = float(request.form['average_strength'])
        df = add_new_data(df, week, average_strength)
        df.to_csv('data.csv', index=False)

        weeks_ahead = int(request.form['weeks_ahead'])
        latest_strength, is_safe = predict_and_evaluate(df, weeks_ahead)
        safety_status = "safe" if is_safe else "NOT safe"
        message = f"The predicted compressive strength is {latest_strength:.2f} N/mm². "
        message += f"The building is predicted to be {safety_status} for the next {weeks_ahead} weeks."
    
    return render_template('index.html', message=message, latest_strength=latest_strength)

if __name__ == '__main__':
    app.run(debug=True)
