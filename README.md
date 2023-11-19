# Electro-Demand
Kerala has seen a critical turnover in the climatic conditions in the last 4 years. Due to sudden climatic conditions the fluctuating water level in dams are causing to not meet the energy requirements of the state during times of high demand period and we have to buy electricity from other states. To tackle it we paln on creating a fully efficient integrated system that can predict as well as take input the climatic factors affecting like temperature, rainfall, etc and predict the weather condition which in turn will predict the type of energy source to be used for that climatic condition(solar/hydro), the energy demand during high peak demand period and the energy to be produced to meet that demand. 

This web application is a sample of that system that predicts energy-related parameters including weather, energy demand, and power generation based on historical data. It utilizes machine learning models trained on a dataset containing weather, energy production, and demand information. Users can input a date and optionally provide temperature and rainfall values to receive predictions about energy-related factors.

# Features
1. Predicts weather conditions (sunny or rainy).
2. Estimates energy demand based on historical patterns.
3. Suggests the optimal power generation source (solar or hydro) depending on the weather.
4. Calculates the total power needed to meet demand.
5. Determines power generation by source (solar and hydro) for user-specified inputs.

# Usage
1. Usage
2. Access the web application at http://localhost:5000.
3. Enter a date (YYYY-MM) and click "Predict" to obtain energy predictions.
4. Optionally, on the "Predict with Temperature and Rainfall" page, provide temperature (in Celsius) and rainfall (in mm) for more accurate predictions.

# Technologies Used
1. Flask for web application development.
2. Python for data preprocessing and machine learning.
3. Scikit-learn for machine learning models.
4. XGBoost for regression.
5. Random Forest for classification and regression.
6. Weather API to fetch weather condition currently updated

# Demo-Video
https://www.loom.com/share/e59039e0bd254d75939274b5277b1343?sid=8e8acfc3-c531-451b-9354-a32925e5be11
