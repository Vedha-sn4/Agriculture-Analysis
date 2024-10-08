from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

data = {
    'rainfall': [120, 85, 100, 150, 90, 200],
    'temperature': [25, 20, 22, 18, 24, 26],
    'soil_nutrients': [7, 6, 7.5, 8, 5.5, 9],
    'market_price': [10, 12, 8, 15, 9, 13],
    'crop': ['wheat', 'rice', 'maize', 'tea', 'coffee', 'cotton']
}

df = pd.DataFrame(data)

X = df[['rainfall', 'temperature', 'soil_nutrients', 'market_price']]
y = df['crop']

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        soil_nutrients = float(request.form['soil_nutrients'])
        market_price = float(request.form['market_price'])

        X_test = pd.DataFrame([[rainfall, temperature, soil_nutrients, market_price]],
                              columns=['rainfall', 'temperature', 'soil_nutrients', 'market_price'])

        prediction = model.predict(X_test)[0].upper()  # Convert to uppercase

        explanation = f"Based on the provided data: Rainfall: {rainfall} cm, Temperature: {temperature}°C, Soil Nutrients: {soil_nutrients}, and Market Price: ${market_price} per unit, the recommended crop to grow is {prediction}. This recommendation is based on optimal growing conditions for {prediction}, including suitable rainfall, temperature, and soil nutrient requirements."

        return render_template('index.html', prediction=prediction, explanation=explanation, rainfall=rainfall, temperature=temperature, soil_nutrients=soil_nutrients, market_price=market_price)

    return render_template('index.html', prediction=None, explanation=None)

if __name__ == '__main__':
    app.run(debug=True)
