from flask import Flask, render_template ,request
import pandas as pd
import pickle
import logging
logging.basicConfig(level=logging.INFO)


model = pickle.load(open("LinearRrgressionModel.pkl", 'rb'))


app = Flask(__name__)
car = pd.read_csv("cleaned_car.csv")


print(car.columns)


@app.route('/')
def home():
    companies = sorted(car['company'].unique())  # Extract unique companies
    car_models = sorted(car['name'].unique())   # Extract unique car models
    years = sorted(car['year'].unique(), reverse=True)  # Extract and sort years
    fuel_types = car['fuel_type'].unique()      # Extract unique fuel types
    return render_template('index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict',methods=['POST'])
def predict():
    try:
     company= request.form.get('company')
     car_model= request.form.get('car_model')
     year= int(request.form.get('year'))
     fuel_type= request.form.get('fuel_type')
     kms_driven= int(request.form.get('kilo_driven'))
     print(company)
     prediction = model.predict(pd.DataFrame([[car_model, company,year,kms_driven,fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))

     print(prediction)
     return str(prediction[0])  # Returning the predicted value as a string
    except Exception as e:
        return f"Error: (e)",400


if __name__ == "__main__":
    app.run(debug=True)
