from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


preprocessor = pickle.load(open(os.path.join('model','preprocessor.pkl'),'rb'))
model = pickle.load(open(os.path.join('model','model.pkl'),'rb'))
application = Flask(__name__)
app = application


@app.route('/', methods=['GET', 'POST'])
def home():
    price = None
    if request.method == 'POST':
        features = {
                'area':float(request.form['area']),
                'bathrooms':int(request.form['bathroom']),
                'bedrooms':int(request.form['bedroom']),
                'stories':int(request.form['stories']),
                'mainroad':request.form['road'],
                'guestroom':request.form['guest'],
                'basement':request.form['basement'],
                'hotwaterheating':request.form['hotwaterheating'],
                'airconditioning':request.form['airconditioning'],
                'parking':int(request.form['parking']),
                'prefarea':request.form['prefarea'],
                'furnishingstatus':request.form['furnishingstatus']
        }
        features_df = pd.DataFrame([features])
        final_features = preprocessor.transform(features_df)

        # Predict price
        price = model.predict(final_features)[0]
        print(type(price))
        return render_template('predict.html',price=f"{price:.3f}")
        

        
    return render_template('home.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Railway provides a PORT env variable
    print("Files in model directory:", os.listdir('model'))
    app.run(host="0.0.0.0", port=port, debug=True)
