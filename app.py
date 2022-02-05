import numpy as np
from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)
model = pickle.load(open('linear_model.pickle', 'rb'))


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    print(output)
    return render_template('index.html', prediction='cost should be Rupee {}'. format(output))


if __name__ == "__main__":
    app.run(debug=True)
