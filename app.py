import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    df=pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    df.info()
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    
    # Loading model to compare the results
    model = pickle.load(open('model.pkl','rb'))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8, inputQuery9, 
             inputQuery10]]
    new_df = pd.DataFrame(data, columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level',
                                           'bmi','gender','ever_married','work_type','Residence_type','smoking_status'])
    
    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:,1]
    print(probability)
    if single==1:
        output = "The patient has stroke "
        output1 = "Confidence: {}".format(probability*100)
    else:
        output = "The patient has not stroke"
        output1 = ""

    return render_template('home.html', output1=output, output2=output1, query1 = request.form['query1'], 
                           query2=request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = 
                           request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'],query8 = 
                           request.form['query8'],query9 = request.form['query9'],query10 = request.form['query10'])


if __name__ == "__main__":
    app.run(debug=True)