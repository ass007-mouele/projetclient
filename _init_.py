from flask import Flask , render_template , request
import sqlite3
#import pyodbc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from flask import url_for

app=Flask(__name__, static_url_path='/static')

@app.route('/')
def home():
	#con = sqlite3.connect("base_h2eau.db")
	#dframe = pd.read_sql_query("SELECT * FROM H2eau", con)
	#graph=px.bar(dframe, x=dframe["Date"][-30:], y=dframe["Combine"][-30:], title='Evolution du chlore combiné sur les 30 jours derniers')
	#graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
	#return render_template ('pages/home.html' , plot= graph)
	return render_template ('pages/home.html')


@app.route('/addmesures')
def mesures():
	return render_template ('pages/addmesures.html')


@app.route('/addrec',methods = ['POST', 'GET'])
def addrec(): 
	if request.method == 'POST':
		try:
			Date = request.form['Date']
			Heure = request.form['Heure']
			Bassin = request.form['Bassin']
			Transparence = request.form['Transparence']
			Temperature_de_l_eau = request.form['Temperature_de_l_eau']
			pH = request.form['pH']
			DPD_1 = request.form['DPD_1']
			DPD_3 = request.form['DPD_3']
			combine = request.form['combine']
			libre_actif = request.form['libre_actif']
			compteur = request.form['compteur']
			with sqlite3.connect("base_h2eau.db") as con:
				cur = con.cursor()
				cur.execute("INSERT INTO H2eau (Date,Heure,Bassin,Transparence,Temperature_de_l_eau,pH,DPD_1,DPD_3,combine,libre_actif,compteur) VALUES (?,?,?,?,?,?,?,?,?,?,?)",(Date,Heure,Bassin,Transparence,Temperature_de_l_eau,pH,DPD_1,DPD_3,combine,libre_actif,compteur))
				con.commit()
				msg = "Enregistrement reussi"
				cursor.close()
		except:
			con.rollback()
			msg = "insertion "#"erreur insertion "
			print("Failed to insert record into Laptop table {}".format(error))
		finally:
			return render_template("pages/resultat.html", msg = msg)
		con.close()
#@app.route('/donnees')
#def donnees():
	#return render_template ('pages/donnees.html')


@app.route('/donnees')
def donnees():
	con = sqlite3.connect("base_h2eau.db")
	con.row_factory = sqlite3.Row
	cur = con.cursor()
	cur.execute("select * from H2eau")
	rows = cur.fetchall();
	return render_template("pages/donnees.html",rows = rows)

@app.route('/predict',methods = ['POST', 'GET'])
def prediction():
	con = sqlite3.connect("base_h2eau.db")
	df = pd.read_sql_query("SELECT * FROM H2eau", con)
	X = df[['DPD_1','DPD_3']] 
	y = df['Combine']
	modelLR = LinearRegression().fit(X, y)
	#if request.methods=='POST':
		#comment=request.form['comment']
		#data=[comment]
	my_prediction=modelLR.predict(X[-4:-1])
	return render_template("pages/predict.html", prediction = str(my_prediction))
	#print("coefficient :",modelLR.coef_)
	#print("interception :", modelLR.intercept_)
	#print("les valeurs de prediction sont:\n",modelLR.predict(X[-4:-1]))


if __name__=='__main__':
	app.run(debug=True,port=3000)




