from flask import Flask , render_template , request
from flask_sqlalchemy import SQLAlchemy
import sqlite3
#import pyodbc
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from flask import url_for
from sqlalchemy import create_engine,MetaData,Table,select
import os

app=Flask(__name__, static_url_path='/static')

app.config['DATABASE_URL'] = 'sqlite:///DataBase.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Post(db.Model):
	__tablename__='Posts'
	id = db.Column(db.Integer, primary_key=True)
	Date = db.Column(db.String(80))
	Heure = db.Column(db.Time())
	Bassin = db.Column(db.String(80))
	Transparence = db.Column(db.String(80))
	Temperature_de_l_eau = db.Column(db.Float())
	pH = db.Column(db.Integer())
	DPD_1 = db.Column(db.Float())
	DPD_3 = db.Column(db.Float())
	combine = db.Column(db.Float())
	libre_actif = db.Column(db.Float())
	compteur = db.Column(db.Integer())
	def __repr__(self):
 		return '<Post "{}">'.format(self.Date)


@app.route('/')
def home():
	#con = sqlite3.connect("base_h2eau.db")
	#dframe = pd.read_sql_query("SELECT * FROM H2eau", con)
	#graph=px.bar(dframe, x=dframe["Date"][-30:], y=dframe["Combine"][-30:], title='Evolution du chlore combiné sur les 30 jours derniers')
	#graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template ('pages/home.html')


@app.route('/addmesures')
def mesures():
	return render_template ('pages/addmesures.html')


@app.route('/addrec',methods = ['POST','GET'])
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
			p=Post(Date, Heure,Bassin,Transparence,Temperature_de_l_eau,pH,DPD_1,DPD_3,combine,libre_actif,compteur)
			db.session.add(p)
			db.session.commit()
		except:
			db.session.rollback()
			msg = "erreur insertion "
		finally:
			db.session.close()
			#msg = [Date,Heure,Bassin,Transparence,Temperature_de_l_eau,pH,DPD_1,DPD_3,combine,libre_actif,compteur]
			return render_template("pages/resultat.html")#, msg = msg)
			
		


@app.route('/donnees')
def donnees():
	Posts=Post.query.all()
	return render_template("pages/donnees.html",Post = Posts)


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












