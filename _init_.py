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

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DataBase.sqlite3'
#app.config['DATABASE_URL'] = 'postgres://zcepddlyyfzmud:f727e3fe19092e4cfe714369694ce462c0eb655c229bb9007d6ce37ea54cee46@ec2-46-137-84-173.eu-west-1.compute.amazonaws.com:5432/d3gmstuaidusif'
#SECRET_KEY='ffgggfgfgggfgfggfgf'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL')
SECRET_KEY=os.environ.get('SECRET_KEY')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False



class Post(db.Model):
	__tablename__='Posts'
	id = db.Column(db.Integer, primary_key=True)
	Date = db.Column(db.String(80),nullable=False)
	Heure = db.Column(db.String(80),nullable=False)
	Bassin = db.Column(db.String(80),nullable=False)
	Transparence = db.Column(db.String(80),nullable=False)
	Temperature_de_l_eau = db.Column(db.String(80),nullable=False)
	pH = db.Column(db.String(80),nullable=False)
	DPD_1 = db.Column(db.String(80),nullable=False)
	DPD_3 = db.Column(db.String(80),nullable=False)
	combine = db.Column(db.String(80),nullable=False)
	libre_actif = db.Column(db.String(80),nullable=False)
	compteur = db.Column(db.Integer)
	
	def __init__(self,Date,Heure,Bassin,Transparence,Temperature_de_l_eau,pH,DPD_1,DPD_3,combine,libre_actif,compteur):
		self.Date = Date
		self.Heure = Heure
		self.Bassin = Bassin
		self.Transparence = Transparence
		self.Temperature_de_l_eau = Temperature_de_l_eau
		self.pH= pH
		self.DPD_1 = DPD_1
		self.DPD_3 = DPD_3
		self.combine = combine
		self.libre_actif = libre_actif
		self.compteur = compteur
	def __repr__(self,id):
 		#return '<Post"{}{}{}{}{}{}{}{}{}{}{}">'.format(self.Date,self.Heure,self.Bassin,self.Transparence,self.Temperature_de_l_eau,self.pH,self.DPD_1,self.DPD_3,self.combine,self.libre_actif,self.compteur)
 		return '<Post"{}">'.format(self.id)


@app.route('/')
def home():
	return render_template ('pages/home.html')


@app.route('/addmesures')
def mesures():
	return render_template ('pages/addmesures.html')


@app.route('/addmesures',methods = ['POST','GET'])
def addmesures(): 
	if request.method == 'POST':
	   Date = request.form.get('Date')
	   Heure = request.form.get('Heure')
	   Bassin = request.form.get('Bassin')
	   Transparence = request.form.get('Transparence')
	   Temperature_de_l_eau = request.form.get('Temperature_de_l_eau')
	   pH = request.form.get('pH')
	   DPD_1 = request.form.get('DPD_1')
	   DPD_3 = request.form.get('DPD_3')
	   combine = request.form.get('combine')
	   libre_actif = request.form.get('libre_actif')
	   compteur = request.form.get('compteur')
	   p=Post(Date=Date,Heure=Heure,Bassin=Bassin,Transparence=Transparence,Temperature_de_l_eau=Temperature_de_l_eau,pH=pH,DPD_1=DPD_1,DPD_3=DPD_3,combine=combine,libre_actif=libre_actif,compteur=compteur)
	   db.session.add(p)
	   db.session.commit()
	   flash("Mesures was successfully created")
	   db.session.close()
	   return render_template("pages/addmesures.html")	

@app.route('/donnees')
def donnees():
	posts=Post.query.all()
	return render_template("pages/donnees.html", Posts=Post.query.order_by(Post.id.desc()).all())


@app.route('/predict',methods = ['POST', 'GET'])
def prediction():
	con = sqlite3.connect("base_h2eau.db")
	df = pd.read_sql_query("SELECT * FROM H2eau", con)
	X = df[['DPD_1','DPD_3']] 
	y = df['Combine']
	modelLR = LinearRegression().fit(X, y)
	my_prediction=modelLR.predict(X[-4:-1])
	return render_template("pages/predict.html", prediction = str(my_prediction))



if __name__=='__main__':
   db.create_all()
   app.run(debug=True, port=3000)

