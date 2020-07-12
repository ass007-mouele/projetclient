from flask import Flask ,Blueprint, render_template ,request,flash,redirect
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

#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DataBase.sqlite3'
#app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
#db = SQLAlchemy(app)
#app.config['SECRET_KEY']=os.environ.get('SECRET_KEY')
app.secret_key = 'many random bytes'

if os.environ.get('ENV')=='production':
   app.config['DEBUG'] = False
   app.config['SQLALCHEMY_DATABASE_URI']= os.environ.get('DATABASE_URL')

else:
   app.config['DEBUG'] = True
   app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DataBase.sqlite3'
   app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	#app.config['SECRET_KEY']=os.environ.get('SECRET_KEY')
	
db = SQLAlchemy(app)


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
		flash("Les mésures ont été enregistrées!!!!", 'success')
		db.session.close()
		return render_template("pages/admessures.html")	
	#return render_template("pages/addmesures.html")

@app.route('/donnees')
def donnees():
	posts=Post.query.all()
	return render_template("pages/donnees.html", Posts=Post.query.order_by(Post.id.asc()).all())

@app.route('/supprimer', methods = ['GET', 'POST'])
def supprimer():
	if request.method == 'GET':
		x = Post.query.all()
		q=Post.query.count()
		if q>0:
		   db.session.delete(x[-1])
		   db.session.commit()
		   return render_template("pages/donnees.html", Posts=Post.query.order_by(Post.id.asc()).all())
		else:
		   return render_template("pages/donnees.html", Posts=Post.query.order_by(Post.id.asc()).all())  


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
   #app.run(debug=True, port=3000)
   app.run(port=3000)

