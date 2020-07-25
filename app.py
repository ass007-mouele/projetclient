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
from flask import (
    Blueprint, flash, redirect, render_template, request, url_for
)



app=Flask(__name__)
db = SQLAlchemy(app)


if os.environ.get('ENV')=='production':
   app.config['DEBUG'] = False
   app.config['SQLALCHEMY_DATABASE_URI']= os.environ.get('DATABASE_URL')
   app.config['SECRET_KEY']=os.environ.get('SECRET_KEY')	
   app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
   	
   	

else:
   app.config['DEBUG'] = True
   #app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DataBase.sqlite3'
   DATABASE_URL = 'sqlite:///DataBase.sqlite3'	
   SECRET_KEY="h2eauassistance"
   #app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	



class Post(db.Model):
	__tablename__='Posts'
	id = db.Column(db.Integer, primary_key=True)
	Date = db.Column(db.String(80),nullable=False)
	Heure = db.Column(db.String(80),nullable=False)
	Bassin = db.Column(db.String(80),nullable=False)
	Transparence = db.Column(db.String(80),nullable=False)
	Temperature_de_l_eau = db.Column(db.Float,nullable=False)
	pH = db.Column(db.Float,nullable=False)
	DPD_1 = db.Column(db.Float,nullable=False)
	DPD_3 = db.Column(db.Float,nullable=False)
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
		db.session.close()
		#flash("Les mésures ont été enregistrées!!!!" , "success")
		return render_template("pages/addmesures.html")
	
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
	

@app.route('/maprediction',methods = ['POST', 'GET'])
def html_table():
	from modelprediction import df_lstm

	###### Aletre Temperature
	# on instancie des variables comportant les recommandations de l'ARS
	orange = "\n==> Recommandation(s) de l'ARS : \n- action corrective nécessaire"
	rouge = "\n==> Recommandation(s) de l'ARS : \n- évacuation du bassin \n- vidange partielle ou totale du bassin"

	# on affiche le date et la valeur de chaque prédiction
	for i in range(2):
		flash("====================")
		flash(df_lstm.index[i])
		prediction = df_lstm["Temperature_de_l_eau"][df_lstm.index[i]]

		# on crée une boucle pour vérifier la valeur affichée et on retourne les recommandations en fonction
		if prediction < 33:
			flash("Température conforme au code de la santé publique",'success')
		else:
			flash("Température non conforme au code de la santé publique","error")
			if prediction >= 33 and prediction < 36:
				flash("Attention : risque de prolifération bactérienne dans l'eau !")
				flash(orange)
			else:
				flash("Attention : risque de prolifération bactérienne dans l'eau, risque pour les femmes enceintes !")
				flash(rouge)
	##### Alerte Ph
	#on instancie des variables comportant les recommandations de l'ARS
	orange = "\n==> Recommandation(s) de l'ARS : \n- action corrective nécessaire"
	rouge = "\n==> Recommandation(s) de l'ARS : \n- évacuation du bassin \n- vidange partielle ou totale du bassin"
	reco = "\n- vérification du dispositif de régulation du pH (pompe doseuse, sonde et bac d'acide ou de base)"

	# on affiche le date et la valeur de chaque prédiction
	for i in range(2):
		flash("====================")
		flash(df_lstm.index[i])
		prediction = df_lstm["pH"][df_lstm.index[i]]

		# on crée une boucle pour vérifier la valeur affichée et on retourne les recommandations en fonction
		if prediction >= 6.9 and prediction <= 7.7:
			flash("pH conforme au code de la santé publique",'success')
		else:
			flash("pH non conforme au code de la santé publique",'error')
		if prediction < 6.9:
			flash("Attention : risque d'irritations des muqueuses des baigneurs !")
			if prediction < 5:
				flash(rouge, reco)
			else:
				flash(orange, reco)
		if prediction > 7.7:
			flash("Attention : risque de prolifération bactérienne dans l'eau (désinfectant moins efficace) !")
			if prediction > 8.5:
				flash(rouge, reco)
			else:
				flash(orange, reco)  

    ##### Alerte Chlore combine  	
	orange = "\n==> Recommandation(s) de l'ARS : \n- action corrective nécessaire"
	rouge = "\n==> Recommandation(s) de l'ARS : \n- évacuation du bassin \n- vidange partielle ou totale du bassin"
	reco = "\n- effectuer un apport d'eau neuve \n- faire une vidange (totale pour les bassins de petit volume - partielle pour les autres) \n- améliorer la ventilation des halls des bassins (maintenir la nuit) \n- vérifier le fonctionnement du filtre et la qualité du matériau filtrant \n- vérifier la qualité des produits de nettoyage des surfaces et leurs procédures d'application \n- respecter la fréquentation maximale instantanée (FMI)"

	# on affiche le date et la valeur de chaque prédiction
	for i in range(2):
		flash("====================")
		flash(df_lstm.index[i])
		prediction = df_lstm["Combine"][df_lstm.index[i]]

		# on crée une boucle pour vérifier la valeur affichée et on retourne les recommandations en fonction
		if prediction <= 0.6:
		   flash("Chlore combiné conforme au code de la santé publique", 'success')
		else:
		   flash("Chlore combiné non conforme au code de la santé publique",'error')
		   flash("Attention : risque d'irritations des muqueuses des yeux et des voies respiratoires !")
		      
		   if prediction > 0.8:
		      flash(rouge, reco)
		   else :
		   	  flash(orange, reco)

	##### Chlore libre actif

	orange = "\n==> Recommandation(s) de l'ARS : \n- action corrective nécessaire"
	rouge = "\n==> Recommandation(s) de l'ARS : \n- évacuation du bassin \n- vidange partielle ou totale du bassin"
	reco = "\n- vérifier le dispositif d'injection et de régulation du chlore \n- effectuer un apport d'eau neuve incluant une vidange partielle si nécessaire"
	reco_bas = "\n- vérifier la qualité des produits de désinfection de l'eau \n- nettoyer et désinfecter les filtres \n- nettoyer et désinfecter les systèmes d'évacuation par la surface \n- augmenter et maintenir la chloration au maximum du seuil réglementaire \n- pour les petits volumes : vidanger, nettoyer et désinfecter le fond et les parois du bassin"
	# on affiche le date et la valeur de chaque prédiction
	for i in range(2):
	    flash("====================")
	    flash(df_lstm.index[i])
	    prediction = df_lstm["Libre_Actif"][df_lstm.index[i]]

	# on crée une boucle pour vérifier la valeur affichée et on retourne les recommandations en fonction
	    if prediction >= 0.4 and prediction <= 1.4:
	      flash("Chlore libre actif conforme au code de la santé publique",'success')
	    else:
	      flash("Chlore libre actif non conforme au code de la santé publique",'error')
	      
	      if prediction < 0.4:
	        flash("Attention : risque de prolifération bactérienne dans l'eau !")
	        if prediction < 0.3:
	          flash(rouge, reco, reco_bas)
	        else:
	          flash(orange, reco, reco_bas)
	      
	      if prediction > 1.4:
	        flash("Attention : risque d'irritation de la peau et de formation de sous-produits de chloration (chloramines) !")
	        if prediction > 5:
	          flash(rouge, reco)
	        else:
	          flash(orange, reco)

	 
	df_lstm=df_lstm.reset_index()
	df_lstm=df_lstm.rename(columns = {'Temperature_de_l_eau': 'Temperature de l eau','DPD_1': 'DPD 1', 'DPD_3': 'DPD 3' ,'Combine': 'Chlore Combiné', 'Libre_Actif':'Chlore libre actif'})

	return render_template("pages/maprediction.html",  tables=[df_lstm.to_html(classes='data', header="true")])



@app.route('/predict',methods = ['POST', 'GET'])
def prediction():
	con = sqlite3.connect("base_h2eau.db")
	df = pd.read_sql_query("SELECT * FROM H2eau", con)
	X = df[['DPD_1','DPD_3']] 
	y = df['Combine']
	modelLR = LinearRegression().fit(X, y)
	my_prediction=modelLR.predict(X[-4:-1])
	return render_template("pages/predict.html", prediction = str(my_prediction))

#db.create_all()
if __name__=='__main__':
   db.create_all()
   #app.run(debug=True, port=3000)
   app.run(debug=True)

