from flask import Flask ,Blueprint, render_template ,request,flash,redirect,url_for
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
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import plotly
import plotly.graph_objs as go
import json

# on importe les modules suivants pour manipuler le jeu de données, le visualiser et créer nos modèles de classification
import plotly.express as px


#app=Flask(__name__)
#db = SQLAlchemy(app)
#app.config['SQLALCHEMY_DATABASE_URI']= os.environ.get('DATABASE_URL')
#db.create_all()
#db.init_app(app)


app = Flask(__name__)
app.config.from_object(Config)
app.secret_key =os.environ.get('SECRET_KEY')
db = SQLAlchemy(app)





...
# Database initialization
#if os.environ.get('DATABASE_URL') is None:
 #   basedir = os.path.abspath(os.path.dirname(__file__))
  #  SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
##else:
#SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']
#db.init_app(app)



#if os.environ.get('ENV')=='production':
   #app.config['DEBUG'] = False
   #app.config['SQLALCHEMY_DATABASE_URI']= os.environ.get('DATABASE_URL')
   #app.config['SECRET_KEY']=os.environ.get('SECRET_KEY')	
   #app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
   	
   	

#else:
   #app.config['DEBUG'] = True
   #app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///DataBase.sqlite3'
   #DATABASE_URL = 'sqlite:///DataBase.sqlite3'	
   #SECRET_KEY="h2eauassistance"
   #app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	



class Post(db.Model):
	__tablename__='indic'
	id = db.Column(db.Integer, primary_key=True)
	Date = db.Column(db.String(80),nullable=False)
	Heure = db.Column(db.String(80),nullable=False)
	Frequentation=db.Column(db.Integer)
	Bassin = db.Column(db.String(80),nullable=False)
	Transparence = db.Column(db.String(80),nullable=False)
	Temperature_de_l_eau = db.Column(db.Float,nullable=False)
	pH = db.Column(db.Float,nullable=False)
	DPD_1 = db.Column(db.Float,nullable=False)
	DPD_3 = db.Column(db.Float,nullable=False)
	combine = db.Column(db.String(80),nullable=False)
	libre_actif = db.Column(db.String(80),nullable=False)
	compteur = db.Column(db.Integer)
	
	def __init__(self,Date,Heure,Frequentation,Bassin,Transparence,Temperature_de_l_eau,pH,DPD_1,DPD_3,combine,libre_actif,compteur):
		self.Date = Date
		self.Heure = Heure
		self.Frequentation = Frequentation
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


	
#db.create_all()

migrate = Migrate(app, db)
db.init_app(app)

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
		Frequentation=request.form.get('Frequentation')
		Bassin = request.form.get('Bassin')
		Transparence = request.form.get('Transparence')
		Temperature_de_l_eau = request.form.get('Temperature_de_l_eau')
		pH = request.form.get('pH')
		DPD_1 = request.form.get('DPD_1')
		DPD_3 = request.form.get('DPD_3')
		combine = request.form.get('combine')
		libre_actif = request.form.get('libre_actif')
		compteur = request.form.get('compteur')
		p=Post(Date=Date,Heure=Heure,Frequentation=Frequentation,Bassin=Bassin,Transparence=Transparence,Temperature_de_l_eau=Temperature_de_l_eau,pH=pH,DPD_1=DPD_1,DPD_3=DPD_3,combine=combine,libre_actif=libre_actif,compteur=compteur)
		db.session.add(p)
		db.session.commit()
		#db.session.close()
		flash("Les mésures ont été enregistrées!!!!" , "success")
	return render_template("pages/addmesures.html")
	#db.session.close()
		#except:
			#flash("Mésures non enregistrées!!!", 'error')
			#db.session.rollback()
			#return render_template("pages/addmesures.html")



@app.route('/donnees')
def donnees():
	posts=Post.query.all()
	return render_template("pages/donnees.html", indic=Post.query.order_by(Post.id.asc()).all())

@app.route('/supprimer', methods = ['GET', 'POST'])
def supprimer():
	if request.method == 'GET':
		x = Post.query.all()
		q=Post.query.count()
		if q>0:
		   db.session.delete(x[-1])
		   db.session.commit()
		   return render_template("pages/donnees.html", indic=Post.query.order_by(Post.id.asc()).all())
		else:
		   return render_template("pages/donnees.html", indic=Post.query.order_by(Post.id.asc()).all())  
	

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
	          flash(rouge, reco)
		  flash(reco_bas)
	        else:
	          flash(orange,reco)
		  flash(reco_bas)
	      
	      if prediction > 1.4:
	        flash("Attention : risque d'irritation de la peau et de formation de sous-produits de chloration (chloramines) !")
	        if prediction > 5:
	          flash(rouge, reco)
	        else:
	          flash(orange, reco)

	 
	df_lstm=df_lstm.reset_index()
	df_lstm=df_lstm.rename(columns = {'Temperature_de_l_eau': 'Temperature de l eau','DPD_1': 'DPD 1', 'DPD_3': 'DPD 3' ,'Combine': 'Chlore Combiné', 'Libre_Actif':'Chlore libre actif'})

	return render_template("pages/maprediction.html",  tables=[df_lstm.to_html(classes='data', header="true")])



@app.route('/dataviz')
def create_plot():
   
    from modelprediction import df_lstm, df_pool,df_proba
# on réinitialise les indexes du jeu de données
    df_lstm.reset_index(inplace=True)

	# on crée une nouvelle colonne pour désigner si les valeurs sont réelles ou prédites...
    df_pool["Catégorie"] = "Données réelles"
    df_lstm["Catégorie"] = "Valeurs prédites"

	# ...puis on concatène les 2 jeux de données
    df_graph = pd.concat([df_pool, df_lstm], ignore_index=True)
    # on affiche un graphique de l'évolution des températures au cours du temps
    fig = px.line(data_frame=df_graph, x=df_graph.index, y="Temperature_de_l_eau", color="Catégorie",color_discrete_sequence=["cornflowerblue", "turquoise"],template="plotly_white").for_each_trace(lambda title : title.update(name=title.name.split("=")[-1]))
	# on affiche le titre et le nom des axes du graphique
	# on supprime les graduations pour que la lecture soit plus agréable
    fig.update_layout(title="Évolution de la température de l'eau au cours de l'année",xaxis_title="Temps",yaxis_title="Temperature de l'eau (en °C)",xaxis_showgrid=False,yaxis_showgrid=False,hovermode="x")

	# on change la position et l'affiche de la légende
    fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.7,xanchor="center",x=0.5))

	# on change les informations qui sont données par les étiquettes
    fig.update_traces(hovertemplate="Température de l'eau : %{y}")

	# on ajoute un slider pour pouvoir sélectionner l'intervalle de données que l'on souhaite
	# on ajoute également des boutons d'options pour sélectionner les données sur un intervalle de temps prédéfini
    fig.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(label="tout sélectionner", step="all"),dict(count=7, label="dernière semaine", step="day", stepmode="backward"),dict(count=14, label="2 dernières semaines", step="day", stepmode="backward"),dict(count=1, label="dernier mois", step="month", stepmode="backward"),dict(count=4, label="4 derniers mois", step="month", stepmode="backward")])))

	# on affiche les constantes qui représenteront les seuils définis par l'ARS
    fig.add_trace(go.Scatter(x=df_graph.index, y=[33]  * len(df_graph.index), name="Non conforme", opacity=0.3, line=(dict(color="orange"))))
    fig.add_trace(go.Scatter(x=df_graph.index, y=[36] * len(df_graph.index), name="Urgence sanitaire", opacity=0.3, line=(dict(color="red"))))

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    ##################

    # on affiche un graphique des probablités de chaque prédiction des températures au cours du temps
    fig1 = px.bar(data_frame=df_proba, x=df_proba.index, y="Temperature_de_l_eau", template="plotly_white")
	# on affiche le titre et le nom des axes du graphique
	# on supprime les graduations pour que la lecture soit plus agréable
    fig1.update_layout(title="Taux de fiabilité des prédictions des températures",xaxis_title="Temps",yaxis_title="Probabilités des prédictions (en %)",xaxis_showgrid=False,yaxis_showgrid=False,hovermode="x")

	# on change la couleur des barres
    fig1.update_traces(marker_color="turquoise",opacity=0.6)

	# on change les informations qui sont données par les étiquettes
    fig1.update_traces(hovertemplate="Probabilité : %{y}%",texttemplate="%{y:.0f}%",textposition="outside")

    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    ##########
    #PH

    # on affiche un graphique de l'évolution du pH au cours du temps
    fig3 = px.line(data_frame=df_graph, x=df_graph.index, y="pH", color="Catégorie",color_discrete_sequence=["cornflowerblue", "turquoise"],template="plotly_white").for_each_trace(lambda title : title.update(name=title.name.split("=")[-1]))

	# on affiche le titre et le nom des axes du graphique
	# on supprime les graduations pour que la lecture soit plus agréable
    fig3.update_layout(title="Évolution du pH au cours de l'année",xaxis_title="Temps",yaxis_title="pH",xaxis_showgrid=False,yaxis_showgrid=False,hovermode="x")

	# on change la position et l'affiche de la légende
    fig3.update_layout(legend=dict(orientation="h",yanchor="bottom",y=-0.7,xanchor="center",x=0.5))

	# on change les informations qui sont données par les étiquettes
    fig3.update_traces(hovertemplate="pH : %{y}")

	# on ajoute un slider pour pouvoir sélectionner l'intervalle de données que l'on souhaite
	# on ajoute également des boutons d'options pour sélectionner les données sur un intervalle de temps prédéfini
    fig3.update_xaxes(rangeslider_visible=True,rangeselector=dict(buttons=list([dict(label="tout sélectionner", step="all"),dict(count=7, label="dernière semaine", step="day", stepmode="backward"),dict(count=14, label="2 dernières semaines", step="day", stepmode="backward"),dict(count=1, label="dernier mois", step="month", stepmode="backward"),dict(count=4, label="4 derniers mois", step="month", stepmode="backward")])))

	# on affiche les constantes qui représenteront les seuils définis par l'ARS
    fig3.add_trace(go.Scatter(x=df_graph.index, y=[6.9] * len(df_graph.index), name="Non conforme (minimum)", opacity=0.3, line=(dict(color="orange"))))
    fig3.add_trace(go.Scatter(x=df_graph.index, y=[7.7] * len(df_graph.index), name="Non conforme (maximum)", opacity=0.3, line=(dict(color="orange"))))

    fig3.add_trace(go.Scatter(x=df_graph.index, y=[5] * len(df_graph.index), name="Urgence sanitaire (minimum)", opacity=0.3, line=(dict(color="red"))))
    fig3.add_trace(go.Scatter(x=df_graph.index, y=[8.5] * len(df_graph.index), name="Urgence sanitaire (maximum)", opacity=0.3, line=(dict(color="red"))))
    graphJSON3 = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)

    ######

    # on affiche un graphique des probablités de chaque prédiction des températures au cours du temps
    fig4 = px.bar(data_frame=df_proba, x=df_proba.index, y="pH", template="plotly_white")

	# on affiche le titre et le nom des axes du graphique
	# on supprime les graduations pour que la lecture soit plus agréable
    fig4.update_layout(title="Taux de fiabilité des prédictions du pH",xaxis_title="Temps",yaxis_title="Probabilités des prédictions (en %)",xaxis_showgrid=False,yaxis_showgrid=False,hovermode="x")

	# on change la couleur des barres
    fig4.update_traces(marker_color="turquoise",opacity=0.6)

	# on change les informations qui sont données par les étiquettes
    fig4.update_traces(hovertemplate="Probabilité : %{y}%",texttemplate="%{y:.0f}%",textposition="outside")

    graphJSON4 = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('pages/dataviz.html', plot=graphJSON, plot1=graphJSON1, plot2=graphJSON3 , plot3=graphJSON4)


@app.before_first_request
def initialize():
    app.logger.info("Creating the tables we need")

