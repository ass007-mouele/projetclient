from app import Post,db
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import keras




df_pool = pd.read_sql(sql = db.session.query(Post)\
                         .with_entities(Post.Date,
                                        Post.Heure,
                                        Post.Bassin,Post.Transparence,Post.Temperature_de_l_eau,Post.pH,Post.DPD_1,Post.DPD_3,Post.combine,Post.libre_actif).statement, 
                 con = db.session.bind)

#######################################################################################

df_pool['Date'] = pd.to_datetime(df_pool['Date'])


##################################################################################################################################################################

#8.Prédictions et calcul du Chlore combiné et libre actif
#Création du dataframe et calcul automatique des dates

# on crée un dataframe pour stocker nos prédictions
df_lstm = pd.DataFrame(0, columns=["Date", "Temperature_de_l_eau", "pH", "DPD_1", "DPD_3", "Combine", "Libre_Actif"], index=np.arange(0, 14))

# on importe le module suivant pour manipuler des dates
import datetime

# on crée une fonction qui va remplir les dates de prédictions en fonction de la denrière date du jeu de données que nous avons en notre possession
def find_date(last_date, new_date):

# on crée une boucle pour générer automatiquement une date en fonction de l'entrée précédente
  if new_date.day != last_date.day:
    predicted_date = new_date + datetime.timedelta(hours=6)
  else:
    predicted_date = new_date - datetime.timedelta(hours=6)
  return predicted_date

# on crée une variable retournant la dernière date du jeu de données en notre possession
last_date = df_pool["Date"].iloc[-1]
last_date = pd.to_datetime(last_date, dayfirst=True)

# on crée une boucle pour générer les dates de prédictions et pour les ajouter à notre dataframe
for number in range(14):
  new_date = last_date + datetime.timedelta(hours=12)
  predicted_date = find_date(last_date, new_date)
  df_lstm.loc[number, "Date"] = predicted_date
  last_date = predicted_date

# on affiche les dates comme étant les index du dataframe
df_lstm.set_index(keys="Date", drop=True, inplace=True)

# on crée un autre dataframe (une copie du dataframe plus haut) pour stocker les probabilités de nos prédictions
df_proba = df_lstm.copy()


# pour charger et afficher les éléments clés de nos modèles de prédiction, on importe les modules suivants
from keras.models import load_model

# on charge les modèles préalablement sauvegardés
regressor_temp = load_model("lstm_temp.h5")
regressor_pH = load_model("lstm_pH.h5")
regressor_DPD1 = load_model("lstm_DPD1.h5")
regressor_DPD3 = load_model("lstm_DPD3.h5")


# avant de faire nos prédictions, on sélectionne les données que l'on souhaite prédire (les 60 dernières observations de chacune des 4 variables indépendantes)
predicting = df_pool.iloc[-60:][["Temperature_de_l_eau", "pH", "DPD_1", "DPD_3"]]

# puis on effectue une mise à l'échelle sur de ces données
# pour cela, on importe le module suivant et on crée ensuite une variable qui va nous servir à instancier notre outil de mise à l'échelle
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# on crée une boucle pour prédire les 14 prochaines valeurs de chacune des variables
for column in predicting:
  predict_set = predicting[column].values.reshape(-1, 1)

# comme pour les données d'entraînement et de test, on met à l'échelle les données de prédiction
  predict_set_scaled = scaler.fit_transform(predict_set)

# on transforme également les données de prédiction
  predict_set_scaled = np.array(predict_set_scaled)
  predict_set_scaled = np.reshape(predict_set_scaled, (predict_set_scaled.shape[1], predict_set_scaled.shape[0], 1))

# on crée une boucle, qui en fonction des données d'entrée, va utiliser l'un des modèles chargés précédemment et prédire des valeurs pour les dernières 14 demi-journées
  if column == "Temperature_de_l_eau":
    predicted_values = regressor_temp.predict(predict_set_scaled)
    proba_prediction = regressor_temp.predict_proba(predict_set_scaled)
  if column == "pH":
    predicted_values = regressor_pH.predict(predict_set_scaled)
    proba_prediction = regressor_pH.predict_proba(predict_set_scaled)
  if column == "DPD_1":
    predicted_values = regressor_DPD1.predict(predict_set_scaled)
    proba_prediction = regressor_DPD1.predict_proba(predict_set_scaled)
  if column == "DPD_3":
    predicted_values = regressor_DPD3.predict(predict_set_scaled)
    proba_prediction = regressor_DPD3.predict_proba(predict_set_scaled)

# on supprime la mise à l'échelle des données prédites
  predicted_values = scaler.inverse_transform(predicted_values)

# on remet les données prédites au bon format
  predicted_values = np.reshape(predicted_values, (predicted_values.shape[1], predicted_values.shape[0]))
  proba_prediction = np.reshape(proba_prediction, (proba_prediction.shape[1], proba_prediction.shape[0]))
  
# on ajoute les valeurs prédites au dataframe crée plus tôt
  df_lstm[column] = predicted_values
  df_proba[column] = proba_prediction

  #Calcul des valeurs predictes de Chlore combiné et libre actif (et probabilités)

  #Rappel des fonctions de calculs du Chlore combiné et libre actif ci-dessous

# Pour le chlore combiné, utilisez la fonction suivante
# Pour le chlore combiné, utilisez la fonction suivante
def chlorecomb_finder(DPD_1, DPD_3):
  result = DPD_3 - DPD_1
  return result

# pour le chlore libre_actif, utilisez la fonction suivante
def libreactif_finder(temperature, pH, DPD_1):

# les coefficients et ordonnée à l'origine de l'équation générée plus haut   
    a = -0.0140047619
    b = 0.00011047619
    c = 7.72062392857

# la formule de l'équation de la constante d'acidité
    pKa = (a * temperature) + (b * (temperature**2)) + c

# le coefficient lié à la température et au pH
    coeff = 100 / (1 + 10**(pH - pKa))

# le calcul du chlore libre actif
    libre_actif = DPD_1 * coeff / 100
    return libre_actif

# on utilise la fonction chlorecomb_finder utilisée pour le calcul automatique du Chlore combiné
df_lstm["Combine"] = chlorecomb_finder(df_lstm["DPD_1"], df_lstm["DPD_3"])

# on utilise la fonction libreactif_finder utilisée pour le calcul automatique du Chlore libre actif
df_lstm["Libre_Actif"] = libreactif_finder(df_lstm["Temperature_de_l_eau"], df_lstm["pH"], df_lstm["DPD_1"])

# on calcule également les probabilités des valeurs prédites pour le chlore combiné... 
df_proba["Combine"] = (df_proba["DPD_1"] + df_proba["DPD_3"]) / 2

# ...et pour le chlore libre actif
df_proba["Libre_Actif"] = (df_proba["Temperature_de_l_eau"] + df_proba["pH"] + df_proba["DPD_1"]) / 3

# on multiplie le tout par 100 pour avoir des pourcentages
df_proba = df_proba * 100
