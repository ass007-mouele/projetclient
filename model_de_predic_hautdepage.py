##6.Importation du jeu de données final et split des données
# on change le format des données de la variable "date"
df_pool['Date'] = pd.to_datetime(df_pool['Date'])

# on split enfin les données en 3 jeux distincts : un pour l'entraînement des modèles, l'autre pour le test, le dernier pour valider les données prédites avec des valeurs réelles
training = df_pool.iloc[:565]
testing = df_pool.iloc[-88:-28]
validation = df_pool.iloc[-28:-14]



#7.Modèle de prédiction RNN LSTM
#Température : entraînement du modèle
# avant d'entraîner notre modèle, on sélectionne les données que l'on souhaite prédire
training_set = training["Temperature_de_l_eau"].values.reshape(-1, 1)

# puis on effectue une mise à l'échelle sur de ces données
# pour cela, on importe le module suivant et on crée ensuite une variable qui va nous servir à instancier notre outil de mise à l'échelle
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# on fait fitter notre outil sur les données d'entraînement, puis on transforme les deux jeux de données
training_set_scaled = scaler.fit_transform(training_set)

# on crée 2 listes vides : une pour les jours qui nous serviront de points de départ (input), l'autre pour les jours que devra prédire le modèle (output)
x_train = []
y_train = []

# on crée 2 variables : le modèle se basera sur 60 observations (soit 30 jours) pour chercher à prédire à chaque fois 14 observations (soit 7 jours)
n_past = 60
n_future = 14

# on crée enfin une boucle pour ajouter les fameux jours à l'une ou l'autre des listes créées plus tôt
for number in range(0, len(training_set_scaled) - n_past - n_future + 1):
  x_train.append(training_set_scaled[number : number + n_past, 0])     
  y_train.append(training_set_scaled[number + n_past : number + n_past + n_future, 0])

# on change le format des deux listes pour qu'elles soient comprises par le modèle
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# on importe les modules qui nous serviront à mettre en place et utiliser notre réseau de neurones LSTM
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# on commence par créer une variable qui va nous servir à créer les couches de notre réseau de neurones
regressor = Sequential()

# on ajoute une couche LSTM
regressor.add(LSTM(units=8, input_shape=(x_train.shape[1], 1)))

# on ajoute une couche dense, qui va nous ressortir des prédictions à 14 demi-journées (le reste de notre jeu de données)
regressor.add(Dense(units=n_future, activation="linear"))

# on compile toutes nos couches et on définit l'optimizer, l'indicateur que le modèle doit chercher à réduire au cours de son apprentissage et les indicateurs qu'il devra afficher
regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

# on crée 2 variables pour contrôler l'entraînement de notre modèle : une qui va permettre d'arrêter les itérations si le modèle n'apprend plus rien, l'autre qui va sauvegarder le modèle avec la meilleure précision au fur et à mesure des itérations
es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=20)
mc = ModelCheckpoint("lstm_temp.h5", monitor="acc", mode="max", verbose=1, save_best_only=True)

# enfin, on fait fitter notre modèle aux données, en entraînant notre modèle sur 500 itérations, et en découpant les données sur 30 jours (60 demi-journées) à chaque itération
regressor.fit(x_train, y_train, epochs=500, batch_size=60, callbacks=[es, mc])

# pour évaluer le dernier modèle sauvegardé sur les données d'entraînement, on utilise la fonction suivante
acc_train = regressor.evaluate(x_train, y_train, verbose=0)
#print("Accuracy score on training set : {}".format((acc_train[1])))

#Température : chargement du modèle et validation

# pour charger et afficher les éléments clés de notre modèle de prédiction, on importe les modules suivants
from keras.models import load_model

# on charge le modèle préalablement sauvegardé
regressor = load_model("lstm_temp.h5")

# on affiche les éléments clés de notre modèle notre modèle via la fonction suivante
regressor.summary()

# avant de faire nos prédictions notre modèle, on sélectionne les données que l'on souhaite prédire
testing_set = testing["Temperature_de_l_eau"].values.reshape(-1, 1)

# comme pour les données d'entraînement, on met à l'échelle les données de test
testing_set_scaled = scaler.transform(testing_set)

# on transforme également les données de test
testing_set_scaled = np.array(testing_set_scaled)
testing_set_scaled = np.reshape(testing_set_scaled, (testing_set_scaled.shape[1], testing_set_scaled.shape[0], 1))

# on prédit les températures pour les dernières demi-journées
predicted_temperature = regressor.predict(testing_set_scaled)

# on supprime la mise à l'échelle des données prédites
predicted_temperature = scaler.inverse_transform(predicted_temperature)

# on remet les données prédites au bon format et on les affiche dans un dataframe Pandas
predicted_temperature = np.reshape(predicted_temperature, (predicted_temperature.shape[1], predicted_temperature.shape[0]))
df_lstm = pd.DataFrame(predicted_temperature, columns=["predicted_temperature"], index=validation["Date"])

# on ajoute les valeurs réelles au dataframe crée plus tôt pour pouvoir comparer les données
df_lstm["real_temperature"] = validation["Temperature_de_l_eau"].values

# avant de calculer notre score de précision, on sélectionne les données réelles qui doivent être prédites
validation_set = validation["Temperature_de_l_eau"].values.reshape(-1, 1)

# comme pour les données d'entraînement et de test, on met à l'échelle les données de validation
validation_set_scaled = scaler.transform(validation_set)

# on transforme également les données de validation
validation_set_scaled = np.array(validation_set_scaled)
validation_set_scaled = np.reshape(validation_set_scaled, (validation_set_scaled.shape[1], validation_set_scaled.shape[0]))

# pour évaluer le dernier modèle sauvegardé sur les données de validation, on utilise la fonction suivante
acc_test = regressor.evaluate(testing_set_scaled, validation_set_scaled, verbose=0)
#print("Accuracy score on testing set : {}".format((acc_test[1])))

#Température : réglages des hyperparamètres

# on importe les modules qui nous serviront à mettre en place et utiliser notre réseau de neurones LSTM, puis à optimiser les hyperparamètres du modèle
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# on crée une fonction qui va créer un modèle LSTM standard, dans lequel on va faire varier plusieurs paramètres
def create_lstm(units):

# on commence par créer une variable qui va nous servir à créer les couches de notre réseau de neurones
  regressor = Sequential()

# on ajoute une couche LSTM
  regressor.add(LSTM(units=units, input_shape=(x_train.shape[1], 1)))

# on ajoute une couche dense, qui va nous ressortir des prédictions à X demi-journées (le reste de notre jeu de données)
  regressor.add(Dense(units=n_future, activation="linear"))

# on compile toutes nos couches et on définit l'optimizer, l'indicateur que le modèle doit chercher à réduire au cours de son apprentissage et les indicateurs qu'il devra afficher
  regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])
  regressor.fit(x_train, y_train, epochs=100, batch_size=60)
  return regressor

# on liste les paramètres qu'on souhaite tester pour optimiser notre modèle
units = [7, 14, 21, 28]
# activation = ["softmax", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
# optimizer = ["SGD", "RMSprop", "adagrad", "adadelta", "adam", "adamax", "nadam"]
# batch_size = [10, 20, 30, 40, 50, 60]

# on crée ensuite un dictionnaire regroupant tous les paramètres qu'on souhaite tester
param_grid = dict(units=units)

# on crée un modèle LSTM via la fonction suivante, afin que le modèle soit compris par la librairie Scikit-Learn
model = KerasRegressor(build_fn=create_lstm)

# on met en place un GridSearchCV pour tester tous les paramètres listés plus haut 
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)

# on affiche la meilleure combinaison de paramètres ainsi que le score obtenu
#print(grid_result.best_params_)
#print(round(grid_result.best_score_, 2))

#pH : entraînement du modèle

# avant d'entraîner notre modèle, on sélectionne les données que l'on souhaite prédire
training_set = training["pH"].values.reshape(-1, 1)

# puis on effectue une mise à l'échelle sur de ces données
# pour cela, on importe le module suivant et on crée ensuite une variable qui va nous servir à instancier notre outil de mise à l'échelle
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# on fait fitter notre outil sur les données d'entraînement, puis on transforme les deux jeux de données
training_set_scaled = scaler.fit_transform(training_set)

# on crée 2 listes vides : une pour les jours qui nous serviront de points de départ (input), l'autre pour les jours que devra prédire le modèle (output)
x_train = []
y_train = []

# on crée 2 variables : le modèle se basera sur 60 observations (soit 30 jours) pour chercher à prédire à chaque fois 14 observations (soit 7 jours)
n_past = 60
n_future = 14

# on crée enfin une boucle pour ajouter les fameux jours à l'une ou l'autre des listes créées plus tôt
for number in range(0, len(training_set_scaled) - n_past - n_future + 1):
  x_train.append(training_set_scaled[number : number + n_past, 0])     
  y_train.append(training_set_scaled[number + n_past : number + n_past + n_future, 0])

# on change le format des deux listes pour qu'elles soient comprises par le modèle
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# on importe les modules qui nous serviront à mettre en place et utiliser notre réseau de neurones LSTM
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# on commence par créer une variable qui va nous servir à créer les couches de notre réseau de neurones
regressor = Sequential()

# on ajoute une couche LSTM
regressor.add(LSTM(units=8, input_shape=(x_train.shape[1], 1)))

# on ajoute une couche dense, qui va nous ressortir des prédictions à 14 demi-journées (le reste de notre jeu de données)
regressor.add(Dense(units=n_future, activation="linear"))

# on compile toutes nos couches et on définit l'optimizer, l'indicateur que le modèle doit chercher à réduire au cours de son apprentissage et les indicateurs qu'il devra afficher
regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

# on crée 2 variables pour contrôler l'entraînement de notre modèle : une qui va permettre d'arrêter les itérations si le modèle n'apprend plus rien, l'autre qui va sauvegarder le modèle avec la meilleure précision au fur et à mesure des itérations
es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=20)
mc = ModelCheckpoint("lstm_pH.h5", monitor="acc", mode="max", verbose=1, save_best_only=True)

# enfin, on fait fitter notre modèle aux données, en entraînant notre modèle sur 500 itérations, et en découpant les données sur 30 jours (60 demi-journées) à chaque itération
regressor.fit(x_train, y_train, epochs=500, batch_size=60, callbacks=[es, mc])

# pour évaluer le dernier modèle sauvegardé sur les données d'entraînement, on utilise la fonction suivante
acc_train = regressor.evaluate(x_train, y_train, verbose=0)
#print("Accuracy score on training set : {}".format((acc_train[1])))

#pH : chargement du modèle et validation

# pour charger et afficher les éléments clés de notre modèle de prédiction, on importe les modules suivants
from keras.models import load_model

# on charge le modèle préalablement sauvegardé
regressor = load_model("lstm_pH.h5")

# on affiche les éléments clés de notre modèle notre modèle via la fonction suivante
regressor.summary()

# avant de faire nos prédictions notre modèle, on sélectionne les données que l'on souhaite prédire
testing_set = testing["pH"].values.reshape(-1, 1)

# comme pour les données d'entraînement, on met à l'échelle les données de test
testing_set_scaled = scaler.transform(testing_set)

# on transforme également les données de test
testing_set_scaled = np.array(testing_set_scaled)
testing_set_scaled = np.reshape(testing_set_scaled, (testing_set_scaled.shape[1], testing_set_scaled.shape[0], 1))

# on prédit les températures pour les dernières demi-journées
predicted_pH = regressor.predict(testing_set_scaled)

# on supprime la mise à l'échelle des données prédites
predicted_pH = scaler.inverse_transform(predicted_pH)

# on remet les données prédites au bon format et on les affiche dans un dataframe Pandas
predicted_pH = np.reshape(predicted_pH, (predicted_pH.shape[1], predicted_pH.shape[0]))
df_lstm = pd.DataFrame(predicted_pH, columns=["predicted_pH"], index=validation["Date"])

# on ajoute les valeurs réelles au dataframe crée plus tôt pour pouvoir comparer les données
df_lstm["real_pH"] = validation["pH"].values

# avant de calculer notre score de précision, on sélectionne les données réelles qui doivent être prédites
validation_set = validation["pH"].values.reshape(-1, 1)

# comme pour les données d'entraînement et de test, on met à l'échelle les données de validation
validation_set_scaled = scaler.transform(validation_set)

# on transforme également les données de validation
validation_set_scaled = np.array(validation_set_scaled)
validation_set_scaled = np.reshape(validation_set_scaled, (validation_set_scaled.shape[1], validation_set_scaled.shape[0]))

# pour évaluer le dernier modèle sauvegardé sur les données de validation, on utilise la fonction suivante
acc_test = regressor.evaluate(testing_set_scaled, validation_set_scaled, verbose=0)
#print("Accuracy score on testing set : {}".format((acc_test[1])))

#Chlore DPD1 : entraînement du modèle

# avant d'entraîner notre modèle, on sélectionne les données que l'on souhaite prédire
training_set = training["DPD_1"].values.reshape(-1, 1)

# puis on effectue une mise à l'échelle sur de ces données
# pour cela, on importe le module suivant et on crée ensuite une variable qui va nous servir à instancier notre outil de mise à l'échelle
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# on fait fitter notre outil sur les données d'entraînement, puis on transforme les deux jeux de données
training_set_scaled = scaler.fit_transform(training_set)

# on crée 2 listes vides : une pour les jours qui nous serviront de points de départ (input), l'autre pour les jours que devra prédire le modèle (output)
x_train = []
y_train = []

# on crée 2 variables : le modèle se basera sur 60 observations (soit 30 jours) pour chercher à prédire à chaque fois 14 observations (soit 7 jours)
n_past = 60
n_future = 14

# on crée enfin une boucle pour ajouter les fameux jours à l'une ou l'autre des listes créées plus tôt
for number in range(0, len(training_set_scaled) - n_past - n_future + 1):
  x_train.append(training_set_scaled[number : number + n_past, 0])     
  y_train.append(training_set_scaled[number + n_past : number + n_past + n_future, 0])

# on change le format des deux listes pour qu'elles soient comprises par le modèle
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# on importe les modules qui nous serviront à mettre en place et utiliser notre réseau de neurones LSTM
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# on commence par créer une variable qui va nous servir à créer les couches de notre réseau de neurones
regressor = Sequential()

# on ajoute une couche LSTM
regressor.add(LSTM(units=8, input_shape=(x_train.shape[1], 1)))

# on ajoute une couche dense, qui va nous ressortir des prédictions à 14 demi-journées (le reste de notre jeu de données)
regressor.add(Dense(units=n_future, activation="linear"))

# on compile toutes nos couches et on définit l'optimizer, l'indicateur que le modèle doit chercher à réduire au cours de son apprentissage et les indicateurs qu'il devra afficher
regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

# on crée 2 variables pour contrôler l'entraînement de notre modèle : une qui va permettre d'arrêter les itérations si le modèle n'apprend plus rien, l'autre qui va sauvegarder le modèle avec la meilleure précision au fur et à mesure des itérations
es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=20)
mc = ModelCheckpoint("lstm_DPD1.h5", monitor="acc", mode="max", verbose=1, save_best_only=True)

# enfin, on fait fitter notre modèle aux données, en entraînant notre modèle sur 500 itérations, et en découpant les données sur 30 jours (60 demi-journées) à chaque itération
regressor.fit(x_train, y_train, epochs=500, batch_size=60, callbacks=[es, mc])

# pour évaluer le dernier modèle sauvegardé sur les données d'entraînement, on utilise la fonction suivante
acc_train = regressor.evaluate(x_train, y_train, verbose=0)
#print("Accuracy score on training set : {}".format((acc_train[1])))

#Chlore DPD1 : chargement du modèle et validation

# pour charger et afficher les éléments clés de notre modèle de prédiction, on importe les modules suivants
from keras.models import load_model

# on charge le modèle préalablement sauvegardé
regressor = load_model("lstm_DPD1.h5")

# on affiche les éléments clés de notre modèle notre modèle via la fonction suivante
regressor.summary()

# avant de faire nos prédictions notre modèle, on sélectionne les données que l'on souhaite prédire
testing_set = testing["DPD_1"].values.reshape(-1, 1)

# comme pour les données d'entraînement, on met à l'échelle les données de test
testing_set_scaled = scaler.transform(testing_set)

# on transforme également les données de test
testing_set_scaled = np.array(testing_set_scaled)
testing_set_scaled = np.reshape(testing_set_scaled, (testing_set_scaled.shape[1], testing_set_scaled.shape[0], 1))

# on prédit les températures pour les dernières demi-journées
predicted_pH = regressor.predict(testing_set_scaled)

# on supprime la mise à l'échelle des données prédites
predicted_pH = scaler.inverse_transform(predicted_pH)

# on remet les données prédites au bon format et on les affiche dans un dataframe Pandas
predicted_pH = np.reshape(predicted_pH, (predicted_pH.shape[1], predicted_pH.shape[0]))
df_lstm = pd.DataFrame(predicted_pH, columns=["predicted_DPD1"], index=validation["Date"])

# on ajoute les valeurs réelles au dataframe crée plus tôt pour pouvoir comparer les données
df_lstm["real_DPD1"] = validation["DPD_1"].values

# avant de calculer notre score de précision, on sélectionne les données réelles qui doivent être prédites
validation_set = validation["DPD_1"].values.reshape(-1, 1)

# comme pour les données d'entraînement et de test, on met à l'échelle les données de validation
validation_set_scaled = scaler.transform(validation_set)

# on transforme également les données de validation
validation_set_scaled = np.array(validation_set_scaled)
validation_set_scaled = np.reshape(validation_set_scaled, (validation_set_scaled.shape[1], validation_set_scaled.shape[0]))

# pour évaluer le dernier modèle sauvegardé sur les données de validation, on utilise la fonction suivante
acc_test = regressor.evaluate(testing_set_scaled, validation_set_scaled, verbose=0)
#print("Accuracy score on testing set : {}".format((acc_test[1])))

#Chlore DPD3 : entraînement du modèle

# avant d'entraîner notre modèle, on sélectionne les données que l'on souhaite prédire
training_set = training["DPD_3"].values.reshape(-1, 1)

# puis on effectue une mise à l'échelle sur de ces données
# pour cela, on importe le module suivant et on crée ensuite une variable qui va nous servir à instancier notre outil de mise à l'échelle
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# on fait fitter notre outil sur les données d'entraînement, puis on transforme les deux jeux de données
training_set_scaled = scaler.fit_transform(training_set)

# on crée 2 listes vides : une pour les jours qui nous serviront de points de départ (input), l'autre pour les jours que devra prédire le modèle (output)
x_train = []
y_train = []

# on crée 2 variables : le modèle se basera sur 60 observations (soit 30 jours) pour chercher à prédire à chaque fois 14 observations (soit 7 jours)
n_past = 60
n_future = 14

# on crée enfin une boucle pour ajouter les fameux jours à l'une ou l'autre des listes créées plus tôt
for number in range(0, len(training_set_scaled) - n_past - n_future + 1):
  x_train.append(training_set_scaled[number : number + n_past, 0])     
  y_train.append(training_set_scaled[number + n_past : number + n_past + n_future, 0])

# on change le format des deux listes pour qu'elles soient comprises par le modèle
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# on importe les modules qui nous serviront à mettre en place et utiliser notre réseau de neurones LSTM
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# on commence par créer une variable qui va nous servir à créer les couches de notre réseau de neurones
regressor = Sequential()

# on ajoute une couche LSTM
regressor.add(LSTM(units=8, input_shape=(x_train.shape[1], 1)))

# on ajoute une couche dense, qui va nous ressortir des prédictions à 14 demi-journées (le reste de notre jeu de données)
regressor.add(Dense(units=n_future, activation="linear"))

# on compile toutes nos couches et on définit l'optimizer, l'indicateur que le modèle doit chercher à réduire au cours de son apprentissage et les indicateurs qu'il devra afficher
regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

# on crée 2 variables pour contrôler l'entraînement de notre modèle : une qui va permettre d'arrêter les itérations si le modèle n'apprend plus rien, l'autre qui va sauvegarder le modèle avec la meilleure précision au fur et à mesure des itérations
es = EarlyStopping(monitor="loss", mode="min", verbose=1, patience=20)
mc = ModelCheckpoint("lstm_DPD3.h5", monitor="acc", mode="max", verbose=1, save_best_only=True)

# enfin, on fait fitter notre modèle aux données, en entraînant notre modèle sur 500 itérations, et en découpant les données sur 30 jours (60 demi-journées) à chaque itération
regressor.fit(x_train, y_train, epochs=500, batch_size=60, callbacks=[es, mc])

# pour évaluer le dernier modèle sauvegardé sur les données d'entraînement, on utilise la fonction suivante
acc_train = regressor.evaluate(x_train, y_train, verbose=0)
#print("Accuracy score on training set : {}".format((acc_train[1])))

#Chlore DPD3 : chargement du modèle et validation

# pour charger et afficher les éléments clés de notre modèle de prédiction, on importe les modules suivants
from keras.models import load_model

# on charge le modèle préalablement sauvegardé
regressor = load_model("lstm_DPD3.h5")

# on affiche les éléments clés de notre modèle notre modèle via la fonction suivante
regressor.summary()

# avant de faire nos prédictions notre modèle, on sélectionne les données que l'on souhaite prédire
testing_set = testing["DPD_3"].values.reshape(-1, 1)

# comme pour les données d'entraînement, on met à l'échelle les données de test
testing_set_scaled = scaler.transform(testing_set)

# on transforme également les données de test
testing_set_scaled = np.array(testing_set_scaled)
testing_set_scaled = np.reshape(testing_set_scaled, (testing_set_scaled.shape[1], testing_set_scaled.shape[0], 1))

# on prédit les températures pour les dernières demi-journées
predicted_pH = regressor.predict(testing_set_scaled)

# on supprime la mise à l'échelle des données prédites
predicted_pH = scaler.inverse_transform(predicted_pH)

# on remet les données prédites au bon format et on les affiche dans un dataframe Pandas
predicted_pH = np.reshape(predicted_pH, (predicted_pH.shape[1], predicted_pH.shape[0]))
df_lstm = pd.DataFrame(predicted_pH, columns=["predicted_DPD3"], index=validation["Date"])

# on ajoute les valeurs réelles au dataframe crée plus tôt pour pouvoir comparer les données
df_lstm["real_DPD3"] = validation["DPD_3"].values

# avant de calculer notre score de précision, on sélectionne les données réelles qui doivent être prédites
validation_set = validation["DPD_3"].values.reshape(-1, 1)

# comme pour les données d'entraînement et de test, on met à l'échelle les données de validation
validation_set_scaled = scaler.transform(validation_set)

# on transforme également les données de validation
validation_set_scaled = np.array(validation_set_scaled)
validation_set_scaled = np.reshape(validation_set_scaled, (validation_set_scaled.shape[1], validation_set_scaled.shape[0]))

# pour évaluer le dernier modèle sauvegardé sur les données de validation, on utilise la fonction suivante
acc_test = regressor.evaluate(testing_set_scaled, validation_set_scaled, verbose=0)
#print("Accuracy score on testing set : {}".format((acc_test[1])))
##################################################################################################################################################################
