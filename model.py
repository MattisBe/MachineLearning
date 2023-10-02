# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:53:14 2023

@author: iroxx
"""
#import des packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#import du dataset
data = pd.read_csv(r"C:\Users\iroxx\OneDrive\Documents\Github\machine learning\Iris Flower Dataset\IRIS.csv")
df = data.copy()
df.head(5)

#statistiques descriptives
df.shape #150 lignes et 5 colonnes
df.describe() #On remarque que la longueur des pétales varie beaucoup selon l'espèce tandis que la largeur des sépales ne varie que etrès peu entre les espèces

#visualisations sur les pétales
plt.figure(figsize=(16,7))
plt.plot(df["petal_length"], df["petal_width"],"o")
plt.xlabel("petal_length")
plt.ylabel("petal_width")
plt.show() 
#On peut supposer qu'il y a une corrélation claire entre la longueur des pétales et sa largeur. Plus la pétale et longue, plus sa largeur augmentent.


#visualisations sur les sépales
plt.figure(figsize=(16,7))
plt.plot(df["sepal_length"], df["sepal_width"],"x")
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.show() 
#A l'inverse, on remarque que pour les sépales, les résultats sont très éparses. On ne remarque pas au premier qu'une d'oeil une quelconque corrélation entre la longueur et la largeur. On peut aussi bien voir une sépale très longue et peu large, qu'une sépale très courte et très large.


#visualisations avec seaborn
plt.figure(figsize=(16,7))
sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
plt.show()

plt.figure(figsize=(16,7))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.show()


plt.figure(figsize=(16,7))
sns.pairplot(data=df, hue='species')
plt.show()

#création d'une matrice de corrélations. Des coefficients compris entre -1 et +1 y  sont associés et représentent la niveau de corrélation entre chaque variable.
corr = df.corr()
fig, ax =plt.subplots(figsize=(16, 7))
sns.heatmap(corr, annot=True, cmap = "Blues")


#Encodage des labels catégoriques, ici "species"
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])
df.head()

#On divise le dataset en deux, l'un pour entrainer le modèle, l'autre pour le tester
from sklearn.model_selection import train_test_split
#train - 70% du dataset
#test - 30% du dataset

x = df.drop(columns=["species"])
y = df["species"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

#entrainement du modèle
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

#suivi de la performance du model
accu = print("Précision: ", model.score(x_test, y_test) * 100)

#graphique qui montre l'évolution de la précision du modèle

    









































































