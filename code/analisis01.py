import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv("../data/StudentDepression_df.csv")

# Mostrar las primeras filas
print(df.head())


#01 Preparación de datos

#Análisis Gender

df["numGender"] = df["Gender"].map({"Male": 1, "Female": 0})
df = df.drop(columns=["Gender"])


categorias_unicas = df["City"].unique()
print(categorias_unicas)
len(categorias_unicas)

#Análisis Profession
categorias_Profession = df["Profession"].unique()

df = df[df["Profession"] == "Student"]

df = df.drop(columns=["Profession"])


#Análisis Work Depression

df["Work Pressure"].unique()
conteo = df["Work Pressure"].value_counts(dropna=False)
print(conteo)

df = df[df["Work Pressure"] == 0]

df = df.drop(columns=["Work Pressure"])


#Análisis Job Satisfaction

df["Job Satisfaction"].unique()
conteo = df["Job Satisfaction"].value_counts(dropna=False)
print(conteo)

df = df[df["Job Satisfaction"] == 0 ]
df = df.drop(columns=["Job Satisfaction"])


#Análisis Sleep
categorias_Sleep = df["Sleep Duration"].unique()

# Crear variables dummy
dummies = pd.get_dummies(df["Sleep Duration"], prefix="Sleep Duration")

# Agregar las variables dummy al dataframe original
df = pd.concat([df, dummies], axis=1)

df = df.drop(columns=["Sleep Duration"])

#Análisis Dietary Habits
categorias_Sleep = df["Dietary Habits"].unique()

# Crear variables dummy
dummies = pd.get_dummies(df["Dietary Habits"], prefix="Dietary Habits")

# Agregar las variables dummy al dataframe original
df = pd.concat([df, dummies], axis=1)

df = df.drop(columns=["Dietary Habits"])


#Análisis Degree

conteo_Degree = df["Degree"].value_counts(dropna=False)

#Análisis Suicidal

df["numSuicidal"] = df["Have you ever had suicidal thoughts ?"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["Have you ever had suicidal thoughts ?"])


#Análisis Mental Illness

df["numIllness"] = df["Family History of Mental Illness"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["Family History of Mental Illness"])


from sklearn.model_selection import train_test_split

# Suponiendo que 'df' es tu DataFrame y 'Depression' es la columna de estratificación
X = df.drop(columns=["Depression", "id"]) 
y = df['Depression']  # Variable objetivo

# Dividir en training y test (estratificado por la variable 'Depression')
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42, 
                                                    stratify=y)


# 1. Calcular las proporciones en el conjunto de entrenamiento
city_counts_train = X_train['City'].value_counts(normalize=True)

# 2. Mapear estas proporciones a un nuevo DataFrame codificado
city_mapping_train = city_counts_train.to_dict()

# 3. Aplicar este encoding al conjunto de entrenamiento
X_train['City_encoded'] = X_train['City'].map(city_mapping_train)

# 4. Para el conjunto de test, usamos el mismo mapeo calculado en el entrenamiento
X_test['City_encoded'] = X_test['City'].map(city_mapping_train)

# 5. Asegurarnos de que las categorías en el test que no estaban en el entrenamiento se codifiquen como NaN o 0 (o cualquier valor que prefieras)
X_test['City_encoded'] = X_test['City_encoded'].fillna(0)

X_train = X_train.drop(columns=["City"])
X_test = X_test.drop(columns=["City"])




# 1. Calcular las proporciones en el conjunto de entrenamiento
degree_counts_train = X_train['Degree'].value_counts(normalize=True)

# 2. Mapear estas proporciones a un nuevo DataFrame codificado
degree_mapping_train = degree_counts_train.to_dict()

# 3. Aplicar este encoding al conjunto de entrenamiento
X_train['Degree_encoded'] = X_train['Degree'].map(degree_mapping_train)

# 4. Para el conjunto de test, usamos el mismo mapeo calculado en el entrenamiento
X_test['Degree_encoded'] = X_test['Degree'].map(degree_mapping_train)

# 5. Asegurarnos de que las categorías en el test que no estaban en el entrenamiento se codifiquen como NaN o 0 (o cualquier valor que prefieras)
X_test['Degree_encoded'] = X_test['Degree_encoded'].fillna(0)


X_train = X_train.drop(columns=["Degree"])
X_test = X_test.drop(columns=["Degree"])


#Imputo valores (hay unos pocos missings que se podrían eliminar pero se prefieren dejar)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # Usar la media para imputar valores faltantes

X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)

X_train = pd.DataFrame(X_train2, columns=X_train.columns)
X_test = pd.DataFrame(X_test2, columns=X_train.columns)


#MODELIZACIÓN

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

# Crear un diccionario con los modelos y sus parámetros sencillos
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, 
                             eval_metric='mlogloss', random_state=42),
    "Logistica": LogisticRegression(penalty=None, solver='lbfgs'),
    "Lasso": LogisticRegression(penalty='l1', solver='liblinear', C=1.0),
    "Ridge": LogisticRegression(penalty='l2', solver='lbfgs', C=1.0),
    "ElasticNet": LogisticRegression(penalty='elasticnet', solver='saga', 
                                     l1_ratio=0.5, C=1.0),
    #"SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

# Suponiendo que ya tienes los conjuntos X_train, y_train, X_test, y_test

# Lista para almacenar los Gini scores

gini_scores = []

# Entrenamiento y evaluación de cada modelo
for model_name, model in models.items():
    # Ajustar el modelo con X_train y y_train
    model.fit(X_train, y_train)
    
    # Predecir las probabilidades de las clases para X_test
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Probabilidad para la clase positiva    
    
    # Calcular el AUC y el índice de Gini
    auc = roc_auc_score(y_test, y_pred_prob)
    gini = 2 * auc - 1  # Cálculo del índice de Gini
    
    # Almacenar el resultado
    gini_scores.append((model_name, gini))

# Crear un DataFrame con los resultados
gini_df = pd.DataFrame(gini_scores, columns=["Model", "Gini Score"])

# Mostrar los resultados
print(gini_df)

# Encontrar el mejor modelo según el Gini Score
best_model = gini_df.loc[gini_df['Gini Score'].idxmax()]
print(f"\nMejor modelo según el índice de Gini: {best_model['Model']} con Gini: {best_model['Gini Score']}")


#Tabla de variables del modelo Ridge que ha salido como campeón

Ridge = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0)
modelo_ridge = Ridge.fit(X_train, y_train)
print(modelo_ridge.get_params())  # Ver parámetros del modelo

# Obtener los coeficientes
coeficientes = modelo_ridge.coef_

coef_df = pd.DataFrame(coeficientes.T, index=X_train.columns, columns=["Coeficiente"])
print(coef_df)


import matplotlib.pyplot as plt
from xgboost import plot_importance

# Asumiendo que ya tienes el modelo entrenado
best_model = models["XGBoost"]
best_model.fit(X_train, y_train)

# Crear el gráfico de importancia
plt.figure(figsize=(10, 8))
plot_importance(best_model, importance_type='weight', max_num_features=10, height=0.8)
plt.title('Importancia de las Variables en el Modelo XGBoost')
plt.show()


import shap
import numpy as np



# Inicializamos el explainer de SHAP para el modelo XGBoost
explainer = shap.Explainer(best_model, X_train)

# Seleccionamos el primer elemento del conjunto de test
X_test_single = X_test.iloc[[0]]

# Calculamos los valores SHAP para ese ejemplo
shap_values = explainer(X_test_single)

# Visualizamos el resumen de SHAP para el primer ejemplo
shap.initjs()
shap.plots.waterfall(shap_values[0])


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Obtener el modelo de Decision Tree
decision_tree_model = models["Decision Tree"]

# Dibujar el árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model, filled=True, feature_names=X_train.columns, class_names=['Clase 0', 'Clase 1'], rounded=True)
plt.show()



# Inicializamos el explainer de SHAP para el modelo XGBoost
explainer = shap.Explainer(best_model, X_train)

# Seleccionamos el primer elemento del conjunto de test
X_test_single = X_test.iloc[[144]]

# Calculamos los valores SHAP para ese ejemplo
shap_values = explainer(X_test_single)

# Visualizamos el resumen de SHAP para el primer ejemplo
shap.initjs()
shap.plots.waterfall(shap_values[0])


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Obtener el modelo de Decision Tree
decision_tree_model = models["Decision Tree"]

# Dibujar el árbol de decisión
plt.figure(figsize=(20, 10))
plot_tree(decision_tree_model, filled=True, feature_names=X_train.columns, class_names=['Clase 0', 'Clase 1'], rounded=True)
plt.show()




