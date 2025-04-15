import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("ObesityDataSet.csv")  

columnas_categoricas = [
    'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC',
    'SMOKE', 'SCC', 'CALC', 'MTRANS'
]

le = LabelEncoder()
for col in columnas_categoricas:
    df[col] = le.fit_transform(df[col])

le_objetivo = LabelEncoder()
df['NObeyesdad'] = le_objetivo.fit_transform(df['NObeyesdad'])


#Separación de variables independientes y dependientes
X = df.drop(columns=['NObeyesdad'])  # Variables de entrada
y = df['NObeyesdad']                # Variable de salida

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

matriz_confusion = confusion_matrix(y_test, y_pred)
print("\n📊 Matriz de Confusión:\n", matriz_confusion)

print("\n📈 Reporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=le_objetivo.classes_))

plt.figure(figsize=(10, 6))
sns.heatmap(matriz_confusion, annot=True, fmt='d',
            xticklabels=le_objetivo.classes_,
            yticklabels=le_objetivo.classes_,
            cmap="YlGnBu")
plt.title("Matriz de Confusión - Regresión Logística")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
