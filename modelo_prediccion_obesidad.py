import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Se guardan los datos del cvs
df = pd.read_csv('ObesityDataSet.csv')

features = [
    'FCVC',   # Frecuencia consumo vegetales
    'NCP',    # Número de comidas
    'CAEC',   # Consumo de alimentos calóricos
    'CH2O',   # Consumo de agua
    'FAF',    # Actividad física
    'SCC',    # Monitorea calorías
    'CALC',   # Consumo de alcohol
    'TUE'     # Tiempo usando tecnología
]

target = 'NObeyesdad'  # ---> Nivel de obesidad

# Variables categóricas
categorical = ['CAEC', 'CALC', 'SCC', 'NObeyesdad']
label_encoders = {}

for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Dividir datos
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predicción
y_pred = model.predict(X_test_scaled)

# ---> Resultados
print("Precisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n")
print(classification_report(
    y_test, y_pred, target_names=label_encoders['NObeyesdad'].classes_
))

# Importancia de variables
importances = model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title("Importancia de las variables en el modelo")
plt.xlabel("Importancia")
plt.ylabel("Características")
plt.tight_layout()
plt.show()
