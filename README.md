# Predicción del Nivel de Obesidad con Machine Learning

Este repositorio contiene un proyecto de Machine Learning que tiene como objetivo predecir el nivel de obesidad de un individuo a partir de variables asociadas a hábitos alimenticios y condición física. El pipeline de solución incluye la preparación de datos, codificación de variables, escalado, entrenamiento de diferentes modelos (por ejemplo, Random Forest y Decision Tree) y visualización de resultados.

## Descripción del Proyecto

El modelo utiliza el dataset **ObesityDataSet.csv** para entrenar un clasificador supervisado. Se seleccionaron variables relevantes como:
- **FCVC**: Frecuencia de consumo de vegetales
- **NCP**: Número de comidas diarias
- **CAEC**: Consumo de alimentos calóricos
- **CH20**: Consumo de agua
- **FAF**: Actividad física
- **CALC**: Consumo de alcohol
- **SCC**: Monitoreo de calorías
- **TUE**: Tiempo usando tecnología

El proyecto respeta los estándares de programación [PEP 8](https://pep8.org/) y se implementa en Python 3, utilizando bibliotecas como `pandas`, `scikit-learn`, `matplotlib` y `seaborn`.

## Contenido del Repositorio

- **modelo_prediccion_obesidad.py**: Script principal que contiene el pipeline completo del modelo de Machine Learning (preprocesamiento, entrenamiento, evaluación y visualización de la importancia de variables).
- **ObesityDataSet.csv**: Dataset empleado para entrenar y validar el modelo.
- **requisitos.txt**: Archivo que lista las dependencias del proyecto.
- **README.md**: Este archivo.
