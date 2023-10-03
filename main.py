from fastapi import FastAPI
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

df_userdata = pd.read_parquet('src/items.parquet')
df_games = pd.read_csv('src/games.csv')
df_reviews = pd.read_csv('src/reviews.csv')

@app.get("/")
async def root():
    return { ''' Hola! Este es un proyecto individual para la carrera de Data Science de Henry. Te recomiendo leer el README'''}

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):
    try:
        # Verificar si el género especificado existe como columna en el DataFrame
        if genero not in df_games.columns:
            return {"mensaje": f"El género '{genero}' no se encuentra en los datos"}

        # Filtrar el DataFrame para obtener solo las filas donde el género sea 1 (verdadero)
        filtered_df = df_games[df_games[genero] == 1]

        if len(filtered_df) == 0:
            return {"mensaje": f"No se encontraron datos para el género '{genero}'"}

        # Encontrar el año con más horas jugadas entre los juegos de ese género
        año_mas_horas = filtered_df.groupby('release_year')['playtimeforever'].sum().idxmax()

        return {f"Año de lanzamiento con más horas jugadas para el género '{genero}'": año_mas_horas}
    except Exception as e:
        return {"error": str(e)}

@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
    try:
        # Verificar si el género especificado existe como columna en el DataFrame
        if genero not in df_userdata.columns:
            return {"mensaje": f"El género '{genero}' no se encuentra en los datos"}

        # Filtrar el DataFrame para obtener solo las filas donde el género sea 1 (verdadero)
        filtered_df = df_userdata[df_userdata[genero] == 1]

        if len(filtered_df) == 0:
            return {"mensaje": f"No se encontraron datos para el género '{genero}'"}

        # Encontrar el usuario con más horas jugadas para ese género
        usuario_mas_horas = filtered_df.groupby('user_id')['playtimeforever'].sum().idxmax()

        return {
            f"Usuario con más horas jugadas para el género '{genero}'": usuario_mas_horas}
    except Exception as e:
        return {"error": str(e)}

@app.get('/UsersRecommend/{anio}')
def UsersRecommend(anio: int):
    try:
        
        # Filtramos el DataFrame por el año especificado
        df_filtered = df_reviews[df_reviews['year'] == anio]
        
        # Filtramos por recomendaciones positivas o neutrales
        df_filtered = df_filtered[(df_filtered['recommend'] == True) & 
                                ((df_filtered['sentiment_analysis'] == 'Positivo') | 
                                (df_filtered['sentiment_analysis'] == 'Neutral'))]
        
        # Contamos la cantidad de recomendaciones por juego
        recommendations_count = df_filtered['app_name'].value_counts()
        
        # Obtenemos los 3 juegos más recomendados
        top_3_games = recommendations_count.head(3)
        
        # Creamos la lista de resultados en el formato deseado
        result = [{"Puesto {}: {}".format(i+1, game): count} for i, (game, count) in enumerate(top_3_games.iteritems())]
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get('/UsersNotRecommend/{anio}')
def UsersNotRecommend(anio: int):
    try:
        # Filtramos el DataFrame por el año especificado
        df_filtered = df_reviews[df_reviews['year'] == anio]
        
        # Filtramos por recomendaciones negativas y comentarios negativos
        df_filtered = df_filtered[(df_filtered['recommend'] == False) & 
                                (df_filtered['sentiment_analysis'] == 'Negativo')]
        
        # Contamos la cantidad de juegos que no fueron recomendados y eran negativos
        not_recommendations_count = df_filtered['app_name'].value_counts()
        
        # Obtenemos los 3 juegos menos recomendados
        bottom_3_games = not_recommendations_count.head(3)
        
        # Creamos la lista de resultados en el formato deseado
        result = [{"Puesto {}: {}".format(i+1, game): count} for i, (game, count) in enumerate(bottom_3_games.iteritems())]
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get('/SentimentAnalysis/{anio}')
def sentiment_analysis(anio: int):
    try:
        # Filtrar el DataFrame por el año especificado
        df_filtered = df_reviews[df_reviews['year'] == anio]
        
        # Contar la cantidad de revisiones en cada categoría de sentimiento
        sentiment_counts = df_filtered['sentiment_analysis'].value_counts()
        
        # Convertir los valores a tipos nativos de Python para evitar el error de serialización
        sentiment_counts = sentiment_counts.to_dict()
        
        # Crear un diccionario con los resultados en el formato deseado
        result = {
            'Negative': sentiment_counts.get('Negativo', 0),
            'Neutral': sentiment_counts.get('Neutral', 0),
            'Positive': sentiment_counts.get('Positivo', 0)
        }
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get('/recomendacion_juego/{id}')
def recomendacion_juego_por_id(id_producto: int):
    try:
        num_recomendaciones=5
        df = df_games.drop(columns=['tags','specs','developer'])
        juego = df[df['id'] == id_producto]

        if juego.empty:
            return "Juego no encontrado"

        # Normaliza las características del juego
        juego_caracteristicas = juego.drop(columns=['id','playtimeforever','app_name'])
        juego_caracteristicas_normalized = (juego_caracteristicas - juego_caracteristicas.min()) / (juego_caracteristicas.max() - juego_caracteristicas.min())
        juego_caracteristicas_normalized = juego_caracteristicas_normalized.fillna(0)
        # Calcula la similitud de coseno entre el juego y todos los demás juegos
        similarity_scores = cosine_similarity(juego_caracteristicas_normalized, df.drop(columns=['id']).values)
        # Encuentra los juegos más similares (excluyendo el juego de entrada)
        similar_games_indices = similarity_scores.argsort()[0][-num_recomendaciones-1:-1][::-1]
        similar_games = df_games.loc[similar_games_indices, 'app_name']

        return similar_games.tolist()
    except Exception as e:
        return {"error": str(e)}


