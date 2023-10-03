from fastapi import FastAPI
import pandas as pd
import numpy as np
from fastapi import FastAPI, Depends
#from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

df_userdata = pd.read_parquet('src/items.parquet')
df_games = pd.read_parquet('src/games.parquet')
df_reviews = pd.read_parquet('src/reviews.parquet')

# Define la función get_df_games como una dependencia
def get_df_games():
    return df_games

@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str, df_games: pd.DataFrame = Depends(get_df_games)):
    try:
        if genero not in df_games.columns:
            return {"mensaje": f"El género '{genero}' no se encuentra en los datos"}
        
        filtered_df = df_games[df_games[genero] == 1]

        if len(filtered_df) == 0:
            return {"mensaje": f"No se encontraron datos para el género '{genero}'"}

        año_mas_horas = filtered_df.groupby('release_year')['playtimeforever'].sum().idxmax()
        
        año_mas_horas = int(año_mas_horas)

        return {f"Año de lanzamiento con más horas jugadas para el género '{genero}'": año_mas_horas}
    except Exception as e:
        return {"error": str(e)}


@app.get('/UserForGenre/{genero}')
def UserForGenre(genero: str):
    try:
        if genero not in df_userdata.columns:
            return {"mensaje": f"El género '{genero}' no se encuentra en los datos"}

        filtered_df = df_userdata[df_userdata[genero] == 1]

        if len(filtered_df) == 0:
            return {"mensaje": f"No se encontraron datos para el género '{genero}'"}

        usuario_mas_horas = filtered_df.groupby('user_id')['playtimeforever'].sum().idxmax()
        
        return {
            f"Usuario con más horas jugadas para el género '{genero}'": usuario_mas_horas}
    except Exception as e:
        return {"error": str(e)}

@app.get('/UsersRecommend/{anio}')
def UsersRecommend(anio: int):
    try:
        
        df_filtered = df_reviews[df_reviews['year'] == anio]
        
        df_filtered = df_filtered[(df_filtered['recommend'] == True) & 
                                ((df_filtered['sentiment_analysis'] == 'Positivo') | 
                                (df_filtered['sentiment_analysis'] == 'Neutral'))]
        
        recommendations_count = df_filtered['app_name'].value_counts()
        
        top_3_games = recommendations_count.head(3)
        
        
        result = [{"Puesto {}: {}".format(i+1, game): count} for i, (game, count) in enumerate(top_3_games.iteritems())]
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get('/UsersNotRecommend/{anio}')
def UsersNotRecommend(anio: int):
    try:
        df_filtered = df_reviews[df_reviews['year'] == anio]
        
        df_filtered = df_filtered[(df_filtered['recommend'] == False) & 
                                (df_filtered['sentiment_analysis'] == 'Negativo')]
        
        not_recommendations_count = df_filtered['app_name'].value_counts()
        
        bottom_3_games = not_recommendations_count.head(3)
        
        del df_filtered
        
        result = [{"Puesto {}: {}".format(i+1, game): count} for i, (game, count) in enumerate(bottom_3_games.iteritems())]
        
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get('/SentimentAnalysis/{anio}')
def sentiment_analysis(anio: int):
    try:
        df_filtered = df_reviews[df_reviews['year'] == anio]
        
        sentiment_counts = df_filtered['sentiment_analysis'].value_counts()
        
        sentiment_counts = sentiment_counts.to_dict()
        
        del df_filtered
        
        result = {
            'Negative': sentiment_counts.get('Negativo', 0),
            'Neutral': sentiment_counts.get('Neutral', 0),
            'Positive': sentiment_counts.get('Positivo', 0)
        }
        return result
    except Exception as e:
        return {"error": str(e)}


'''@app.get('/recomendacion_juego/{id}')
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
        return {"error": str(e)}'''