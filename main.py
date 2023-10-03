from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

df_userdata = pd.read_parquet('src/items.parquet')
df_games = pd.read_parquet('src/games.parquet')
df_reviews = pd.read_parquet('src/reviews.parquet')


@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre(genero: str):
    try:
        if genero not in df_games.columns:
            return {"mensaje": f"El género '{genero}' no se encuentra en los datos"}

        filtered_df = df_games[df_games[genero] == 1]

        if len(filtered_df) == 0:
            return {"mensaje": f"No se encontraron datos para el género '{genero}'"}

        año_mas_horas = filtered_df.groupby('release_year')['playtimeforever'].sum().idxmax()

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
        
        result = {
            'Negative': sentiment_counts.get('Negativo', 0),
            'Neutral': sentiment_counts.get('Neutral', 0),
            'Positive': sentiment_counts.get('Positivo', 0)
        }
        return result
    except Exception as e:
        return {"error": str(e)}
