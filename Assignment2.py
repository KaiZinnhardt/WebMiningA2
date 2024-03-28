import streamlit as st
#Spotify imports
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.client import SpotifyException
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

#Imports for visualizations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
#To visualize a Spider Radar
import plotly.graph_objects as go
import squarify
#To visualize to word of clouds
from wordcloud import WordCloud

from PIL import Image
import requests
from io import BytesIO

import threading
import time
from queue import Queue


#Credentials for the Spotify API
client = "081325a37839423fbe05b84de04ea7df"
secret = "ea97778035e449cfaeb4d2290f1ebbba"
username = "mt.prause"
client_credentials_manager = SpotifyClientCredentials(client_id=client, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

#Set a maximum limit for the runtime
def run_with_timeout(func, timeout, *args):
    print(*args)
    # Define a wrapper function to call the long-running function with arguments
    def func_wrapper(*args):
        #with st.report_thread():
            func(*args)

    # Create a thread to run the function
    thread = threading.Thread(target=func_wrapper,args=args)
    add_script_run_ctx(thread)
    thread.start()

    # Wait for the timeout
    thread.join(timeout)

    # Check if thread is still alive (meaning it has exceeded the timeout)
    if thread.is_alive():
        thread.join()
        raise TimeoutError("Building Wordcloud failed, unable to connect to API")


#Helper function to retrieve the playlist ID
def get_playlist_id(playlist_url):
    return playlist_url.split('/')[-1].split('?')[0]

# Function to retrieve playlist
def retrieve_playlist(playlist_url):
    try:
        playlist_id = get_playlist_id(playlist_url)
        playlist = sp.user_playlist(user=None, playlist_id=playlist_id)
        return playlist
    except SpotifyException as e:
        # Handle different types of Spotify exceptions
        if e.http_status == 404:
            print("Playlist not found.")
        elif e.http_status == 401:
            print("Unauthorized request.")
        else:
            print("An error occurred:", e)

#Retrieve general information about the playlis
def get_playlist_info(playlist_url):
  playlist_songs = []

  for t in sp.playlist_tracks(get_playlist_id(playlist_url))["items"]:
    try :
        artist = sp.artist(t['track']["artists"][0]["external_urls"]["spotify"])
        genres_list = list(map(lambda x: x.replace(' ', '-'), artist['genres']))
        songs_data = {
                'id':t['track']['id'],
                'song name':t['track']['name'],
                'artist name':t['track']['artists'][0]['name'],
                'artist genres': ' '.join(genres_list),
                'release_date':t['track']['album']['release_date'],
                'song link':t['track']['external_urls']['spotify'],
                'image': t['track']['album']['images'][0]['url'],
                'song duration':t['track']['duration_ms'],
                'song popularity' : t['track']['popularity']
        }
        playlist_songs.append(songs_data)
    except :
        print("An error occurred:", e)
  df_playlist_songs = pd.DataFrame(playlist_songs)
  return df_playlist_songs

#Retrieve the ID for each song from the playlist
def get_songs_id_playlist(play_list):
  play_tracks = play_list['tracks']
  play_list_songs = play_tracks['items']
  print(play_tracks['next'])
  while play_tracks['next']:
    play_tracks = sp.next(play_tracks)
    play_list_songs.extend(play_tracks['items'])
  songs_id = [play_list_songs[i]['track']['id'] for i in range(0, len(play_list_songs))]
  return songs_id

#Retrieve the ID for each song from the playlist
def get_name_playlist(playlist):
  return playlist['name']
#Retrieve the URL for the playlist image
def get_image_url(playlist):
  return playlist['images'][0]['url']

# get artists
def extract_artist_names(data):
    artist_names = []

    # Navigate through the JSON structure to access each artist name
    for item in data['tracks']['items']:
        for artist in item['track']['album']['artists']:
            artist_names.append(artist['name'])

    return artist_names

def get_audio_features(songs_list_id,num_play_list):
  features=[]
  for i in range(0, len(songs_list_id), 50):
    try:
      audio_features = sp.audio_features(songs_list_id[i:i + 50])
      for track in audio_features:
          if track is not None:
              features.append(track)
              features[len(features) - 1]['class'] = num_play_list
    except SpotifyException as e:
      # Handle different types of Spotify exceptions
      if e.http_status == 404:
          print("Playlist not found.")
      elif e.http_status == 401:
          print("Unauthorized request.")
      else:
          print("An error occurred:", e)

  return features

# get popularity
def extract_popularity_score(data):
    scores = []

    # Navigate through the JSON structure to access the release date of each track's album
    for item in data['tracks']['items']:
        scores.append(int(item['track']['popularity']))

    return scores

# get years
def extract_release_years(data):
    release_years = []

    # Navigate through the JSON structure to access the release date of each track's album
    for item in data['tracks']['items']:
        release_date = item['track']['album']['release_date']
        release_year = release_date.split('-')[0]  # Assuming the format is 'YYYY-MM-DD'
        release_years.append(int(release_year))

    return release_years

def plot_treemap(ax, artist_names, title):
    # Count the frequency of each artist
    artist_counts = {}
    for name in artist_names:
        artist_counts[name] = artist_counts.get(name, 0) + 1

    # Sort and select top 20 artists
    sorted_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)
    top_artists = dict(sorted_artists[:20])

    # Count for "Others"
    others_count = sum(count for name, count in sorted_artists if (count == 1 or name not in top_artists))

    # Remove artists with only one song from top artists
    top_artists = {name: count for name, count in top_artists.items() if count > 1}

    # Add "Others" category
    if others_count > 0:
        top_artists['Others'] = others_count

    # Prepare data for the treemap
    labels = [f'{name} ({count})' for name, count in top_artists.items()]
    sizes = list(top_artists.values())

    # Assign colors
    colors = ['#D3D3D3' if name == 'Others' else plt.cm.Paired(i/len(top_artists)) for i, name in enumerate(top_artists)]

    # Assign font size
    text_kwargs = {'fontsize': 8}

    # Create the treemap
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax, text_kwargs=text_kwargs)
    ax.axis('off')
    ax.set_title(title)

def fig_artists(play_list_1, play_list_2,play_list_name_1, play_list_name_2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first treemap
    artist_names1 = extract_artist_names(play_list_1)
    plot_treemap(axes[0], artist_names1, 'Artist Distribution: ' + play_list_name_1)

    # Plot the second treemap
    artist_names2 = extract_artist_names(play_list_2)
    plot_treemap(axes[1], artist_names2, 'Artist Distribution: ' + play_list_name_2)

    # Display the treemaps
    plt.tight_layout()
    st.pyplot(fig)

def cloud_of_words(url1, url2, play_list_name_1, play_list_name_2): #Create the cloud of Words for the Genres of the Artists
    df_genre_1 = get_playlist_info(url1)
    df_genre_2 = get_playlist_info(url2)  #("https://open.spotify.com/playlist/37i9dQZF1DZ06evO0XObfi?si=adcdfd01c74747a9")

    text_1 = ' '.join(df_genre_1['artist genres'])
    text_2 = ' '.join(df_genre_2['artist genres'])

    # Generate word cloud
    wordcloud_1 = WordCloud(width=800, height=400, background_color='white').generate(text_1)
    wordcloud_2 = WordCloud(width=800, height=400, background_color='white').generate(text_2)

    # Display the word cloud using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(wordcloud_1, interpolation='bilinear')
    axes[0].set_title('Genre Word Cloud: ' + play_list_name_1, fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(wordcloud_2, interpolation='bilinear')
    axes[1].set_title('Genre Word Cloud: ' + play_list_name_2, fontsize=16)
    axes[1].axis('off')
    st.pyplot(fig)

def spider_radar_avg(df_play_list_1,df_play_list_2,play_list_name_1,play_list_name_2):
    spider_attributes = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                         'valence']
    df_spider_1 = df_play_list_1[spider_attributes]
    df_spider_2 = df_play_list_2[spider_attributes]
    spider_avg_1 = df_spider_1.mean()
    spider_avg_2 = df_spider_2.mean()
    avg_df = pd.DataFrame({
        'Category': ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                     'valence'],
        'Average_1': spider_avg_1,
        'Average_2': spider_avg_2
    })
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_df['Average_1'],
        theta=avg_df['Category'],
        fill='toself',
        name=play_list_name_1
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_df['Average_2'],
        theta=avg_df['Category'],
        fill='toself',
        name=play_list_name_2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Compares the strength of each Audio feature that ranges between 0 and 1 based on averages"
    )

    #st.pyplot(fig)
    st.plotly_chart(fig)

def spider_radar_stdev(df_play_list_1,df_play_list_2,play_list_name_1,play_list_name_2):
    spider_attributes = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                         'valence']
    df_spider_1 = df_play_list_1[spider_attributes]
    df_spider_2 = df_play_list_2[spider_attributes]
    spider_stdev_1 = df_spider_1.std()
    spider_stdev_2 = df_spider_2.std()
    stdev_df = pd.DataFrame({
        'Category': ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness',
                     'valence'],
        'Average_1': spider_stdev_1,
        'Average_2': spider_stdev_2
    })
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=stdev_df['Average_1'],
        theta=stdev_df['Category'],
        fill='toself',
        name=play_list_name_1
    ))
    fig.add_trace(go.Scatterpolar(
        r=stdev_df['Average_2'],
        theta=stdev_df['Category'],
        fill='toself',
        name=play_list_name_2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True
            )),
        showlegend=True,
        title="Compares the variation of each Audio Feature"
    )

    st.plotly_chart(fig)

def fig_popularity(playlist_1, playlist_2,play_list_name_1,play_list_name_2):
    popularity_scores_playlist1 = extract_popularity_score(playlist_1)  # Replace with your scores for playlist 1
    popularity_scores_playlist2 = extract_popularity_score(playlist_2)  # Replace with your scores for playlist 2

    df1 = pd.DataFrame({'Playlist': play_list_name_1, 'Popularity Score': popularity_scores_playlist1})
    df2 = pd.DataFrame({'Playlist': play_list_name_2, 'Popularity Score': popularity_scores_playlist2})

    # Concatenate the two DataFrames
    df = pd.concat([df1, df2], ignore_index=True)

    # Create the boxplot
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(x='Playlist', y='Popularity Score', data=df)
    plt.title('Boxplot of Popularity Scores of the Playlist Items')
    st.pyplot(fig)

def fig_song_attributes(df_play_list_1,df_play_list_2,play_list_name_1,play_list_name_2,play_list_1,play_list_2):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes = axes.flatten()

    #Duration
    df_play_list_1['duration_sec'] = df_play_list_1['duration_ms'] / 1000
    df_play_list_2['duration_sec'] = df_play_list_2['duration_ms'] / 1000
    sns.kdeplot(df_play_list_1['duration_sec'], color='red', alpha=0.5, label=play_list_name_1, ax=axes[0])
    sns.kdeplot(df_play_list_2['duration_sec'], color='blue', alpha=0.5, label=play_list_name_2, ax=axes[0])
    axes[0].set_xlabel('Time in sec')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Density of the two Playlists Regarding \n the Duration of the Playlist Items')
    legend = axes[0].legend()
    legend.fontsize = 'x-small'

    #Tempo
    sns.kdeplot(df_play_list_1['tempo'], color='red', alpha=0.5, label=play_list_name_1, ax=axes[1])
    sns.kdeplot(df_play_list_2['tempo'], color='blue', alpha=0.5, label=play_list_name_2, ax=axes[1])
    axes[1].set_xlabel('Tempo in Beats per Minute')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Density of the two Playlists Regarding\n the Tempo of the Playlist Items')
    legend = axes[1].legend()
    legend.fontsize = 'x-small'

    #Loudness
    sns.kdeplot(df_play_list_1['loudness'], color='red', alpha=0.5, label=play_list_name_1, ax=axes[2])
    sns.kdeplot(df_play_list_2['loudness'], color='blue', alpha=0.5, label=play_list_name_2, ax=axes[2])
    axes[2].set_xlabel('Loudness in Decibels')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Density of the two Playlists Regarding\n the Loudness of the Playlist Items')
    legend = axes[2].legend()
    legend.fontsize = 'x-small'

    #Release Years
    # Extract release years
    years1 = extract_release_years(play_list_1)
    years2 = extract_release_years(play_list_2)

    # Combine the years for bin calculation
    combined_years = years1 + years2

    # Determine dynamic bin width or edges
    min_year, max_year = min(combined_years), max(combined_years)
    bins = range(min_year, max_year + 1)  # One bin per year

    # Create a DataFrame for easier plotting with Seaborn
    df_years1 = pd.DataFrame({'Playlist 1': years1})
    df_years2 = pd.DataFrame({'Playlist 2': years2})

    sns.kdeplot(df_years1['Playlist 1'], color='red', alpha=0.5, label=play_list_name_1, ax=axes[3])
    sns.kdeplot(df_years2['Playlist 2'], color='blue', alpha=0.5, label=play_list_name_2, ax=axes[3])
    axes[3].set_xlabel('Distribution of Release Years')
    axes[3].set_ylabel('Frequency')
    axes[3].set_title('Density of the two Playlists Regarding\n the Release Year of the Playlist Items')
    legend = axes[3].legend()
    legend.fontsize = 'x-small'


    plt.subplots_adjust(top=1,bottom=0.000001)#,wspace=0.5,hspace=2)

    st.pyplot(fig)



def main():
    st.title("Playlist Comparison")

    st.write("Enter URLs of Spotify Playlists below:")
    url1 = st.text_input("URL 1")
    url2 = st.text_input("URL 2")

    if st.button("Compare Playlists"):
        playlist_1 = retrieve_playlist(url1)
        playlist_2 = retrieve_playlist(url2)

        songs_list_1 = get_songs_id_playlist(playlist_1)
        songs_list_2 = get_songs_id_playlist(playlist_2)
        play_list_name_1 = get_name_playlist(playlist_1)
        play_list_name_2 = get_name_playlist(playlist_2)

        audio_features_play_list_1 = get_audio_features(songs_list_1, 1)
        audio_features_play_list_2 = get_audio_features(songs_list_2, 0)
        audio_features = audio_features_play_list_1 + audio_features_play_list_2

        df = pd.DataFrame(audio_features)
        non_features = ['analysis_url', 'id', 'track_href', 'type', 'uri']
        df.drop(labels=non_features, axis=1, inplace=True)

        df_play_list_1 = df[df['class'] == 1]
        df_play_list_2 = df[df['class'] == 0]

        # URLs of the images
        image_url1 = get_image_url(playlist_1)
        image_url2 = get_image_url(playlist_2)

        # Download images
        response1 = requests.get(image_url1)
        response2 = requests.get(image_url2)
        image1 = Image.open(BytesIO(response1.content))
        image2 = Image.open(BytesIO(response2.content))

        # Display images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### *"+play_list_name_1+"*")  # Bigger and bolder title
            st.image(image1)

        with col2:
            st.markdown("### *"+play_list_name_2+"*")  # Bigger and bolder title
            st.image(image2)

        st.markdown("""### Word of Clouds for Genres Contained in the Playlist""")
        placeholder = st.empty()
        #cloud_of_words(url1, url2, play_list_name_1, play_list_name_2)
        st.markdown("""### Artist Distribution in the Playlist""")
        fig_artists(playlist_1, playlist_2, play_list_name_1, play_list_name_2)
        with st.expander("Description of Tree Plot"):
            st.markdown("""
                The tree plot shows the distribution of artists present in each playlist. The 20 most frequent artists that are present at least twice in the playlist are displayed individually. The remaining artists are aggregated in the "Others" category. The frequency is presented in parenthesis.
            """)
        spider_radar_avg(df_play_list_1, df_play_list_2,play_list_name_1,play_list_name_2)
        with st.expander("Description of Audio Attributes"):
            st.markdown("""
                **Danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.\n
                **Acousticness:** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.\n
                **Energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.\n
                **Instrumentalness:** Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.\n
                **Liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.\n
                **Speachiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.\n
                **Valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).\n
                
            """)
        spider_radar_stdev(df_play_list_1, df_play_list_2, play_list_name_1, play_list_name_2)
        with st.expander("Description of Audio Attributes"):
            st.markdown("""
                **Danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.\n
                **Acousticness:** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.\n
                **Energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.\n
                **Instrumentalness:** Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.\n
                **Liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.\n
                **Speachiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.\n
                **Valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).\n

            """)
        #    fig_song_duration(df_play_list_1, df_play_list_2,play_list_name_1,play_list_name_2)
        #with st.expander("Tempo of Songs"):
        #    fig_song_tempo(df_play_list_1, df_play_list_2,play_list_name_1,play_list_name_2)
        #with st.expander("Loudness of Songs"):
        #    fig_song_loudness(df_play_list_1, df_play_list_2,play_list_name_1,play_list_name_2)
        fig_popularity(playlist_1, playlist_2, play_list_name_1, play_list_name_2)
        with st.expander("Description of Popularity"):
            st.markdown("""
                **Popularity:** The popularity of the track. The value will be between 0 and 100, with 100 being the most popular. \n
                The popularity of a track is a value between 0 and 100, with 100 being the most popular. The popularity is calculated by algorithm and is based, in the most part, on the total number of plays the track has had and how recent those plays are. \n
                Generally speaking, songs that are being played a lot now will have a higher popularity than songs that were played a lot in the past. Duplicate tracks (e.g. the same track from a single and an album) are rated independently. Artist and album popularity is derived mathematically from track popularity. Note: the popularity value may lag actual popularity by a few days: the value is not updated in real time.
            """)
        st.markdown("""### Density Plots of Additional Attributes""")
        fig_song_attributes(df_play_list_1, df_play_list_2,play_list_name_1,play_list_name_2,playlist_1,playlist_2)

        with placeholder.container():
            with st.spinner('Building Wordcloud'):
                try:
                    #cloud_of_words(url1, url2, play_list_name_1, play_list_name_2)
                    run_with_timeout(cloud_of_words,120,url1, url2, play_list_name_1, play_list_name_2)
                except TimeoutError as e:
                    st.markdown("Building Wordcloud failed, unable to connect to API\n",e)

if __name__ == "__main__":
    main()
    #https://open.spotify.com/playlist/37i9dQZEVXbLp5XoPON0wI
    #https://open.spotify.com/playlist/37i9dQZF1DXcF6B6QPhFDv