import pandas as pd
import os
import requests


def clean_data(path):
    df = pd.read_csv(path)

    # Parsing the b_day and draft_year features as datetime objects
    df['b_day'] = pd.to_datetime(df['b_day'], format="%m/%d/%y")
    df['draft_year'] = pd.to_datetime(df['draft_year'], format="%Y")

    # Replacing the missing values in team feature with "No Team"
    df['team'] = df['team'].fillna("No Team")

    # Taking the height feature in meters and removes the customary units
    df['height'] =  df['height'].apply(lambda x: x.split("/")[1].strip().split()[0])

    # Taking the weight feature in meters and removes the customary units
    df['weight'] = df['weight'].apply(lambda x: x.split("/")[1].strip().split()[0])

    # Removing the extraneous $ character from the salary feature
    df['salary'] = df['salary'].apply(lambda x: x.replace('$', ''))

    # Parsing height, weight, and salary features as floats
    df[['height', 'weight', 'salary']] = df[['height', 'weight', 'salary']].astype(float)










    return df


def main():
    # Checking ../Data directory presence
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'nba2k-full.csv' not in os.listdir('../Data'):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/nba2k-full.csv', 'wb').write(r.content)
        print('Loaded.')

    data_path = "../Data/nba2k-full.csv"

    clean_data(data_path)

if __name__ == "__main__":
    main()

