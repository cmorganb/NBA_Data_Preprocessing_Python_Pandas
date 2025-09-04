import pandas as pd
import os
import requests


def clean_data(path):
    """
    Receives the path of an NBA dataset and returns a cleaned dataframe by applying the following steps:
      - Parsing the b_day and draft_year features as datetime objects
      - Replacing the missing values in team feature with "No Team"
      - Taking the height feature in meters and removes the customary units
      - Taking the weight feature in meters and removes the customary units
      - Removing the extraneous $ character from the salary feature
      - Parsing height, weight, and salary features as floats
      - Categorizing the country feature as "USA" and "Not-USA"
      - Replacing the cells containing "Undrafted" in the draft_round feature with the string "0"
    """
    df = pd.read_csv(path)

    df['b_day'] = pd.to_datetime(df['b_day'], format="%m/%d/%y")
    df['draft_year'] = pd.to_datetime(df['draft_year'], format="%Y")
    df['team'] = df['team'].fillna("No Team")
    df['height'] =  df['height'].apply(lambda x: x.split("/")[1].strip().split()[0])
    df['weight'] = df['weight'].apply(lambda x: x.split("/")[1].strip().split()[0])
    df['salary'] = df['salary'].apply(lambda x: x.replace('$', ''))
    df[['height', 'weight', 'salary']] = df[['height', 'weight', 'salary']].astype(float)
    df['country'] = df['country'].apply(lambda x: 'Not-USA' if x != 'USA' else x)
    df['draft_round'] = df['draft_round'].apply(lambda x: '0' if x == 'Undrafted' else x)

    return df

def feature_data(df):
    """
    Receives a clean dataset and returns a dataset composed of feature data by:
    - Engineering the age, experience and bmi features
    - Dropping high cardinality and dependent features
    """

    df['version'] = pd.to_datetime(df['version'], format='NBA2k%y')
    df['age'] = df['version'].dt.year - df['b_day'].dt.year
    df['experience'] = df['version'].dt.year - df['draft_year'].dt.year
    df['bmi'] = df['weight'] / df['height'] ** 2

    df = df.drop(columns=['draft_year', 'b_day', 'weight', 'height', 'version'])

    # Identify categorical columns and drop the ones with high cardinality (50 or more unique values)
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    cols_to_drop = [col for col in df.columns if col in categorical_cols and df[col].nunique() >= 50]
    df = df.drop(columns=cols_to_drop)

    return df

def multicol_data(df):
    """
    Receives a dataframe and returns it without the multicollinear features
    """

    numeric_df = df.select_dtypes(include=['number'])
    df_corr = numeric_df.corr()

    for i in df_corr.index.to_list():
        for col in df_corr.columns:
            if i != 'salary' and col != 'salary' and i != col and abs(df_corr.loc[i, col]) > 0.5:
                if df_corr.loc[i, 'salary'] > df_corr.loc[col, 'salary']:
                    df = df.drop(columns=[col], errors='ignore')
                else:
                    df = df.drop(columns=[i], errors='ignore')

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

    clean_df = clean_data(data_path)
    feature_df = feature_data(clean_df)
    multicol_data(feature_df)

if __name__ == "__main__":
    main()

