import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def csvtofeather(filepath):
    '''
    Convert csv to feather format

    Parameters
    ----------
    filepath: str
        Path to csv file

    Returns
    -------
    dataframe: pd.DataFrame
        Dataframe of the file
    '''
    ## Read the csv file
    df = pd.read_csv(
    filepath,
    header=None,
    error_bad_lines=False,
    infer_datetime_format=True,
    parse_dates=[4],
    dtype={
        0: np.int32,
        1: bool,
        2: np.float32,
        3: np.float32,
        5: np.uint8,
        6: np.int8,
        7: bool,
        8: np.int8,
        9: np.int8,
        },
    )

    ## Rename the columns
    df.columns = ['car_id', 'status', 'lon', 'lat', 'timestamp', 'velocity', 'direction', 'gps_effectiveness', 'unknown_1', 'unknown_2']

    print(f"Dataframe info before processing: \n")
    print(df.info())

    ## Remove car id with more than 5 digits
    df = df[df['car_id'] < 99999]

    ## Drop rows where data has no variation
    if df['unknown_1'].std() == 0:
        df = df.drop(columns=['unknown_1'])
        print('Dropped unknown_1 column')
    if df['unknown_2'].std() == 0:
        df = df.drop(columns=['unknown_2'])
        print('Dropped unknown_2 column')

    ## drop rows where gps effectiveness is false
    df = df[df['gps_effectiveness'] == True]

    ## drop rows with velocity above 150 and below 0
    df = df[df['velocity'] >= 0]
    df = df[df['velocity'] < 150]

    ## Filter the lat and lon to roughly within Shanghai city
    # df = df.query('lat > 30.6 and lat < 31.8 and lon > 120.8 and lon < 122')
    # df = df.query('lat > 30.65 and lat < 31.55 and lon > 120.95 and lon < 122') # This is a 100 square km box
    df = df.query('lat > 31.245026 and lat < 31.356345 and lon > 121.453877 and lon < 121.580795') # Master area

    ## filter the data to the correct date
    date = filepath.replace('gps', '').replace('.csv', '')
    df = df[df['timestamp'].dt.date == pd.to_datetime(date).date()]

    ## transform from latlong to web mercator
    k = 6378137
    df["x"] = df['lon'] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df['lat']) * np.pi/360.0)) * k

    ## drop the lat and lon columns
    df = df.drop(columns=['lon', 'lat'])

    ## cast x and y to float32
    df['x'] = df['x'].astype('float32')
    df['y'] = df['y'].astype('float32')

    ## Remove duplicate rows
    df = df.drop_duplicates()

    ## Sort dataframe to neighboring points
    centerpoints = pd.read_csv('centerpoints.csv')
    df = keep_neighbors(df, centerpoints)

    ## Save the file as a feather file
    feather_filepath = filepath.replace('.csv', '.feather')
    df = df.reset_index()
    df['index'] = pd.to_numeric(df['index'], downcast='unsigned')
    df.to_feather(feather_filepath)

    print(f"Dataframe after processing: \n")
    print(df.info())

    print(f'File saved as {filepath}.feather')
    return df


def keep_neighbors(point_df, centers_df):
    '''
    Keep only the points that are within a certain distance of the centers

    Parameters
    ----------
    point_df: pd.DataFrame
        Dataframe of points
    centers_df: pd.DataFrame
        Dataframe of centers

    Returns
    -------
    point_df: pd.DataFrame
        Dataframe of points that are within a certain distance of the centers
    '''
    ## Convert the dataframes to numpy arrays
    points = point_df[['x', 'y']].values
    centers = centers_df[['x', 'y']].values

    ## Create a KDTree
    tree = KDTree(centers)

    ## Find the nearest neighbors
    dist, ind = tree.query(points)

    ## Keep only the points that are within a certain distance of the centers
    point_df = point_df[dist < 200]

    return point_df


# csvtofeather('gps2016-01-23.csv')
