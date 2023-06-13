from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import seaborn as sns
from rasterio.plot import show
import copy
import scipy.signal as signal

def transform_to_geodetic(coords):
    '''
    Transform coordinates from Web Mercator to WGS 84
    :param coords: (x, y) coordinates in Web Mercator
    :return: (lat, lon) coordinates in WGS 84
    '''
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")
    return transformer.transform(coords[0], coords[1])

#function to find the nearest point with docstring
def wgs84_to_web_mercator(df, lon="x", lat="y"):
    '''
    Converts decimal longitude/latitude to Web Mercator format
    (EPSG:4326) -> (EPSG:3857)
    :param df: dataframe with columns 'lon' and 'lat', containing
    decimal longitude and latitude values
    :param lon: name of the longitude column (default: 'lon')
    :param lat: name of the latitude column (default: 'lat')
    :return: dataframe with columns 'x' and 'y', containing
    coordinates in Web Mercator format
    '''
    k = 6378137
    df["x"] = df[lon] * (k * np.pi/180.0)
    df["y"] = np.log(np.tan((90 + df[lat]) * np.pi/360.0)) * k
    return df

def find_angle(x, y, neighbors_points):
    '''
    Find the angle between the central point and its neighbors.
    The angle is relative to the x-axis.
    :param x: x coordinate of the central point
    :param y: y coordinate of the central point
    :param neighbors_points: (x, y) coordinates of the neighbors
    :return: dataframe with the angle between the central point and each neighbor
    '''
    df = neighbors_points.copy()
    df['angle'] = np.arctan2(neighbors_points['y'] - y, neighbors_points['x'] - x)
    df['angle'] = df['angle'].apply(lambda x: x * 180 / np.pi)
    df['angle'] = df['angle'].apply(lambda x: x + 360 if x < 0 else x)
    return df


def find_angle_absolute(x, y, neighbors_points):
    '''
    Find the angle between the central point and its neighbors.
    The angle is relative to the true north.
    :param x: x coordinate of the central point
    :param y: y coordinate of the central point
    :param neighbors_points: (x, y) coordinates of the neighbors
    :return: dataframe with the angle between the central point and each neighbor
    '''
    df = neighbors_points.copy()

    # Transform central point and neighbors_points to WGS 84
    center_lat, center_lon = transform_to_geodetic((x, y))
    df['lat'], df['lon'] = zip(*df.apply(lambda row: transform_to_geodetic((row['x'], row['y'])), axis=1))

    # Calculate bearing to true north for each point
    center_rad = np.radians((center_lat, center_lon))
    points_rad = np.radians(df[['lat', 'lon']].values)

    delta_lon = points_rad[:,1] - center_rad[1]
    y = np.sin(delta_lon) * np.cos(points_rad[:,0])
    x = np.cos(center_rad[0]) * np.sin(points_rad[:,0]) - np.sin(center_rad[0]) * np.cos(points_rad[:,0]) * np.cos(delta_lon)
    bearings = np.arctan2(y, x)

    # Convert bearings to degrees and ensure they are in [0, 360) range
    df['angle'] = np.degrees(bearings)
    df['angle'] = df['angle'].apply(lambda x: x + 360 if x < 0 else x)

    return df

def count_peaks(central_point, neighbors_points, n=20, return_histogram=False):
    '''
    Count the number of peaks in the histogram of the angles between the central point and its neighbors.
    :param central_point: (x, y) coordinates of the central point
    :param neighbors_points: (x, y) coordinates of the neighbors
    :param n: number of bins in the histogram
    :return: number of peaks, peaks
    '''
    histogram, edges = np.histogram(find_angle_absolute(central_point[0], central_point[1], neighbors_points)['angle'], bins=n, range=(0, 360))
    histogram = np.insert(histogram, 0, 0)
    histogram = np.append(histogram, 0)
    histogram = np.where(histogram < np.max(histogram)*0.15, 0, histogram)
    peaks = signal.find_peaks(histogram, distance=2, height=0.3*histogram.max())[0]
    # Uncomment to plot the histogram and the peaks
    # plt.plot(histogram)
    # plt.plot(peaks, histogram[peaks], 'x')
    # plt.show()
    if return_histogram:
        return len(peaks), peaks, histogram, edges
    return len(peaks), peaks

def ransac_line_fitting(point_df, max_trials=50, min_samples=2, residual_threshold=1):
    '''
    RANSAC algorithm for line fitting.
    :param point_df: dataframe with columns 'x' and 'y' containing the coordinates of the points
    :param max_trials: maximum number of trials
    :param min_samples: minimum number of samples
    :param residual_threshold: threshold for the residual
    :return: best model, best inliers, best inlier count, best residual type, boolean for vertical line
    '''
    best_model = None
    best_inliers = None
    best_inlier_count = 0
    best_residual_type = None
    vertical_line = False

    for _ in range(max_trials):
        # Select random samples
        samples = point_df.sample(min_samples)
        x1, y1 = samples.iloc[0][['x', 'y']]
        x2, y2 = samples.iloc[1][['x', 'y']]
        y_MAD = np.median(np.abs(point_df['y'] - np.median(point_df['y'])))
        x_MAD = np.median(np.abs(point_df['x'] - np.median(point_df['x'])))

        # Calculate line coefficients (slope and intercept) or vertical line constant x
        if x1 == x2:
            continue
        else:
            vertical_line = False
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

        # Randomly choose between vertical and horizontal residual calculations
        residual_type = np.random.choice(['vertical', 'horizontal'])

        if residual_type == 'vertical':
            residuals = np.abs(point_df['y'] - (slope * point_df['x'] + intercept))
            residual_threshold = y_MAD*0.8
        else:  # 'horizontal'
            residuals = np.abs(point_df['x'] - (point_df['y'] - intercept) / slope)
            residual_threshold = x_MAD*0.8

        # Find inliers
        inliers = point_df[residuals <= residual_threshold]

        # Update best model if a better one is found
        inlier_count = len(inliers)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_model = {'slope': slope, 'intercept': intercept}
            best_residual_type = residual_type

    inlier_percent = best_inlier_count / len(point_df)
    if best_model is None or inlier_percent < 0.5:
        # If no model is found or the inlier percentage is too low, test a vertical line
        x1_mode = point_df['x'].mode()[0]
        vertial_inliers = point_df[(point_df['x'] >= x1_mode - x_MAD) & (point_df['x'] <= x1_mode + x_MAD)]
        if len(vertial_inliers) > best_inlier_count:
            best_model = {'x_const': x1_mode}
            vertical_line = True
            best_inliers = vertial_inliers

    best_model['is_vertical'] = vertical_line

    return best_model, best_inliers, best_residual_type, vertical_line

def line_plot(ax, model, is_vertical, x_min, x_max):
    '''
    Plot a line on the given axis.
    :param ax: axis
    :param model: model containing the line coefficients
    :param is_vertical: boolean for vertical line
    :param x_min: minimum x value
    :param x_max: maximum x value
    '''
    if is_vertical:
        ax.axvline(model['x_const'], c='r')
    else:
        x = np.linspace(x_min, x_max, 100)
        y = model['slope'] * x + model['intercept']
        ax.plot(x, y, c='r')

def find_intersection(model1, model2):
    '''
    Find the intersection point of two lines.
    :param model1: model containing the line coefficients of the first line
    :param model2: model containing the line coefficients of the second line
    :return: x, y coordinates of the intersection point
    '''
    if model1.get('x_const'):
        x = model1['x_const']
        y = model2['slope'] * x + model2['intercept']
    elif model2.get('x_const'):
        x = model2['x_const']
        y = model1['slope'] * x + model1['intercept']
    else:
        # check for parallel lines
        if model1['slope'] == model2['slope']:
            return None, None
        x = (model2['intercept'] - model1['intercept']) / (model1['slope'] - model2['slope'])
        y = model1['slope'] * x + model1['intercept']
    return x, y

def RANSAC_line(point_df):
    '''
    Find two lines using RANSAC. The first line is found using all points. The second line is found by removing the inliers of the first line.
    :param point_df: dataframe with columns 'x' and 'y' containing the coordinates of the points
    :return: model of the first line, model of the second line
    '''
    model, inliers, residual_type, vertical_line = ransac_line_fitting(point_df)
    inlier_mask = point_df.index.isin(inliers.index)
    outlier_mask = np.logical_not(inlier_mask)
    # Find second line while removing first inliers
    outlier_df = point_df.iloc[outlier_mask]
    model2, inliers2, residual_type2, is_vertical = ransac_line_fitting(outlier_df)
    return model, model2

def plot_with_annotate(ax, point_df, color='b'):
    '''
    Plot points and annotate them with their index.
    :param ax: axis
    :param point_df: dataframe with columns 'x' and 'y' containing the coordinates of the points
    :param color: color of the points
    '''
    point_df['index'] = range(len(point_df))
    ax.scatter(point_df['x'], point_df['y'], s=25, c=color)
    for i in range(len(point_df)):
        ax.annotate(i, (point_df.iloc[i]['x'], point_df.iloc[i]['y']))

def points_within_circle(point_df, center_x, center_y, radius):
    '''
    Find all points within a circle.
    :param point_df: dataframe with columns 'x' and 'y' containing the coordinates of the points
    :param center_x: x coordinate of the circle center
    :param center_y: y coordinate of the circle center
    :param radius: radius of the circle
    :return: dataframe with columns 'x' and 'y' containing the coordinates of the points within the circle
    '''
    point_df = point_df.copy()
    point_df['distance'] = np.sqrt((point_df['x'] - center_x)**2 + (point_df['y'] - center_y)**2)
    return point_df[point_df['distance'] <= radius]

def iterative_circle_centering(point_df, center_x, center_y, radius, iterations=10):
    '''
    Find the center of a circle by iteratively moving the center to the mean of the points within the circle. 
    The radius is reduced by 20% each iteration.
    :param point_df: dataframe with columns 'x' and 'y' containing the coordinates of the points
    :param center_x: x coordinate of the circle center
    :param center_y: y coordinate of the circle center
    :param radius: radius of the circle
    :param iterations: number of iterations
    :return: x coordinate of the circle center, y coordinate of the circle center, list of (x, y, radius) tuples for each iteration
    '''
    radius = radius
    centers = []
    for i in range(iterations):
        circle_points = points_within_circle(point_df, center_x, center_y, radius)
        circle_points_mean = circle_points.mean()
        if circle_points_mean['x'] == center_x or circle_points_mean['y'] == center_y:
            break
        if circle_points_mean['x'] == np.nan or circle_points_mean['y'] == np.nan:
            break
        center_x = circle_points_mean['x']
        center_y = circle_points_mean['y']
        centers.append((center_x, center_y, radius))
        radius = radius * 0.8
    if center_x == np.nan or center_y == np.nan:
        raise Warning(f'Center is NaN')
    return center_x, center_y, centers

def calculate_peak_angle(peaks, histogram):
    '''
    Calculate the angle of the peaks in the histogram.
    :param peaks: list of peak indices
    :param histogram: histogram
    :return: list of angles
    '''
    angles = [n for n in range(0, 360, 360//20)]
    peak_angles = []
    histogram = histogram[1:-1]
    for peak in peaks:
        hist_sum = np.sum([histogram[peak-1], histogram[peak], histogram[((peak+1)%20)]]) # wrap around for last bin
        peak_angle = angles[peak]
        left_add = (histogram[peak-1]/hist_sum) * 18
        right_add = (histogram[((peak+1)%20)]/hist_sum) * 18
        peak_angle = peak_angle - left_add + right_add
        peak_angle %= 360
        peak_angle = round(peak_angle, 2)
        peak_angles.append(peak_angle)
    return peak_angles

def count_peaks_simplified_angles(central_point, neighbors_points, n=20):
    '''
    Count the number of peaks in the histogram of the angles between the central point and its neighbors.
    Uses simplified find_angles which are x relative to the central point.
    :param central_point: (x, y) coordinates of the central point
    :param neighbors_points: (x, y) coordinates of the neighbors
    :param n: number of bins in the histogram
    :return: number of peaks, peaks
    '''
    histogram, edges = np.histogram(find_angle(central_point[0], central_point[1], neighbors_points)['angle'], bins=n, range=(0, 360))
    histogram = np.insert(histogram, 0, 0)
    histogram = np.append(histogram, 0)
    histogram = np.where(histogram < np.max(histogram)*0.15, 0, histogram)
    peaks = signal.find_peaks(histogram, distance=2, height=0.3*histogram.max())[0]
    # Uncomment to plot the histogram and the peaks
    # plt.plot(histogram)
    # plt.plot(peaks, histogram[peaks], 'x')
    # plt.show()
    return len(peaks), peaks