from __future__ import print_function

from datetime import date
from weather.weather import WeatherExtractor, WeatherApi


def print_data(weather_data):
    for row in weather_data.iterrows():
        # row is tuple (index, columns)
        measure = row[1]

        print("Measurement of %s at from %s for %s" % (measure['shortName'], measure['validDateTime'], measure['validityDateTime']))
        for lat, lon, val in zip(measure['lats'], measure['lons'], measure['values']):
            print("%f N %f S = %f" % (lat, lon, val))


wa = WeatherApi()

# download data for november 2017
wa.get(from_date=date(2017, 11, 1), to_date=date(2017, 11, 30),
       target='nov2017.grib')

# query the downloaded data
we = WeatherExtractor()

# load actual and forecasted weather data
we.load(['nov2017.grib'])

""" Get forecasted data from 1-11-2017 for 2-11-2017, 3-11-2017 and 4-11-2017 for all grid points. """
weather_data = we.get_forecast(base_date=date(2017, 11, 1), from_date=date(
    2017, 11, 2), to_date=date(2017, 11, 4), aggtime='hour', aggloc='grid')

# print the result
print_data(weather_data)

""" Get forecasted data from 1-11-2017 for 2-11-2017, 3-11-2017 and 4-11-2017 for
two specific points with latitudes and longitudes: (45.01, 13.00) and (46.00, 12.05) """
points = [{'lat': 45.01, 'lon': 13.0}, {'lat': 46.0, 'lon': 12.05}]
weather_data = we.get_forecast(base_date=date(2017, 11, 1), from_date=date(
    2017, 11, 2), to_date=date(2017, 11, 4), aggtime='hour', aggloc='points', interp_points=points)
# print the result
print_data(weather_data)

""" Get actual weather data for 2-11-2017, 3-11-2017 and 4-11-2017 for
two specific points with latitudes and longitudes: (45.01, 13.00) and (46.00, 12.05) """
points = [{'lat': 45.01, 'lon': 13.0}, {'lat': 46.0, 'lon': 12.05}]
weather_data = we.get_actual(from_date=date(
    2017, 11, 2), to_date=date(2017, 11, 4), aggtime='hour', aggloc='points', interp_points=points)

# print the result
print_data(weather_data)

""" Get actual weather data for 2-11-2017, 3-11-2017 and 4-11-2017 for
bounding box defined by its corner points with latitudes and longitudes: (45.45, 13.70) and (46.85, 16.56) """
bounding_box = [[45.45, 13.70], [46.85, 16.56]]
weather_data = we.get_actual(
    from_date=date(2018, 6, 1), to_date=date(2018, 6, 7), aggtime='hour', aggloc='bbox',
    bounding_box=bounding_box)

# print the result
print_data(weather_data)

# store weather data in pickle format for faster loading in the future
we.store('nov2017.pkl')
