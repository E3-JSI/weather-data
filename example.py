from datetime import date

from weather.weather import WeatherExtractor, WeatherApi


def download_data():
    """ Download example data, use default parameters. """
    wa = WeatherApi()
    wa.get(from_date=date(2015, 1, 1), to_date=date(
        2015, 2, 28), target='jan2015-feb2015.grib')


# download data from MARS - might take some time
download_data()

# query the downloaded data
we = WeatherExtractor()
we.load('weather/data/jan2015-feb2015.grib')

# get mean forecasted weather on 2015-1-10, 2015-1-11 and 2015-1-12 for whole country
weather_result = we.get_forecast(from_date=date(
    2015, 1, 10), to_date=date(2015, 1, 12), aggtime='day', aggloc='country')

assert len(weather_result) > 0

# print results
for row in weather_result.iterrows():
    # row is tuple (index, columns)
    measure = row[1]

    print "Measurement of %s at from %s for %s" % (measure['shortName'], measure['validDateTime'], measure['validityDateTime'])
    for lat, lon, val in zip(measure['lats'], measure['lons'], measure['values']):
        print "%f N %f S = %f" % (lat, lon, val)
