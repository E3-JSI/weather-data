from datetime import date

from weather.weather import WeatherExtractor, WeatherApi

# download data - use default parameters
def download_data():
    wa = WeatherApi()
    wa.get(from_date=date(2015, 1, 1), to_date=date(
        2015, 2, 28), target='jan2015-feb2015.grib')

# download data from MARS - might take some time
download_data()
    
# query the downloaded data
we = WeatherExtractor()
we.load('jan2015-feb2015.grib')

# get actual weather on 2015-1-10, 2015-1-11 and 2015-1-12 by regions
weather_result = we.get_actual(from_date=date(
    2015, 1, 10), to_date=date(2015, 1, 12), aggtime='day', aggloc='region')

lats, lons = weather_result.get_latslons()

# print results
for measure_datetimerange, measure_param, measure_values in weather_result:
    print "Measurement of %s at (%s, %s): " % (str(measure_param), str(measure_datetimerange[0]), str(measure_datetimerange[1]))
    for lat, lon, val in zip(lats, lons, measure_values):
        print "%f N %f S = %f" % (lat, lon, val)
