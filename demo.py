import weather
import datetime

# server instance
server = weather.EcmwfServer()

# request
req = weather.WeatherReq()

req.set_noon()
req.set_step([24, 2*24, 3*24, 4*24, 5*24, 6*24, 7*24])
req.set_date(datetime.date(2015,1,1), datetime.date(2015,6,30))
req.set_target('jan2015-june2015-weekly-forecast.grib')
#req.set_area([46.53, 13.23, 45.25, 16.36]) # extreme points of slovenia
req.set_area([46.50, 13.25, 45.25, 16.25]) # extreme points of slovenia
req.set_grid((0.25,0.25)) 

# execute request
server.retrieve(req)
