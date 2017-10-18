#!/usr/bin/python
"""
    ECMWF MARS archive access code.
"""
from collections import defaultdict
import datetime

# from datetime import datetime, timedelta, date, time
import numpy as np
import json

import ecmwfapi
import pygrib

from .request import EcmwfServer, WeatherReq, Area

"""
    Current weather parameters:

    (u'Cloud base height', u'cbh'),
    (u'Maximum temperature at 2 metres in the last 6 hours', u'mx2t6'),
    (u'Minimum temperature at 2 metres in the last 6 hours', u'mn2t6'),
    (u'10 metre wind gust in the last 6 hours', u'10fg6'),
    (u'Surface pressure', u'sp'),
    (u'Total column water vapour', u'tcwv'),
    (u'Snow depth', u'sd'),
    (u'Snowfall', u'sf'),
    (u'Total cloud cover', u'tcc'),
    (u'2 metre temperature', u'2t'),
    (u'Total precipitation', u'tp')
"""


class WeatherExtractor:
    """
        Interface for extracting weather data from pre-downloaded GRIB file.
        Each GRIB file is a collection of self-contained grib messages the following
        important fileds:
            - lat/lon: lattitude and longtitude of weather-points
            - values: 
            ...
        EXAMPLE:
        > we = WeatherExtractor()
        > we.load('example_data.grib') 

        > we.get_actual(fromDate=date(2017,3,12), timeOfDay=[0, 12])
        > we.get_actual(fromDate=date(2017,1,1), toDate=date(2017,6,30), groupby=['hour'/'day'/'week'/'month'])

        > we.get_forecast(fromDate=date(2017,3,12), nDays=5, timeOfDay=[6,9,12,15,18], groupby=['hour'/'day'/'week'])
    """

    def __init__(self):
        """
            Use weather state in specific points.
        """
        self.interp_points = self._read_points('arso_weather_stations')

        # forecast index
        self.index_valid_date = defaultdict(lambda: [])

    def load(self, filepath):
        """
            Load GRIB messages from file.
        """
        grbs = pygrib.open(filepath)

        # load GRIB messages in memory - may take some time
        self.index_valid_date = defaultdict(lambda: [])

        # index by date
        for grib_msg in grbs:
            self.index_valid_date[grib_msg.validDate.date()].append(grib_msg)

        grbs.close()

    def _read_points(self, points_file):
        """
            Read interpolation points / regions from file.
        """
        if points_file == 'arso_weather_stations':
            weather_stations = []
            with open('./data/arso_weather_stations.json', 'r') as f:
                weather_stations = json.loads(f.read())

            lats, lons = np.zeros(len(weather_stations)), np.zeros(
                len(weather_stations))
            for i, weather_station in enumerate(weather_stations):
                lats[i], lons[i] = weather_station['lat'], weather_station['lon']

            return (lats, lons)
        else:
            raise ValueError('Unrecognized points file %s' % repr(points_file))

    def _calc_closest(self, lats, lons, target_lats, target_lons):
        """
            For each point Pi = (lats[i], lons[i]) calculate the point Pj = (target_lats[j], target_lons[j])
            with minimum euclidean distance. In case of a tie take the first point with minimum distance.
        """
        num_points = lats.shape[0]
        num_target = target_lats.shape[0]

        closest = np.zeros(num_points, dtype=np.int)
        for i in xrange(num_points):
            best_dist = (lats[i] - target_lats[0])**2 + \
                (lons[i] - target_lons[0])**2
            for j in xrange(1, num_target):
                curr_dist = (lats[i] - target_lats[j])**2 + \
                    (lons[i] - target_lons[j])**2
                if curr_dist < best_dist:
                    best_dist = curr_dist
                    closest[i] = j
        return closest

    @staticmethod
    def _str_to_datetime(val):
        """ Convert datetime string 'YYYYMMDDHHMM' to datetime object. """
        tmp_date = datetime.date(int(val[:4]), int(val[4:6]), int(val[6:8]))

        time_str = val[8:]
        assert len(time_str) in [1, 3, 4]
        if len(time_str) == 1:
            # midnight
            return datetime.datetime.combine(tmp_date, datetime.time(int(time_str)))
        elif len(time_str) == 3:
            # HMM
            return datetime.datetime.combine(tmp_date, datetime.time(int(time_str[:1]), int(time_str[1:])))
        elif len(time_str) == 4:
            # HHMM
            return datetime.datetime.combine(tmp_date, datetime.time(int(time_str[:2]), int(time_str[2:])))

    class WeatherResult:
        """ 
            Class wrapping the results of a data query.

            EXAMPLE:
                > wr = we.get_forecast(...)
                > for ws in wr:
                    ws.date_time
                    print ws[Params.Temperature]
                > wr[datetime(2015,3,12,18,0)][param.Temperature]

                > lats, lons = wr.get_latslons()
                > for week in wr:
                    temps = week.get(WeatherParams.Temperature)
                    for lat, lon, temp in zip(lats, lons, temps):
                        print "(%f, %f) = %f" % (lat, lon, temp)

                    humids = week.get(WeatherParams.Humidity)
                    for lat, lon, humid in zip(lats, lons, humids):
                        print "(%f, %f) = %f" % (lat, lon, humid)

        """

        def __init__(self, daterange, aggby):
            """
                timerange: datetime range query contains data about

            """
            self.aggby = aggby
            self.daterange = daterange

            self.lats, self.lons = None, None

            self.weather_datetimeranges = []
            self.weather_values = []
            self.weather_params = []

        def _set_latlons(self, lats, lons):
            self.lats = np.copy(lats)
            self.lons = np.copy(lons)

        def get_latslons(self):
            """ Get lattitudes and longtitudes of weather measurement points """
            return self.lats, self.lons

        def _n_values(self):
            return len(self.weather_values)

        def __iter__(self):
            for i in xrange(self._n_values()):
                yield self.weather_datetimeranges[i], self.weather_params[i], self.weather_values[i]

        def __getitem__(self, key):
            return self.weather_datetimeranges[key], self.weather_params[key], self.weather_values[key]

        def __len__(self):
            return self._n_values()

        def _append_msg(self, grib_msg):
            """ 
                Append new GRIB message. Assume all messages have the same lattitude and longtitude.
            """
            if self.lats is None:
                self.lats = grib_msg.latlons()[0].flatten()
                self.lons = grib_msg.latlons()[1].flatten()

            self.weather_values.append(grib_msg.values.flatten())

            validity_datetime = WeatherExtractor._str_to_datetime(
                str(grib_msg.validityDate) + str(grib_msg.validityTime))
            self.weather_datetimeranges.append(
                (validity_datetime, validity_datetime))

            self.weather_params.append(grib_msg.shortName)

        def _append(self, datetimerange, params, values):
            self.weather_datetimeranges.append(datetimerange)
            self.weather_params.append(params)
            self.weather_values.append(values)

    def _aggregate_points(self, weather_result, target_lats, target_lons, aggr='one'):
        """
            Do a kind of point aggregation if the target points are different>
            then the retrieved grid points. Aggregation can be done based on different
            criteria:

                'one' - keep only the value of a grid point which is closest to the target point
                'mean' - calculate the mean value of all grid points closest to the target point

                TODO:
                    'interpolate' - do a kind of ECMWF interpolation
        """
        assert aggr in ['one', 'mean']

        lats, lons = weather_result.get_latslons()

        if aggr == 'one':
            # each target point has only one closest grid point
            closest = self._calc_closest(target_lats, target_lons, lats, lons)
        elif aggr == 'mean':
            # each grid point has only one closest target point
            closest = self._calc_closest(lats, lons, target_lats, target_lons)

        num_original = lats.shape[0]
        num_targets = target_lats.shape[0]

        # create new weather object
        result = self.WeatherResult(
            weather_result.daterange, weather_result.aggby)
        result._set_latlons(target_lats, target_lons)

        for datetimerange, param, values in weather_result:
            # get interpolated values
            result_values = np.zeros(num_targets)
            if aggr == 'one':
                for i in xrange(num_targets):
                    result_values[i] = values[closest[i]]
            elif aggr == 'mean':
                result_count = np.zeros(num_targets)
                for i in xrange(num_original):
                    result_values[closest[i]] += values[i]
                    result_count[closest[i]] += 1
                result_count[result_count == 0] = 1.  # avoid dividing by zero
                result_values /= result_count

            result._append(datetimerange, param, result_values)
        return result

    def _aggregate_values(self, weather_result, aggby):
        """
            Aggregate weather values on hour, day or week level. Calculate the mean 
            parameter value for each point over time.
        """
        assert weather_result.aggby == 'hour'
        assert aggby in ['hour', 'day']

        result = self.WeatherResult(weather_result.daterange, aggby)
        result._set_latlons(*weather_result.get_latslons())

        if aggby == 'hour':
            return weather_result
        elif aggby == 'day':
            curr_pos = 0
            while curr_pos < weather_result._n_values():
                # datetime of a current group
                group_datetimerange, _, _ = weather_result[curr_pos]
                group_date = group_datetimerange[0].date()

                end_date = group_date

                agg_values = defaultdict(lambda: [])
                while curr_pos < weather_result._n_values():
                    curr_datetimerange, curr_param, curr_values = weather_result[curr_pos]
                    curr_date = curr_datetimerange[0].date()

                    # messages are sorted
                    if curr_date > group_date:
                        break

                    agg_values[curr_param].append(curr_values)
                    end_date = max(end_date, curr_date)
                    curr_pos += 1

                # aggregate values
                for param, values_list in agg_values.iteritems():
                    # calculate mean value
                    result._append((datetime.datetime.combine(group_date, datetime.time(0)), datetime.datetime.combine(
                        end_date, datetime.time(23, 59))), param, np.array(values_list).mean(axis=0))
        return result

    def get_actual(self, fromDate, toDate, aggby='hour'):
        """ 
            Get an actual weather. Do a daily, weekly or monthly aggregation. 
        """
        assert isinstance(fromDate, datetime.date)
        assert isinstance(toDate, datetime.date)
        assert aggby in ['hour', 'day', 'week', 'month']
        assert fromDate <= toDate

        # start with a hour aggregation
        tmp_result = self.WeatherResult(
            daterange=(fromDate, toDate), aggby='hour')

        curr_date = fromDate
        end_date = toDate

        while curr_date <= end_date:
            for grib_msg in self.index_valid_date[curr_date]:

                forecast_datetime = WeatherExtractor._str_to_datetime(
                    str(grib_msg.validityDate) + str(grib_msg.validityTime))

                if curr_date == forecast_datetime.date():  # forecast for current date
                    tmp_result._append_msg(grib_msg)
                else:
                    break  # assume messages are sorted by datetime
            curr_date += datetime.timedelta(days=1)

        # are there any results?
        assert tmp_result._n_values() > 0

        # point aggregation
        tmp_result = self._aggregate_points(
            tmp_result, self.interp_points[0], self.interp_points[1])

        # time aggregation
        tmp_result = self._aggregate_values(tmp_result, aggby)
        return tmp_result

    def get_forecast(self, fromDate, nDays, aggby='hour', timeOfDay='default'):
        """
            Get weather forecast from a given date.
        """
        assert isinstance(fromDate, datetime.date)

        # beginning date and time
        start_datetime = datetime.datetime.combine(
            fromDate, datetime.time(0, 0))
        end_datetime = datetime.datetime.combine(
            start_datetime.date() + datetime.timedelta(days=nDays), datetime.time(23, 59))

        # start with a hour aggregation
        tmp_result = self.WeatherResult(
            daterange=(start_datetime.date(), end_datetime.date()), aggby='hour')
    
        for grib_msg in self.index_valid_date[fromDate]:
            forecast_datetime = WeatherExtractor._str_to_datetime(
                    str(grib_msg.validityDate) + str(grib_msg.validityTime))

            if forecast_datetime < end_datetime:
                tmp_result._append_msg(grib_msg)
            else:
                break  # assume messages are sorted by datetime
            
        # are there any results?
        assert tmp_result._n_values() > 0

        # point aggregation
        tmp_result = self._aggregate_points(
            tmp_result, self.interp_points[0], self.interp_points[1])

        # time aggregation
        tmp_result = self._aggregate_values(tmp_result, aggby)
        return tmp_result

class WeatherApi:
    """
        EXAMPLE:
            > wa = WeatherApi()
            > wa.get(dateFrom=date(2017,3,12), dateTo=date(2017,4,12), startTime='noon', steps=[12,24,...], area='slovenia')
    """

    def __init__(self):
        self.server = EcmwfServer()

    def get(self, dateFrom, dateTo, steps, time, area, target):
        """
            Execute a mars request with given parameters and store the result
            in GRIB format to file 'target'.
        """
        assert time in ['midnight', 'noon']

        # create new mars request
        req = WeatherReq()
        req.set_date(dateFrom, end_date=dateTo)

        if time == 'noon':
            req.set_noon()
        else:
            req.set_midnight()

        if area == 'slovenia':
            area = Area.Slovenia
        req.set_area(area)

        req.set_step(steps)
        req.set_grid((0.25, 0.25))
        req.set_target(target)

        self.server.retrieve(req)
        # return pygrib.open(target)
