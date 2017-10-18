#!/usr/bin/python
"""
Module for downloading and using ECMWF weather data.

There are two important classes:
    * WeatherApi: used for downloading GRIB files from MARS
    * WeatherExtractor: used for querying weather data from a pre-downloaded GRIB file

Example:
    Examples of class usages are given in class docstring.

Todo:
    * read interpolation points from file
"""
import datetime
import json
from collections import defaultdict
import numpy as np

import ecmwfapi
import pygrib

from .request import Area, EcmwfServer, WeatherReq

"""
List of weather measuring station of Slovenian environment agency.
"""
arso_weather_stations = [
    {"lat": 45.8958, "alt": 55.0, "lon": 13.6289,
        "shortTitle": "BILJE", "title": "NOVA GORICA"},
    {"lat": 46.2447, "alt": 244.0, "lon": 15.2525,
        "shortTitle": "CELJE", "title": "CELJE"},
    {"lat": 45.5603, "alt": 157.0, "lon": 15.1508,
        "shortTitle": "\u010cRNOMELJ - DOBLI\u010cE", "title": "CRNOMELJ"},
    {"lat": 46.3794, "alt": 2514.0, "lon": 13.8539,
        "shortTitle": "KREDARICA", "title": "KREDARICA"},
    {"lat": 45.8936, "alt": 154.0, "lon": 15.525,
        "shortTitle": "CERKLJE - LETALI\u0160\u010cE", "title": "CERKLJE - LETALISCE"},
    {"lat": 46.48, "alt": 264.0, "lon": 15.6869,
        "shortTitle": "MARIBOR - LETALI\u0160\u010cE", "title": "MARIBOR/SLIVNICA"},
    {"lat": 46.2178, "alt": 364.0, "lon": 14.4775,
        "shortTitle": "BRNIK - LETALI\u0160\u010cE", "title": "LJUBLJANA/BRNIK"},
    {"lat": 46.37, "alt": 515.0, "lon": 14.18,
        "shortTitle": "LESCE", "title": "LESCE"},
    {"lat": 45.4756, "alt": 2.0, "lon": 13.6206,
        "shortTitle": "PORTORO\u017d - LETALI\u0160\u010cE", "title": "PORTOROZ/SECOVLJE"},
    {"lat": 46.0681, "alt": 943.0, "lon": 15.2897,
        "shortTitle": "LISCA", "title": "LISCA"},
    {"lat": 46.0658, "alt": 299.0, "lon": 14.5172,
        "shortTitle": "LJUBLJANA - BE\u017dIGRAD", "title": "LJUBLJANA/BEZIGRAD"},
    {"lat": 46.6525, "alt": 188.0, "lon": 16.1961,
        "shortTitle": "MURSKA SOBOTA - RAKI\u010cAN", "title": "MURSKA SOBOTA"},
    {"lat": 45.8019, "alt": 220.0, "lon": 15.1822,
        "shortTitle": "NOVO MESTO", "title": "NOVO MESTO"},
    {"lat": 45.7664, "alt": 533.0, "lon": 14.1975,
        "shortTitle": "POSTOJNA", "title": "POSTOJNA"},
    {"lat": 46.4975, "alt": 864.0, "lon": 13.7175,
        "shortTitle": "RATE\u010cE - PLANICA", "title": "RATECE"},
    {"lat": 46.49, "alt": 455.0, "lon": 15.1161,
        "shortTitle": "\u0160MARTNO PRI SLOVENJ GRADCU", "title": "SLOVENJ GRADEC"}
]

"""
    List of returned weather measurements:
        Name:                                               Short Name:
        Cloud base height                                   cbh
        Maximum temperature at 2 metres in the last 6 hours mx2t6
        Minimum temperature at 2 metres in the last 6 hours mn2t6
        10 metre wind gust in the last 6 hours              10fg6
        Surface pressure                                    sp
        Total column water vapour                           tcwv
        Snow depth                                          sd
        Snowfall                                            sf
        Total cloud cover                                   tcc
        2 metre temperature                                 2t
        Total precipitation                                 tp
"""


class WeatherExtractor:
    """
    Interface for extracting weather data from pre-downloaded GRIB file. 
    Each GRIB file is a collection of self contained weather messages.

    It supports actual weather queries ( via .get_actual(...) ) and forecasted weather
    queries ( via .get_forecast(...) )

    Examples
        $ we = WeatherExtractor()
        $ we.load('example_data.grib')  

        Queries about actual weather have the following format:

            $ wa.get_actual(from_date, to_date, aggby)

            Where:
                from_date, to_date: time window in days
                aggby: aggregation of weather data on different levels:
                    aggby='hour': aggregate by hour
                    aggby='day': aggregation by day
                    aggby='week': aggregation by week

        Queries about forecasted weather have the following format:

            $ wa.get_forecast(from_date, n_days, aggby)

            Where:
                from_date, to_date: time window in days
                aggby: aggregation of weather data on different levels:
                    aggby='hour': aggregate by hour
                    aggby='day': aggregation by day
                    aggby='week': aggregation by week

        > we.get_forecast(fromDate=date(2017,3,12), nDays=5, timeOfDay=[6,9,12,15,18], groupby=['hour'/'day'/'week'])
    """

    def __init__(self):
        self.interp_points = self._read_points('arso_weather_stations')

        # forecast index
        self.index_valid_date = defaultdict(lambda: [])

    def load(self, filepath):
        """ Load GRIB file from given filepath. """
        grbs = pygrib.open(filepath)

        # load GRIB messages in memory - may take some time
        self.index_valid_date = defaultdict(lambda: [])

        # index by date
        for grib_msg in grbs:
            self.index_valid_date[grib_msg.validDate.date()].append(grib_msg)

        grbs.close()

    def _read_points(self, points_file):
        """ Read interpolation points or regions from file. """
        if points_file == 'arso_weather_stations':
            #weather_stations = []
            #with open('./data/arso_weather_stations.json', 'r') as f:
            #    weather_stations = json.loads(f.read())

            lats, lons = np.zeros(len(arso_weather_stations)), np.zeros(
                len(arso_weather_stations))
            for i, weather_station in enumerate(arso_weather_stations):
                lats[i], lons[i] = weather_station['lat'], weather_station['lon']
            return (lats, lons)
        else:
            raise ValueError('Unrecognized points file %s' % repr(points_file))

    def _calc_closest(self, lats, lons, target_lats, target_lons):
        """
        For each point Pi = (lats[i], lons[i]) calculate the closest point Pj = (target_lats[j], target_lons[j])
        according to euclidean distance. In case of a tie take the first point with minimum distance.

        Args:
            lats, lons, target_lats, target_lons (np.array(dtype=float)): The first parameter.

        Returns:
            np.array(dtype=int): array where value at index i represents the index of closest point j
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
            # midnight - only one number
            return datetime.datetime.combine(tmp_date, datetime.time(int(time_str)))
        elif len(time_str) == 3:
            # hmm format
            return datetime.datetime.combine(tmp_date, datetime.time(int(time_str[:1]), int(time_str[1:])))
        elif len(time_str) == 4:
            # hhmm format
            return datetime.datetime.combine(tmp_date, datetime.time(int(time_str[:2]), int(time_str[2:])))

    class WeatherResult:
        """
        Wrapper for query result returned by WeatherExtractor's .get_* functions. 

        Examples:
            $ wr = we.get_actual(from_date=date(2015,1,1), to_date=date(2015,1,31))
            $ for datetime_range, weather_param, values in wr:
                print datetime_range, weather_param, values

            Where:
                datetime_range: 
                weather_param: shortname of weather measurement
                values: 
        """

        def __init__(self, daterange, aggby):
            """
            Args:
                daterange ( tuple(datetime.date, datetime.date) ): time window of contained measurements
                aggby (str): measurement aggregation level (i.e. 'hour', 'day', 'week', ...)
            """
            self.aggby = aggby
            self.daterange = daterange

            self.lats, self.lons = None, None

            self.weather_datetimeranges = []
            self.weather_values = []
            self.weather_params = []

        def __iter__(self):
            for i in xrange(self._n_values()):
                yield self.weather_datetimeranges[i], self.weather_params[i], self.weather_values[i]

        def __getitem__(self, key):
            return self.weather_datetimeranges[key], self.weather_params[key], self.weather_values[key]

        def __len__(self):
            return self._n_values()

        def _set_latlons(self, lats, lons):
            self.lats = np.copy(lats)
            self.lons = np.copy(lons)

        def _n_values(self):
            return len(self.weather_values)

        def get_latslons(self):
            """ Get lattitudes and longtitudes of weather measurement points """
            return self.lats, self.lons

        def _append_msg(self, grib_msg):
            """ Append new GRIB message. Assume all messages have the same lattitude and longtitude. """
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
        Do an interpolation of measurement values for target points (given with target_lats and target_lons)
        from weather_result points.

        Args:

            weather_result (:obj:WeatherResult): 
            target_lats, target_lons (np.array(dtype=float)): lattitudes and longtitudes of target points
            aggr (str): aggregation level, which can be one of the following:

                'one' - keep only the value of a grid point which is closest to the target point
                'mean' - calculate the mean value of all grid points closest to the target point

                TODO:
                    'interpolate' - do a kind of ECMWF interpolation

        Returns:
            WeatherResult: resulting object with interpolated points
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
        Aggregate weather values on hourly, daily or weekly level. Calculate the mean 
        value for each measurement point over given time period.

        Args:
            weather_result (:obj:WeatherResult):
            aggrby (str): aggregation level, which can be 'hour' or 'day'
                
        TODO:
            * add weekly and monthly aggregation

        Returns:
            WeatherResult: resulting object with aggregated values
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
                    if curr_date != group_date:
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
    Interface for downloading weather data from MARS. 

    Examples
        $ wa = WeatherApi()
        $ wa.get(dateFrom=date(2017,3,12), dateTo=date(2017,4,12), startTime='noon', steps=[12,24,...], area='slovenia')
    """
    def __init__(self):
        self.server = EcmwfServer()

    def get(self, dateFrom, dateTo, steps, time, area, target):
        """
        Execute a MARS request with given parameters and store the result to file 'target.grib'.

        Args:
            weather_result (:obj:WeatherResult):
            aggrby (str): aggregation level, which can be 'hour' or 'day'
                
        TODO:
            * add weekly and monthly aggregation

        Returns:
            WeatherResult: resulting object with aggregated values
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
