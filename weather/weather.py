#!/usr/bin/python
"""
Module for downloading and using ECMWF weather data.

There are two important classes:
    * WeatherApi: used for downloading GRIB files from MARS
    * WeatherExtractor: used for querying weather data from a pre-downloaded GRIB file

Example:
    Examples of class usages are given in class docstring.

Todo:
    * add interpolation capability to WeatherExtractor._aggregate_points
"""
import datetime
import json
import cPickle as pickle
from collections import defaultdict

import numpy as np
import pandas as pd

import ecmwfapi
import pygrib

from .request import Area, EcmwfServer, WeatherReq

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

    Warning:
        * after 2015-5-13 number of parameters increases from 11 to 15
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

            $ wa.get_actual(from_date, to_date, aggtime)

            Where:
                from_date, to_date: time window in days
                aggtime: aggregation of weather data on different levels:
                    aggtime='hour': aggregate by hour
                    aggtime='day': aggregation by day
                    aggtime='week': aggregation by week

        Queries about forecasted weather have the following format:

            $ wa.get_forecast(base_date, from_date, to_date, aggtime)

            Where:
                from_date, to_date: time window in days
                aggtime: aggregation of weather data on different levels:
                    aggtime='hour': aggregate by hour
                    aggtime='day': aggregation by day
                    aggtime='week': aggregation by week

    """

    def __init__(self):
        self.grib_msgs = None

    def load(self, filepaths):
        """
        Load weather data from grib file obtained via API request or from
        the pickled pandas.DataFrame.

        Arguments:
            filepaths (list): list of paths to files containing weather data

        Warning:
            after 2015-5-13 number of parameters increases from 11 to 15 and
            additional parameter 'ptype' which disturbs the indexing 
            (because of inconsistent 'validDateTime') sneaks in 
        """

        def _load_from_grib(filepath, append=False):
            """ Load measurements from GRIB file. """
            grbs = pygrib.open(filepath)

            # load grib messages
            grib_messages = []

            lats, lons = grbs.message(1).latlons()
            lats, lons = lats.flatten(), lons.flatten()

            grbs.rewind()
            for grib_msg in grbs:
                grib_messages.append({
                    'shortName': grib_msg.shortName,
                    'values': grib_msg.values.flatten(),
                    'validDateTime': grib_msg.validDate,
                    'validityDateTime': WeatherExtractor._str_to_datetime(
                        str(grib_msg.validityDate) + str(grib_msg.validityTime)),
                    'lats': lats,
                    'lons': lons,
                    'type': grib_msg.marsType  # forecast or actual
                })
            grbs.close()
            return pd.DataFrame.from_dict(grib_messages)

        def _load_from_pkl(filepath, append=False):
            """ Load pandas.DataFrame containing measurements. """
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        if not isinstance(filepaths, list):
            filepaths = [filepaths]  # wrap in list

        for filepath in filepaths:
            curr_msgs = None
            if filepath.endswith('.grib'):
                curr_msgs = _load_from_grib(filepath, append=True)
            elif filepath.endswith('.pkl'):
                curr_msgs = _load_from_pkl(filepath, append=True)
            else:
                raise Exception("File format not recognized")

            # append messages
            if self.grib_msgs == None:
                self.grib_msgs = curr_msgs
            else:
                self.grib_msgs = pd.concat([self.grib_msgs, curr_msgs])

            # remove 'ptype'
            self.grib_msgs = self.grib_msgs[self.grib_msgs['shortName'] != 'ptype']

            # index by base date (date when the forecast was made)
            self.grib_msgs.set_index('validDateTime', drop=False, inplace=True)
            self.grib_msgs.sort_index(inplace=True)

    def _latslons_from_dict(self, points):
        """ Get lattitudes and longtitudes from list of points. """
        assert isinstance(points, list)

        n_points = len(points)
        lats, lons = np.zeros(n_points), np.zeros(n_points)
        for i, point in enumerate(points):
            lats[i], lons[i] = point['lat'], point['lon']
        return (lats, lons)

    def _calc_closest(self, lats, lons, target_lats, target_lons):
        """
        For each point Pi = (lats[i], lons[i]) calculate the closest point Pj = (target_lats[j], target_lons[j])
        according to euclidean distance. In case of a tie take the first point with minimum distance.

        Args:
            lats, lons (np.array(dtype=float)): latitudes and longtitudes of original points
            target_lats, target_lons (np.array(dtype=float)): latitudes and longtitudes of target points

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

    def _interpolate_values(self, values, closest, num_original, num_targets, aggtype):
        """
        Do a value interpolation for given target points according to aggregation type.

        Args:
            values (np.array(dtype=float)): original values
            closest (np.array(dtype=int)):
            num_original (int):
            num_targets (int):
            aggtype (str):

        Returns:
            np.array(dtype=float): interpolated values for target points
        """
        # get interpolated values
        result_values = np.zeros(num_targets)
        if aggtype == 'one':
            for i in xrange(num_targets):
                result_values[i] = values[closest[i]]
        elif aggtype == 'mean':
            result_count = np.zeros(num_targets)
            for i in xrange(num_original):
                result_values[closest[i]] += values[i]
                result_count[closest[i]] += 1.
            result_count[result_count == 0] = 1.  # avoid dividing by zero
            result_values /= result_count
        return result_values

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

    def _aggregate_points(self, weather_result, aggloc, aggtype='one', interp_points=None):
        """
        Do an interpolation of measurement values for target points (given with target_lats and target_lons)
        from weather_result points.

        Args:
            weather_result (pandas.DataFrame): object containing original measurements and points
            aggloc (str): aggregation level
            aggtype (str): aggregation type, can be one of the following:

                'one' - keep only the value of a grid point which is closest to the target point
                'mean' - calculate the mean value of all grid points closest to the target point

                TODO:
                    'interpolate' - do a kind of ECMWF interpolation

        Returns:
            pandas.DataFrame: resulting object with interpolated points
        """
        assert aggloc in ['grid', 'points', 'country']
        assert aggtype in ['one', 'mean']
        if aggloc == 'points':
            assert interp_points is not None
        assert len(weather_result) > 0

        lats, lons = weather_result['lats'].iloc[0], weather_result['lons'].iloc[0]

        if aggloc == 'grid':  # no aggregation
            return weather_result
        elif aggloc == 'points':
            target_lats, target_lons = interp_points[0], interp_points[1]
        elif aggloc == 'country':  # center of slovenia
            target_lats, target_lons = np.array([46.1512]), np.array([14.9955])

        if aggtype == 'one':
            # each target point has only one closest grid point
            closest = self._calc_closest(target_lats, target_lons, lats, lons)
        elif aggtype == 'mean':
            # each grid point has only one closest target point
            closest = self._calc_closest(lats, lons, target_lats, target_lons)

        num_original = lats.shape[0]
        num_targets = target_lats.shape[0]

        # create new weather object
        tmp_result = list()

        columns = weather_result.columns
        for raw_row in weather_result.itertuples(index=False):
            row_data = dict()
            for col_pos, col_str in enumerate(columns):
                # affected columns are 'values', 'lats' and 'lons'
                if col_str == 'values':
                    row_data[col_str] = self._interpolate_values(
                        raw_row[col_pos], closest, num_original, num_targets, aggtype)
                elif col_str == 'lats':
                    row_data[col_str] = target_lats
                elif col_str == 'lons':
                    row_data[col_str] = target_lons
                else:
                    row_data[col_str] = raw_row[col_pos]
            tmp_result.append(row_data)

        return pd.DataFrame.from_dict(tmp_result)

    def _aggregate_values(self, weather_result, aggtime):
        """
        Aggregate weather values on hourly, daily or weekly level. Calculate the mean
        value for each measurement point over given time period.

        Serves more as an aggregation example. For more complex aggregations set aggtime='hour'
        and implement own aggregation policy on pandas.DataFrame.

        Args:
            weather_result (pandas.DataFrame): object containing original measurements
            aggtime (str): aggregation level which can be 'hour', 'day' or 'week'

        Returns:
            pandas.DataFrame: resulting object with aggregated values
        """
        assert aggtime in ['hour', 'day', 'week', 'H', 'D', 'W']
        aggtime = {'hour': 'H', 'day': 'D', 'week': 'W'}[aggtime]

        if aggtime == 'H':
            return weather_result

        weather_result.set_index(
            ['validDateTime', 'validityDateTime', 'shortName'], drop=True, inplace=True)

        groups = weather_result.groupby([pd.Grouper(freq='D', level='validDateTime'), pd.Grouper(
            freq=aggtime, level='validityDateTime'), pd.Grouper(level='shortName')])

        tmp_result = groups.apply(
            lambda group:
            pd.Series(
                {
                    'values': group['values'].mean(),
                    'lats': group['lats'].iloc[0],
                    'lons': group['lons'].iloc[0]
                })
        )
        tmp_result.reset_index(drop=False, inplace=True)

        return tmp_result

    def get_actual(self, from_date, to_date, aggtime='hour', aggloc='grid', interp_points=None):
        """
        Get the actual weather for each day from a given time window.

        Args:
            from_date (datetime.date): start of the timewindow
            to_date (datetime.date): end of the timewindow
            aggtime (str): time aggregation level; can be 'hour', 'day' or 'week'
            aggloc (str): location aggregation level; can be 'country', 'points' or 'grid'
            interp_points (list of dicts): list of interpolation points with each point represented
                as dict with fields 'lon' and 'lat' representing longtitude and lattitude if aggloc='points'
        Returns:
            pandas.DataFrame: resulting object with weather measurements
        """
        assert type(from_date) == datetime.date
        assert type(to_date) == datetime.date
        assert from_date <= to_date
        assert aggtime in ['hour', 'day', 'week']
        assert aggloc in ['country', 'points', 'grid']

        if aggloc == 'points':
            if interp_points is None:
                raise ValueError(
                    "interp_points cannot be None if aggloc is set to 'points'.")
            interp_points = self._latslons_from_dict(interp_points)

        req_period = self.grib_msgs.loc[from_date:to_date]
        tmp_result = req_period[req_period['validDateTime'].dt.date ==
                                req_period['validityDateTime'].dt.date]
        actual_data = tmp_result[tmp_result['type'] == 'an']

        # get data about actual weather
        if len(actual_data) == 0:
            raise RuntimeWarning("No data about actual weather available!")
        else:
            tmp_result = actual_data

        # drop 'type' column
        tmp_result.drop('type', axis=1, inplace=True)

        # reset original index
        tmp_result.reset_index(drop=True, inplace=True)

        # point aggregation
        tmp_result = self._aggregate_points(
            tmp_result, aggloc, interp_points=interp_points)

        # time aggregation
        tmp_result = self._aggregate_values(tmp_result, aggtime)

        return tmp_result

    def get_forecast(self, base_date, from_date, to_date, aggtime='hour', aggloc='grid', interp_points=None):
        """
        Get the weather forecast for a given time window from a given date.

        Args:
            base_date (datetime.date): base date for the forecast
            from_date (datetime.date): start of the time window
            end_date (datetime.date): end of the timewindow
            aggtime (str): time aggregation level; can be 'hour', 'day' or 'week'
            aggloc (str): location aggregation level; can be 'country', 'points' or 'grid'

        Returns:
            pandas.DataFrame: resulting object with weather measurements
        """
        assert type(base_date) == datetime.date
        assert type(from_date) == datetime.date
        assert type(to_date) == datetime.date
        assert base_date <= from_date <= to_date
        assert aggtime in ['hour', 'day', 'week']
        assert aggloc in ['country', 'points', 'grid']

        if aggloc == 'points':
            if interp_points is None:
                raise ValueError(
                    "interp_points cannot be None if aggloc is set to 'points'.")
            interp_points = self._latslons_from_dict(interp_points)

        req_period = self.grib_msgs.loc[base_date]

        # start with default (hourly) aggregation
        tmp_result = req_period[req_period['validityDateTime'].dt.date >= from_date]
        tmp_result = tmp_result[tmp_result['validityDateTime'].dt.date <= to_date]

        # keep only forecast data
        tmp_result = tmp_result[tmp_result['type'] == 'fc']

        # drop 'type' column
        tmp_result.drop('type', axis=1, inplace=True)

        # reset original index
        tmp_result.reset_index(drop=True, inplace=True)

        # point aggregation
        tmp_result = self._aggregate_points(
            tmp_result, aggloc, interp_points=interp_points)

        # time aggregation
        tmp_result = self._aggregate_values(tmp_result, aggtime)

        return tmp_result


class WeatherApi:
    """
    Interface for downloading weather data from MARS.

    Example:
        $ wa = WeatherApi()
    """

    def __init__(self):
        self.server = EcmwfServer()

    def get(self, from_date, to_date, target, request_type='forecast', base_time='midnight',
            steps=None, area='slovenia', grid=(0.25, 0.25)):
        """
        Execute a MARS request with given parameters and store the result to file 'target.grib'.

        Args:
            base_time (str): 'midnight' or 'noon'
        """
        assert isinstance(from_date, datetime.date)
        assert isinstance(to_date, datetime.date)
        assert from_date <= to_date

        assert base_time in ['midnight', 'noon']
        assert request_type in ['forecast', 'actual']

        # create new mars request
        req = WeatherReq()

        # set date
        req.set_date(from_date, end_date=to_date)

        # set target grib file
        req.set_target(target)

        # set base time
        if base_time == 'midnight':
            req.set_midnight()
        else:
            req.set_noon()

        if request_type == 'actual':  # get actual weather
            req.set_type('an')
            req.set_analysis_cycle()  # overrides the time
        elif request_type == 'forecast':  # get weather forecast
            req.set_type('fc')
            if steps is None:
                    # assume base time is 'midnight'
                    # base_date is the date the forecast was made
                steps = []

                # current day + next three days
                for day_off in range(4):
                    steps += [day_off * 24 +
                              hour_off for hour_off in [0, 6, 9, 12, 15, 18, 21]]

                # other 4 days
                for day_off in range(4, 8):
                    steps += [day_off * 24 +
                              hour_off for hour_off in [0, 6, 12, 18]]

                if base_time == 'noon':
                    steps = [step for step in steps in step - 12 >= 0]

            req.set_step(steps)

        # set area
        if area == 'slovenia':
            area = Area.Slovenia
        req.set_area(area)

        # set grid resolution
        req.set_grid(grid)

        self.server.retrieve(req)
