from __future__ import print_function

import ecmwfapi
from datetime import date
class EcmwfServer():
    """
        Connection to the ECMWF server.
        Assumes a valid .ecmwfapirc file is present in the same folder as the code.
    """

    def __init__(self):
        """
            Set up a connection to the MARS data service.
        """
        self.service = ecmwfapi.ECMWFService('mars')

    def _check_target(self, target):
        """Check if target file is ok."""
        # try to create the file in order to avoid costly queries that fail in the end
        try:
            open(target, 'a').close()
        except IOError as err:
            raise IOError(
                "Problem creating request target file: " + err.args[1])

    def list(self, request):
        """
            Check the request and report a summary of statistics about the requested data.

            list with keyword 'output=cost' will generate a summary of the size, 
            number of fields and the storage information about the data: disk or tapes.
        """
        request.check()

        # check target file
        target = request.target
        self._check_target(target)

        # build the request string
        req_str = "list,output=cost," + request.to_req_str()

        # execute the request
        self.service.execute(req_str, target)

        # print the stats
        with open(target) as infile:
            print("=== request info ===")
            print(infile.read())
            print("====================")

    def retrieve(self, request):
        """
            Check and execute the request.
            Result of the request is stored to file 'target' in GRIB format.
        """
        request.check()

        # check target file
        target = request.target
        self._check_target(target)

        # build the request string
        req_str = "retreive," + request.to_req_str()

        # execute the request
        self.service.execute(req_str, target)


# the set of allowed steps in the request
ALLOWED_STEPS = set(list(range(0, 90)) + list(range(90, 144, 3)) + list(range(144, 246, 6)))

"""
    Bounding boxes in [maxLat, minLon, minLat, maxLon] format.
"""


class Area:
    Slovenia = [46.53, 13.23, 45.25, 16.36]

# Visibility and percipitation type are not available before mid 2015
AFTER_2015_PARAMS = "20.3/134.128/141.128/144.128/164.128/165.128/166.128/167.128/168.128/176.128/189.128/228.128/260015"
BEFORE_2015_PARAMS = "134.128/141.128/144.128/164.128/165.128/166.128/167.128/168.128/176.128/189.128/228.128"

class WeatherReq():
    """
        A weather data request.
    """

    def __init__(self):
        # VARIABLE PARAMETERS
        # ===================

        # mandatory params
        # ----------------

        # target:
        #   * filename where the requested data is dumped
        self.target = None

        # date:
        #   * "YYYY-MM-DD" or "YYYY-MM-DD/to/YYYY-MM-DD"
        #   * date or date range of the data
        self.date = None
        self.end_date = None

        # time:
        #   * "00:00:00" or "12:00:00" in case of forecast
        #                   OR
        #   * the time (GMT) of the weather state on each day (at step 0)
        self.time = None

        # step:
        #   * in [0,1,2,...,89] u [90,93,96,...,141] u [144,150,156,...,240]
        #   * time in hours for which the data is returned
        #   * step 0 is current weather state, step X is the forecasted weather state in X hours
        self.step = None

        # optional params
        # ----------------

        # area:
        #   * a lat/lon bounding box specifying the area for the data
        #   * format is [north, west, south, east] or differently [maxLat, minLon, minLat, maxLon]
        self.area = None

        # grid:
        #   * a lat/lon resolution of the data
        #   * format is [latRes, lonRes] (e.g. [1.5, 1.5])
        self.grid = None

        # FIXED PARAMETERS
        self.params = {
            "class": "od",
            "stream": "oper",
            "expver": "1",
            "type": "fc",
            "levtype": "sfc",
            "param": AFTER_2015_PARAMS
        }

    def __str__(self):
        """Strig representation of the request is simply the representation of its parameters."""
        max_k = max(len(key) for key in self.params.keys())
        template = "{:%d} : {}" % max_k

        ret = "ECMWF MARS API request:\n"
        ret += '\n'.join(template.format(param, val)
                         for param, val in self.params.iteritems())
        return ret

    def check(self):
        """ Check if request is consistent and has all the parameters needed for execution. """
        for param in ['target', 'date', 'time', 'step']:
            if param not in self.params:
                raise RuntimeError(
                    'Request has a missing field: \'%s\'' % param)

    def set_target(self, target):
        """Set the target filename to dump the requested data."""
        assert isinstance(
            target, str), "string expected as target filename, not %s" % repr(target)

        self.target = target
        self.params['target'] = target

    def set_date(self, req_date, end_date=None):
        """Set the date (range) of the data."""
        assert isinstance(
            req_date, date), "date object expected as input, not %s" % repr(req_date)
        if end_date is not None:
            assert isinstance(
                end_date, date), "date object expected as input, not %s" % repr(end_date)
            assert req_date < end_date, "start date should be before end date"

        self.date = req_date
        # ECMWF API expects date info serialized as YYYY-MM-DD which is Python default
        self.params['date'] = str(req_date)

        if end_date is not None:
            self.end_date = end_date
            self.params['date'] += '/to/%s' % str(end_date)

    def set_midnight(self):
        """Set measurement time to midnight (00:00:00)."""
        self.time = "00:00:00"
        self.params['time'] = "00:00:00"

    def set_noon(self):
        """Set measurement time to noon (12:00:00)."""
        self.time = "12:00:00"
        self.params['time'] = "12:00:00"

    def set_step(self, step):
        """Set the steps for which you want the weather state."""
        assert isinstance(step, (list, tuple)), "Expectiong a list or tuple"
        assert len(step) > 0, "Expecting at least some steps."
        for s in step:
            assert isinstance(s, int), "Each step should be an int."
        assert set(step).issubset(ALLOWED_STEPS), \
            "Steps %s not possible. Step values can be: %s" % (
                sorted(set(step) - ALLOWED_STEPS), ALLOWED_STEPS)

        self.step = list(sorted(step))
        self.params['step'] = '/'.join(str(s) for s in self.step)

    def set_area(self, area):
        """Set the area for which you want the data. [N,W,S,E]"""
        assert isinstance(area, (list, tuple)), "Expectiong a list or tuple"
        assert len(area) == 4, "Expecting 4 values for area."
        for res in area:
            assert isinstance(
                res, (int, float)), "Each area value should be an int or float."

        assert check_area_ranges(area), "Expecting sane area borders."

        self.area = area
        self.params['area'] = '/'.join(str(x) for x in self.area)

    def set_grid(self, grid):
        """Set the lat/lon grid resolution of the data."""
        assert isinstance(grid, (list, tuple)), "Expectiong a list or tuple"
        assert len(grid) == 2, "Expecting 2 values for grid.."
        for res in grid:
            assert isinstance(
                res, float), "Each grid resolution value should be a float."

        self.grid = grid
        latRes = ('%f' % grid[0]).rstrip('0').rstrip('.')
        lonRes = ('%f' % grid[1]).rstrip('0').rstrip('.')
        self.params['grid'] = '%s/%s' % (latRes, lonRes)

    def to_req_str(self):
        """Transform the request into a string expected by the ECMWF service."""
        return ','.join(["%s=%s" % (param, val) for param, val in sorted(self.params.items())])


def check_area_ranges(area):
    """Check if given list/tuple holds a set of sane area values."""
    # unpack values
    N, W, S, E = area
    # check ranges and ordering
    return -90 <= S < N <= 90 and -180 <= W < E <= 180
