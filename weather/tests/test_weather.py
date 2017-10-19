#!/usr/bin/python

"""
ECMWF MARS archive access code.
"""

from .. import request
import unittest

from datetime import date


class TestEcmwfServer(unittest.TestCase):
    """Unit tests for the WeatherReq class."""
    @classmethod
    def setUpClass(self):
        # prepare the server object
        self.server = request.EcmwfServer()

        # prepare a test request
        self.req = request.WeatherReq()
        self.req.set_target("TEST.txt")
        self.req.set_date(date(2016, 9, 17))
        self.req.set_midnight()
        self.req.set_step((1, 93, 234))


    def test_list(self):
        """
        Test if request stats are queried ok.
        Only checks the offline checks of the function.
        Does not actually query the ECMWF service.
        """
        # complain if the request is not complete
        self.req.target = None
        with self.assertRaises(RuntimeError):
            self.server.list(self.req)

        # test with a bad file path
        self.req.set_target("thisFolderDoesNotExist/TEST.txt")
        with self.assertRaises(IOError):
            self.server.list(self.req)


class TestWeatherReq(unittest.TestCase):
    """Unit tests for the WeatherReq class."""
    @classmethod
    def setUpClass(self):
        # no setup needed
        pass


    def test_is_complete(self):
        """Test if request is correctly checked for completeness."""
        req = request.WeatherReq()
        # request is not complete until all fields are set
        self.assertFalse(req.is_complete())
        req.set_target("filename.bin")
        self.assertFalse(req.is_complete())
        req.set_date(date(2016, 9, 17))
        self.assertFalse(req.is_complete())
        req.set_midnight()
        self.assertFalse(req.is_complete())
        req.set_step((1, 93, 234))
        self.assertTrue(req.is_complete())


    def test_set_target(self):
        """Test if request target is set properly."""
        req = request.WeatherReq()
        target = "filename.bin"

        req.set_target(target)
        self.assertEqual(req.target, "filename.bin")
        self.assertEqual(req.params['target'], "filename.bin")

        with self.assertRaises(AssertionError):
            req.set_target(1)


    def test_set_date(self):
        """Test if request dates are set properly."""
        req = request.WeatherReq()
        req_date = date(2016, 9, 17)

        # set just one date
        req.set_date(req_date)
        self.assertEqual(req.date, req_date)
        self.assertEqual(req.params['date'], '2016-09-17')

        # set both start and end date
        req_date = date(2016, 9, 18)
        end_date = date(2016, 9, 20)

        req.set_date(req_date, end_date)
        self.assertEqual(req.date, req_date)
        self.assertEqual(req.end_date, end_date)
        self.assertEqual(req.params['date'], '2016-09-18/to/2016-09-20')

        # error if not given date
        with self.assertRaises(AssertionError):
            req.set_date("not_date", end_date)
        with self.assertRaises(AssertionError):
            req.set_date(req_date, "not_date")

        # error if start date not before end date
        with self.assertRaises(AssertionError):
            req.set_date(end_date, req_date)
        with self.assertRaises(AssertionError):
            req.set_date(req_date, req_date)


    def test_set_midnight(self):
        """Test if request time is properly set to midnight."""
        req = request.WeatherReq()
        req.set_midnight()
        self.assertEqual(req.time, "00:00:00")
        self.assertEqual(req.time, req.params['time'])


    def test_set_noon(self):
        """Test if request time is properly set to noon."""
        req = request.WeatherReq()
        req.set_noon()
        self.assertEqual(req.time, "12:00:00")
        self.assertEqual(req.time, req.params['time'])


    def test_set_step(self):
        """Test if request step is properly set."""
        req = request.WeatherReq()

        # all allowed values are ok
        req.set_step(sorted(request.ALLOWED_STEPS))
        self.assertEqual(req.step, sorted(request.ALLOWED_STEPS))

        # should be list or tuple
        req.set_step((1, 93, 234))
        self.assertEqual(req.step, [1, 93, 234])
        self.assertEqual(req.params['step'], "1/93/234")
        with self.assertRaises(AssertionError):
            req.set_step("bad_input")

        # all values should be ints
        with self.assertRaises(AssertionError):
            req.set_step([1, '2', 3])

        # empty list not allowed
        with self.assertRaises(AssertionError):
            req.set_step([])

        # test not allowed values
        with self.assertRaises(AssertionError):
            req.set_step([91])
        with self.assertRaises(AssertionError):
            req.set_step([1000])
        with self.assertRaises(AssertionError):
            req.set_step([-3])


    def test_to_req_str(self):
        """Test if request is properly formatted for querying the ECMWF service."""
        test_req = request.WeatherReq()
        test_req.set_date(date(2016, 9, 17))
        test_req.set_noon()
        test_req.set_step([0])

        expected = "class=od,date=2016-09-17,expver=1,levtype=sfc,param=167.128,step=0,stream=oper,time=12:00:00,type=fc"
        self.assertEqual(test_req.to_req_str(), expected)


if __name__=="__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEcmwfServer)
    unittest.TextTestRunner(verbosity=3).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestWeatherReq)
    unittest.TextTestRunner(verbosity=3).run(suite)
