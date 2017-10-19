""" Requires bs4 for XML parsing """
from bs4 import BeautifulSoup
import argparse
import json

def weather_xml_to_json(xml_data):
    """
        Get locations of ARSO weather-measuring station.
    """
    soup = BeautifulSoup(xml_data)
    
    ws = [] # list of weather-measuring stations
    for metdata in soup.find_all('metdata'):
        
        ws.append({
            'lat': float(metdata.domain_lat.text),
            'lon': float(metdata.domain_lon.text),
            'alt': float(metdata.domain_altitude.text),
            'title': metdata.domain_title.text,
            'shortTitle': metdata.domain_shorttitle.text
        })

    return json.dumps(ws, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ARSO xml to json format.')
    parser.add_argument('xml_file', type=str, help='Path of ARSO xml file')
    parser.add_argument('json_file', type=str, help='Path of json output file')
    args = parser.parse_args()

    xml_data = ''
    with open(args.xml_file, 'r') as f:
        xml_data = f.read()
    
    json_data = weather_xml_to_json(xml_data)
    with open(args.json_file, 'w') as f:
        f.write(json_data)

