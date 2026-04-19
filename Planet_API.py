import os
import json
import requests
import geojsonio
import time
from requests.auth import HTTPBasicAuth
from requests import Session



# Helper function to print formatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))

# if your Planet API Key is not set as an environment variable, you can paste it below
if os.environ.get('PLA...ec', ''):
    API_KEY = os.environ.get('PLA...ec', '')
else:
  API_KEY = 'PLA...ec'



path = r'/image_repo/.../10_west/519064.geojson'
# get geojson geometry coordinates
with open(path) as f:
    data = json.load(f)
    geojson_geometry = data["features"][0]['geometry']
    # print(geojson_geometry)

# get images that overlap with our AOI
geometry_filter = {"type": "GeometryFilter","field_name": "geometry","config": geojson_geometry}

# get images acquired within a date range
date_range_filter = {  "type": "DateRangeFilter","field_name": "acquired",
                       "config": {"gte": "2022-06-01T00:00:00.000Z", "lte": "2022-07-31T00:00:00.000Z"}
                    }

# only get images which have <10% cloud coverage
cloud_cover_filter = {"type": "RangeFilter","field_name": "cloud_cover","config": {"gte":0.0, "lte": 0.1}}

# combine our geo, date, cloud filters
combined_filter = {"type": "AndFilter", "config": [geometry_filter, date_range_filter, cloud_cover_filter]}

# define item type
item_type = "PSScene"

# API request object
search_request = {"item_types": [item_type], "filter": combined_filter}

# fire off the POST request
search_result = \
  requests.post(
    'https://api.planet.com/data/v1/quick-search',
    auth=HTTPBasicAuth(API_KEY, ''),
    json=search_request)

geojson = search_result.json()

# let's look at the first result
# print(list(geojson.items())[1][1][0])
#
# extract image IDs only
image_ids = [feature['id'] for feature in geojson['features']]
# print(image_ids)

# download '20220603_182723_26_248f' "ortho_analytic_4b_sr"
last_id = image_ids[-1]
id_url = 'https://api.planet.com/data/v1/item-types/{}/items/{}/assets'.format(item_type, last_id)


# Returns JSON metadata for assets in this ID. Learn more: planet.com/docs/reference/data-api/items-assets/#asset
result = \
  requests.get(
    id_url,
    auth=HTTPBasicAuth(API_KEY, '')
  )

# List of asset types available for this particular satellite image
print(result.json().keys())

# find the status of the image
print(result.json()['ortho_analytic_4b_sr']['status'])

# Parse out useful links
links = result.json()[u"ortho_analytic_4b_sr"]["_links"]
self_link = links["_self"]
activation_link = links["activate"]

# Request activation of the 'ortho_analytic_4b' asset:
activate_result = \
  requests.get(activation_link, auth=HTTPBasicAuth(API_KEY, ''))

# find the status of the image
print(result.json()['ortho_analytic_4b_sr']['status'])