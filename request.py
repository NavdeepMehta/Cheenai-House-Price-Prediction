import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'AREA':2, 'INT_SQFT':2122, 'N_BEDROOM':3,'N_BATHROOM':2, 'PARK_FACIL':2, 'BUILDTYPE':2,'STREET':1, 'MZZONE':2})

print(r.json())