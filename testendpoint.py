import requests 
import json
URL = 'https://m-e0ef896ebd654a899ed28b0e3785fcf9-m.default.model-v2.inferless.com/v2/models/Toxic-Language_e0ef896ebd654a899ed28b0e3785fcf9/versions/1/infer'
headers = {"Content-Type": "application/json",  "Authorization": "Bearer 84d51dc484078596e1d023a8dd55dbff5e9c7cb7d861dedb5a72cf0d85a2a514c6d329ad789f2353e9f08cb682f51a8233fdf4c85ccf719b7c2f78f9377ac3cb"}
          
data = {
	"inputs": [
		{
			"data": [
				"fuck this man this shit is not working"
			],
			"name": "text",
			"shape": [
				1
			],
			"datatype": "BYTES"
		},
		{
			"data": [
				"fuck shit man"
			],
			"name": "text1",
			"shape": [
				1
			],
			"datatype": "BYTES"
		},
	]
}
response = requests.post(URL, headers=headers, data=json.dumps(data))
print(response.json())