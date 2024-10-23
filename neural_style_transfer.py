import requests

api_key = '827fa0c0-f519-40a1-8ef4-2072fef5047f'

content_image_url = 'URL_OF_YOUR_CONTENT_IMAGE'  
style_image_url = 'URL_OF_YOUR_STYLE_IMAGE'      

url = 'https://api.deepai.org/api/style-transfer'

response = requests.post(
    url,
    data={
        'content': content_image_url,
        'style': style_image_url,
    },
    headers={'api-key': api_key}
)

if response.status_code == 200:
    result = response.json()
    output_image_url = result['output_url']
    print('Output image URL:', output_image_url)
else:
    print('Error:', response.text)
