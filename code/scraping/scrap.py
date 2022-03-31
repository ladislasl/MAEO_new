import requests
from bs4 import BeautifulSoup
import urllib.request

url = "http://www.reddit.com/r/BabyYoda"

response = requests.get(url)

soup = BeautifulSoup(response.content, "html.parser")

images = soup.find_all("img", attrs = {"alt": "Post image"}) #profile pictures ? 4:00

number = 0
for image in images:
    if number < 6:
        image_src = image["src"]
        print(image_src)
        urllib.request.urlretrieve(image_src,str(number))
        number += 1