from bs4 import BeautifulSoup
import requests

response = requests.get("https://news.ycombinator.com/")
site = BeautifulSoup(response.text, 'html.parser')

links = site.find_all(class_="titleline")
text = []
href = []
scores_list = []
scores_list2 = []
for link in links:
    text.append(link.find(name="a").getText())
    href.append(link.find(name="a").get("href"))

scores = site.find_all(class_="score")
for score in scores:
    scores_list.append(score.getText())

for score in scores_list:
    scores_list2.append(int(score.split()[0]))

zipper = zip(text, href)
onion_zip = zip(zipper, scores_list2)

for dat in onion_zip:
    if dat[1] == max(scores_list2):
        print(dat)



for layer in onion_zip:
    print(layer)
