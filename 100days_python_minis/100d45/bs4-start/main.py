from bs4 import BeautifulSoup
from wheel.cli import parser

with open("website.html", "r") as site:
    content = site.read()
    site.close()

soup =BeautifulSoup(content, 'html.parser')
paragraphs = soup.find_all(name="p")
for paragraph in paragraphs:
    print(paragraph.getText())
    print(paragraph.get("font-family"))

#find a p tag that is inside an a tag
company_url = soup.select_one(selector="p a")
#find element by id
name = soup.select_one(selector="#name")
#select the element that has a class of heading
headings = soup.select(".heading")
for heading in headings:
    print(heading.getText())