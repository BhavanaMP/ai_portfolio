import random
import requests
word_file = "./Wikipedia_topics/Wikipedia_topics.txt"
with open(word_file,encoding="utf8") as infile:
    input = random.choices(list(open(word_file,encoding="utf8")),k=1000) #choosing random items from the text file into a list
    input = list(map(lambda s: s.strip(), input)) #removing the /n char sequence
for i in range(len(input)):
    #print(input[i])
    text_config = {
        'action': 'query',
        'format': 'json',
        'titles': input[i],
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
    }
    #print(text_config)
    text_response = requests.get('https://en.wikipedia.org/w/api.php', params=text_config).json()
    text_page = next(iter(text_response['query']['pages'].values()))
    print(text_page)
    try:
       file1 = open(text_page['title']+".txt","w",encoding="utf8")#write mode
       file1.write(text_page['extract'])
       file1.close()
    except OSError as e:
        pass

