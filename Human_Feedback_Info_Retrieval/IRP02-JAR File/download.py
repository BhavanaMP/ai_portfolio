import requests
import re
import random
import os
import pandas as pd
    
def ExtractTopics(extractedDataSetFolderPath,wikiTopicsFilePath,num_items):
    
    #the program extracts text and image data for a given topic
    #and writes them in the CURRENT DIRECTORY, relative to the location the py script is invoked from
    #make sure you set it properly, in case you dont want the default one

    #this is the title we will search
    #topic = "Rahul_Dravid"

    root_folder = extractedDataSetFolderPath #r"C:\Users\user\IRUpdated\Dataset"
    word_file = wikiTopicsFilePath#"./Wikipedia_topics/Wikipedia_topics"
    with open(word_file,encoding="utf8") as infile:
        topic = random.choices(list(open(word_file,encoding="utf8")),k=num_items) #choosing random items from the text file into a list
        topic = list(map(lambda s: s.strip(), topic)) #removing the /n char sequence
    for i in range(len(topic)):
        tpc = topic[i]
        if (tpc.find('\\') !=-1 or tpc.find('/')!=-1 or tpc.find(':')!=-1 or tpc.find('*')!=-1 or tpc.find('?')!=-1 or tpc.find('"')!=-1 or tpc.find('<')!=-1 or tpc.find('>')!=-1 or tpc.find('|')!=-1):
            continue
        os.mkdir(os.path.join(root_folder,tpc))
        path = os.path.join(root_folder,tpc)
        os.chdir(path)
     #this is the config for to get the first introduction of a title
        text_config = {
            'action': 'query',
            'format': 'json',
            'titles': topic[i],
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True,
        }
        text_response = requests.get('https://en.wikipedia.org/w/api.php',params=text_config).json()
        text_page = next(iter(text_response['query']['pages'].values()))
        file1 = open(text_page['title']+".txt","w",encoding="utf8")#write mode
        file1.write(text_page['extract'])
        file1.close()
    #print(text_page['extract'])

    #this is the config to get the images that are in the topic
    #we use this to count the number of images
        num_image_config = {
            'action': 'parse',
            'pageid': text_page['pageid'],
            'format': 'json',
            'prop': 'images'
        }
        num_image_response = requests.get('https://en.wikipedia.org/w/api.php',params=num_image_config).json()



    #now that we havae the number of images in the page, we ask for the images that are in the page with the title
        image_config = {
            'action': 'query',
            'format': 'json',
            'titles': topic[i],
            'prop': 'images',
            'imlimit': len(num_image_response['parse']['images'])
        }
        image_response = requests.get('https://en.wikipedia.org/w/api.php',params=image_config).json()
        image_page = next(iter(image_response['query']['pages'].values()))


    #and we  write the image files one by one in the currect directory
    #we also dont write the svg files, since as they are mostly the logos
    #modily the filename_pattern regex for to accept the proper files
        print("writing files")
        filename_pattern = re.compile(".*\.(?:jpe?g|gif|png|JPE?G|GIF|PNG)")
        try:
            for j in range(len(image_page['images'])):

                url_config = {
                'action': 'query',
                'format': 'json',
                'titles': image_page['images'][j]['title'],
                'prop': 'imageinfo',
                'iiprop': 'url'
                }
                url_response = requests.get('https://en.wikipedia.org/w/api.php',params=url_config).json()
                url_page = next(iter(url_response['query']['pages'].values()))
                print(url_page['imageinfo'][0]['url'])
                if(filename_pattern.search(url_page['imageinfo'][0]['url'])):

                    print("writing image "+url_page['imageinfo'][0]['url'].rsplit("/",1)[1])
                    with open(url_page['imageinfo'][0]['url'].rsplit("/",1)[1], 'wb') as handle:

                        response = requests.get(url_page['imageinfo'][0]['url'], stream=True)

                        if not response.ok:
                            print (response)

                        for block in response.iter_content(1024):
                            if not block:
                                continue

                            handle.write(block)
        except:
            pass
