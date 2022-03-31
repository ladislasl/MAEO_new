# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os

#Species dictionnary
dict = {1 : {'family' : 'Myliobatidae_1', 'species' : ['Aetobatus narinari','Aetobatus ocellatus']},
        20 : {'family' : 'Myliobatidae_2', 'species' : ['Myliobatis aquila']},
        2 : {'family' : 'Dasyatidae', 'species' : ['Dasyatis thetidis','Taeniura melanospilos','Taeniura meyeni','Pteroplatytrygon violacea',
                        'Pateobatis fai']},
        3 : {'family' : 'Alopiidae', 'species' : ['Alopias pelagicus','Alopias superciliosus','Alopias vulpinus']},
        4 : {'family' : 'Carcharhinidae_1', 'species' : ['Carcharhinus albimarginatus','Carcharhinus amblyrhynchos','Carcharhinus brevipinna',
                        'Carcharhinus falciformis', 'Carcharhinus leucas','Carcharhinus limbatus','Carcharhinus melanopterus',
                        'Carcharhinus plumbeus','Galeocerdo cuvier','Negaprion acutidens','Triaenodon obesus']}, #Bull, Tiger, ...
        5 : {'family' : 'Carcharhinidae_2', 'species' : ['Carcharhinus longimanus']},
        6 : {'family' : 'Carcharhinidae_3', 'species' : ['Loxodon macrorhinus','Prionace glauca']},
        7 : {'family' : 'Ginglymostomatidae', 'species' : ['Pseudoginglymostoma brevicaudatum','Nebrius ferrugineus']}, #Nurse
        8 : {'family' : 'White', 'species' : ['Carcharodon carcharias']}, #White
        9 : {'family' : 'Mako', 'species' : ['Isurus oxyrinchus','Isurus paucus']}, #Mako
        10 : {'family' : 'Manta', 'species' : ['Manta alfredi','Manta birostris']},
        11 : {'family' : 'Mobula', 'species' : ['Mobula eregoodootenkee','Mobula japanica','Mobula tarapacana','Mobula thurstoni']},
        12 : {'family' : 'Whale', 'species' : ['Rhincodon typus']},
        13 : {'family' : 'Guitarfish', 'species' : ['Rhynchobatus djiddensis']},
        14 : {'family' : 'Striped', 'species' : ['Poroderma africanum']},
        15 : {'family' : 'Hammer', 'species' : ['Sphyrna lewini','Sphyrna mokarran','Sphyrna zygaena']},
        16 : {'family' : 'Zebra', 'species' : ['Stegostoma fasciatum']},
        17 : {'family' : 'Torpedinidae', 'species' : ['Torpedo fuscomaculata']},
        18 : {'family' : 'Torpedinidae', 'species' : ['Torpedo fuscomaculata']},
        19 : {'family' : 'Starspotted', 'species' : ['Mustelus manazo']}}
#Define file path
webdriver_path = os.path.normpath(os.getcwd()+"\\webdriver\\chromedriver.exe")
#image_path = os.path.normpath(os.getcwd()+"\\Myliobatidae")

#Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
#search_keys= ["Aetobatus narinari","Aetobatus ocellatus"]

#Parameters
number_of_images = 200
headless = False
min_resolution=(0,0)
max_resolution=(3000,3000)

#Main program
for key in dict:
    print(dict[key]['family'])
    for search_key in dict[key]['species']:
        image_path = os.path.normpath(os.getcwd()+"\\"+dict[key]['family'])
        image_scrapper = GoogleImageScraper(webdriver_path,image_path,search_key,number_of_images,headless,min_resolution,max_resolution)
        image_urls = image_scrapper.find_image_urls()
        image_scrapper.save_images(image_urls)
    #Release resources    
    del image_scrapper