import gzip
import json

def load_data():
    g = gzip.open('data.gz', 'r')

    data = []
    for l in g:
        data.append(json.loads(l))

    return data

data = load_data()

"""
Counter({'Tools & Home Improvement': 74358,
 'Automotive': 73434,
   'Arts, Crafts & Sewing': 72556,
     'Toys & Games': 72034,
       'Office Products': 71681, 
       'Amazon Home': 71362, 'Grocery': 70184,
         'Sports & Outdoors': 70177, 
         'Books': 69685,
           'Computers': 67156, 
           'Movies & TV': 60608,
             'Amazon Fashion': 59747,
               'Cell Phones & Accessories': 59432,
                 'Pet Supplies': 57490, 
                 'Industrial & Scientific': 55503, 
                 'All Electronics': 52878, 
                 'Digital Music': 48319, 
                 'Camera & Photo': 36687,
                   'Musical Instruments': 35873,
                     'Home Audio & Theater': 33401, 
                     'Video Games': 21736,
                       'Health & Personal Care': 14116})
"""