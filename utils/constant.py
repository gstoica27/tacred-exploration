"""
Define common constants.
"""
TRAIN_JSON = 'train.json'
DEV_JSON = 'dev.json'
TEST_JSON = 'test.json'

GLOVE_DIR = 'dataset/glove'

EMB_INIT_RANGE = 1.0
MAX_LEN = 100

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
ENT_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'award': 2, 'news_agency': 3, 'education': 4, 'broadcast_program': 5, 'god': 6, 'broadcast_network': 7, 'music': 8, 'newspaper': 9, 'written_work': 10, 'art': 11, 'title': 12, 'geography': 13, 'broadcast': 14, 'product': 15, 'building': 16, 'government_agency': 17, 'person': 18, 'location': 19, 'organization': 20, 'software': 21, 'government': 22, 'park': 23, 'event': 24, 'military': 25, 'people': 26, 'internet': 27, 'transportation': 28, 'religion': 29, 'food': 30, 'language': 31}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

LABEL_TO_ID = {'/location/in_state/legislative_capital': 0, '/location/it_region/capital': 1, '/business/company_advisor/companies_advised': 2, '/location/cn_province/capital': 3, '/film/film_location/featured_in_films': 4, '/people/place_of_interment/interred_here': 5, '/business/company/major_shareholders': 6, '/people/deceased_person/place_of_burial': 7, '/film/film_festival/location': 8, '/people/deceased_person/place_of_death': 9, '/business/company/advisors': 10, '/time/event/locations': 11, '/location/in_state/administrative_capital': 12, '/location/de_state/capital': 13, '/location/location/contains': 14, '/business/business_location/parent_company': 15, '/broadcast/producer/location': 16, '/business/company/place_founded': 17, '/sports/sports_team/location': 18, '/people/person/place_lived': 19, '/people/person/religion': 20, '/people/person/place_of_birth': 21, '/location/mx_state/capital': 22, '/location/province/capital': 23, '/business/company/locations': 24, '/business/person/company': 25, '/people/person/nationality': 26, '/location/country/languages_spoken': 27, '/people/person/children': 28, '/location/jp_prefecture/capital': 29, '/location/br_state/capital': 30, '/location/administrative_division/country': 31, '/location/fr_region/capital': 32, '/location/country/administrative_divisions': 33, '/location/in_state/judicial_capital': 34, '/people/person/profession': 35, '/film/film/featured_film_locations': 36, '/people/person/ethnicity': 37, '/location/country/capital': 38, '/people/ethnicity/geographic_distribution': 39, '/business/company/founders': 40, '/people/ethnicity/included_in_group': 41, '/location/us_state/capital': 42, '/location/us_county/county_seat': 43, '/location/neighborhood/neighborhood_of': 44}

INFINITY_NUMBER = 1e12
NO_RELATION_ID = 0