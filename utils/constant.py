
"""
Define constants for semeval-10 task.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
NO_RELATION_ID = 0
MAX_LEN = 100

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'PERSON': 2, 'COUNTRY': 3, 'LOCATION': 4, 'ORGANIZATION': 5, 'CITY': 6, 'STATE_OR_PROVINCE': 7, 'CAUSE_OF_DEATH': 8, 'NATIONALITY': 9, 'RELIGION': 10, 'O': 11, 'MISC': 12, 'IDEOLOGY': 13, 'TITLE': 14, 'ORDINAL': 15, 'DATE': 16, 'CRIMINAL_CHARGE': 17, 'URL': 18, 'TIME': 19, 'NUMBER': 20, 'SET': 21}

OBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'PERSON': 2, 'LOCATION': 3, 'COUNTRY': 4, 'CITY': 5, 'ORGANIZATION': 6, 'STATE_OR_PROVINCE': 7, 'RELIGION': 8, 'MISC': 9, 'O': 10, 'CAUSE_OF_DEATH': 11, 'NATIONALITY': 12, 'IDEOLOGY': 13, 'TITLE': 14, 'ORDINAL': 15, 'DATE': 16, 'CRIMINAL_CHARGE': 17, 'URL': 18, 'TIME': 19, 'NUMBER': 20}

NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'PERSON': 3, 'MISC': 4, 'DATE': 5, 'COUNTRY': 6, 'CRIMINAL_CHARGE': 7, 'LOCATION': 8, 'CAUSE_OF_DEATH': 9, 'CITY': 10, 'ORDINAL': 11, 'ORGANIZATION': 12, 'PERCENT': 13, 'MONEY': 14, 'NUMBER': 15, 'TITLE': 16, 'STATE_OR_PROVINCE': 17, 'NATIONALITY': 18, 'RELIGION': 19, 'DURATION': 20, 'SET': 21, 'IDEOLOGY': 22, 'TIME': 23, 'URL': 24, 'EMAIL': 25}


POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'CC': 2, 'NNP': 3, 'POS': 4, 'JJ': 5, 'NN': 6, 'IN': 7, ',': 8, 'RB': 9, 'VBN': 10, 'VBG': 11, 'HYPH': 12, 'NNS': 13, 'VB': 14, 'DT': 15, 'WDT': 16, 'MD': 17, '.': 18, 'PRP': 19, 'VBD': 20, 'NNPS': 21, 'RBR': 22, 'PRP$': 23, 'JJS': 24, 'TO': 25, 'CD': 26, '$': 27, 'JJR': 28, 'VBZ': 29, 'VBP': 30, 'RBS': 31, 'WP': 32, 'WP$': 33, '-LRB-': 34, 'RP': 35, 'WRB': 36, ':': 37, 'PDT': 38, 'UH': 39, 'SYM': 40, 'EX': 41, "''": 42, 'ADD': 43, '``': 44, '-RRB-': 45, 'NFP': 46, 'AFX': 47, 'LS': 48, 'FW': 49, 'GW': 50}

DEPREL_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'cc': 2, 'nmod:poss': 3, 'flat': 4, 'case': 5, 'amod': 6, 'nsubj': 7, 'nmod': 8, 'punct': 9, 'nsubj:pass': 10, 'conj': 11, 'advmod': 12, 'acl': 13, 'advcl': 14, 'xcomp': 15, 'compound': 16, 'obl': 17, 'root': 18, 'det': 19, 'obj': 20, 'aux': 21, 'acl:relcl': 22, 'obl:tmod': 23, 'appos': 24, 'cop': 25, 'mark': 26, 'nummod': 27, 'obl:npmod': 28, 'aux:pass': 29, 'fixed': 30, 'ccomp': 31, 'parataxis': 32, 'list': 33, 'compound:prt': 34, 'nmod:npmod': 35, 'iobj': 36, 'csubj': 37, 'det:predet': 38, 'nmod:tmod': 39, 'discourse': 40, 'expl': 41, 'vocative': 42, 'cc:preconj': 43, 'orphan': 44, 'reparandum': 45, 'goeswith': 46, 'csubj:pass': 47, 'flat:foreign': 48}

NEGATIVE_LABEL = 'None'

# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9, 'Entity-Destination-rev': 10, 'Cause-Effect-rev': 11, 'Member-Collection-rev': 12, 'Entity-Origin-rev': 13, 'Message-Topic-rev': 14, 'Component-Whole-rev': 15, 'Instrument-Agency-rev': 16, 'Product-Producer-rev': 17, 'Content-Container-rev': 18}
LABEL_TO_ID = {'None': 0, '/location/country/administrative_divisions': 1, '/location/administrative_division/country': 2, '/location/location/contains': 3, '/location/country/capital': 4, '/business/company/founders': 5, '/business/person/company': 6, '/people/person/nationality': 7, '/people/deceased_person/place_of_death': 8, '/people/person/place_of_birth': 9, '/people/person/place_lived': 10, '/location/neighborhood/neighborhood_of': 11, '/business/company_shareholder/major_shareholder_of': 12, '/business/company/major_shareholders': 13, '/people/person/children': 14, '/sports/sports_team/location': 15, '/sports/sports_team_location/teams': 16, '/business/company/place_founded': 17, '/people/ethnicity/people': 18, '/people/person/ethnicity': 19, '/people/person/religion': 20, '/people/ethnicity/geographic_distribution': 21, '/business/company/advisors': 22, '/people/person/profession': 23, '/business/company/industry': 24}
# LABEL_TO_ID = { 'Other': 0, 'Component-Whole(e2,e1)': 1, 'Instrument-Agency(e2,e1)': 2, 'Member-Collection(e1,e2)': 3, 'Cause-Effect(e2,e1)': 4, 'Entity-Destination(e1,e2)': 5, 'Content-Container(e1,e2)': 6, 'Message-Topic(e1,e2)': 7, 'Product-Producer(e2,e1)': 8, 'Member-Collection(e2,e1)': 9, 'Entity-Origin(e1,e2)': 10, 'Cause-Effect(e1,e2)': 11, 'Component-Whole(e1,e2)': 12, 'Message-Topic(e2,e1)': 13, 'Product-Producer(e1,e2)': 14, 'Entity-Origin(e2,e1)': 15, 'Instrument-Agency(e1,e2)': 16, 'Content-Container(e2,e1)': 17, 'Entity-Destination(e2,e1)': 18}

INFINITY_NUMBER = 1e12

















