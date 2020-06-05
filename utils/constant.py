
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

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'COUNTRY': 2, 'STATE_OR_PROVINCE': 3, 'NATIONALITY': 4, 'CITY': 5, 'PERSON': 6, 'ORGANIZATION': 7, 'LOCATION': 8, 'O': 9, 'CAUSE_OF_DEATH': 10, 'IDEOLOGY': 11, 'TITLE': 12, 'RELIGION': 13, 'MISC': 14, 'CRIMINAL_CHARGE': 15}

OBJ_NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'COUNTRY': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'CITY': 5, 'NATIONALITY': 6, 'STATE_OR_PROVINCE': 7, 'LOCATION': 8, 'O': 9, 'MISC': 10, 'TITLE': 11, 'IDEOLOGY': 12, 'RELIGION': 13, 'DATE': 14, 'CAUSE_OF_DEATH': 15, 'CRIMINAL_CHARGE': 16}

NER_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'O': 2, 'CAUSE_OF_DEATH': 3, 'COUNTRY': 4, 'NUMBER': 5, 'PERSON': 6, 'DATE': 7, 'TITLE': 8, 'RELIGION': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'TIME': 12, 'ORGANIZATION': 13, 'NATIONALITY': 14, 'LOCATION': 15, 'CITY': 16, 'ORDINAL': 17, 'DURATION': 18, 'SET': 19, 'IDEOLOGY': 20, 'MONEY': 21, 'PERCENT': 22, 'CRIMINAL_CHARGE': 23, 'URL': 24}


POS_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'DT': 2, 'RB': 3, 'JJ': 4, 'NN': 5, 'IN': 6, 'WDT': 7, 'VBD': 8, 'NNS': 9, 'NNP': 10, 'CC': 11, 'MD': 12, 'VB': 13, 'CD': 14, 'PRP': 15, 'VBP': 16, '.': 17, 'POS': 18, ',': 19, 'TO': 20, 'VBN': 21, 'VBG': 22, 'PRP$': 23, 'VBZ': 24, 'HYPH': 25, 'WP': 26, 'NNPS': 27, 'ADD': 28, 'WP$': 29, 'RBR': 30, 'WRB': 31, 'EX': 32, 'PDT': 33, ':': 34, 'JJR': 35, 'JJS': 36, '$': 37, 'RP': 38, 'NFP': 39, 'RBS': 40, 'SYM': 41, '-LRB-': 42, '-RRB-': 43, '``': 44, "''": 45, 'UH': 46, 'FW': 47, 'AFX': 48, 'LS': 49}

DEPREL_TO_ID = {'<PAD>': 0, '<UNK>': 1, 'det': 2, 'advmod': 3, 'amod': 4, 'nsubj': 5, 'case': 6, 'nmod': 7, 'acl:relcl': 8, 'obj': 9, 'cc': 10, 'conj': 11, 'aux': 12, 'cop': 13, 'nummod': 14, 'root': 15, 'punct': 16, 'nmod:poss': 17, 'flat': 18, 'compound': 19, 'obl': 20, 'mark': 21, 'advcl': 22, 'xcomp': 23, 'ccomp': 24, 'parataxis': 25, 'obl:tmod': 26, 'csubj': 27, 'nmod:tmod': 28, 'appos': 29, 'list': 30, 'aux:pass': 31, 'acl': 32, 'nsubj:pass': 33, 'expl': 34, 'fixed': 35, 'det:predet': 36, 'obl:npmod': 37, 'compound:prt': 38, 'nmod:npmod': 39, 'cc:preconj': 40, 'iobj': 41, 'discourse': 42, 'goeswith': 43, 'vocative': 44, 'orphan': 45, 'reparandum': 46, 'csubj:pass': 47, 'flat:foreign': 48}

NEGATIVE_LABEL = 'None'

# LABEL_TO_ID = {'Other': 0, 'Entity-Destination': 1, 'Cause-Effect': 2, 'Member-Collection': 3, 'Entity-Origin': 4, 'Message-Topic': 5, 'Component-Whole': 6, 'Instrument-Agency': 7, 'Product-Producer': 8, 'Content-Container': 9, 'Entity-Destination-rev': 10, 'Cause-Effect-rev': 11, 'Member-Collection-rev': 12, 'Entity-Origin-rev': 13, 'Message-Topic-rev': 14, 'Component-Whole-rev': 15, 'Instrument-Agency-rev': 16, 'Product-Producer-rev': 17, 'Content-Container-rev': 18}
LABEL_TO_ID = {'None': 0, '/location/administrative_division/country': 1, '/location/country/administrative_divisions': 2, '/people/deceased_person/place_of_death': 3, '/people/person/place_lived': 4, '/location/location/contains': 5, '/location/country/capital': 6, '/people/person/nationality': 7, '/location/neighborhood/neighborhood_of': 8, '/business/person/company': 9, '/people/person/children': 10, '/business/company/founders': 11, '/people/person/place_of_birth': 12, '/people/ethnicity/geographic_distribution': 13, '/business/company/place_founded': 14, '/sports/sports_team/location': 15, '/sports/sports_team_location/teams': 16, '/business/company/major_shareholders': 17, '/business/company_shareholder/major_shareholder_of': 18, '/people/person/religion': 19, '/business/company/advisors': 20, '/people/ethnicity/people': 21, '/people/person/ethnicity': 22}
# LABEL_TO_ID = { 'Other': 0, 'Component-Whole(e2,e1)': 1, 'Instrument-Agency(e2,e1)': 2, 'Member-Collection(e1,e2)': 3, 'Cause-Effect(e2,e1)': 4, 'Entity-Destination(e1,e2)': 5, 'Content-Container(e1,e2)': 6, 'Message-Topic(e1,e2)': 7, 'Product-Producer(e2,e1)': 8, 'Member-Collection(e2,e1)': 9, 'Entity-Origin(e1,e2)': 10, 'Cause-Effect(e1,e2)': 11, 'Component-Whole(e1,e2)': 12, 'Message-Topic(e2,e1)': 13, 'Product-Producer(e1,e2)': 14, 'Entity-Origin(e2,e1)': 15, 'Instrument-Agency(e1,e2)': 16, 'Content-Container(e2,e1)': 17, 'Entity-Destination(e2,e1)': 18}

INFINITY_NUMBER = 1e12

















