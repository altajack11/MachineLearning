car_columns = ['buying', 'maint', 'doors',
               'persons', 'lug_boot', 'safety', 'label']

car_attribute_values = {
    'buying': ['vhigh', 'high', 'med', 'low'],
    'maint': ['vhigh', 'high', 'med', 'low'],
    'doors': ['2', '3', '4', '5more'],
    'persons': ['2', '4', 'more'],
    'lug_boot': ['small', 'med', 'big'],
    'safety': ['low', 'med', 'high'],
}

car_attribute_types = {
    'buying': "categorical",
    'maint': "categorical",
    'doors': "categorical",
    'persons': "categorical",
    'lug_boot': "categorical",
    'safety': "categorical",
}

car_labels = ['unacc', 'acc', 'good', 'vgood']


bank_columns = ['age',
                'job',
                'marital',
                'education',
                'default',
                'balance',
                'housing',
                'loan',
                'contact',
                'day',
                'month',
                'duration',
                'campaign',
                'pdays',
                'previous',
                'poutcome',
                "label"]
bank_attribute_values = {'age': [],
                         'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
                                 "blue-collar", "self-employed", "retired", "technician", "services"],
                         'marital': ["married", "divorced", "single"],
                         'education': ["unknown", "secondary", "primary", "tertiary"],
                         'default': ["yes", "no"],
                         'balance': [],
                         'housing': ["yes", "no"],
                         'loan': ["yes", "no"],
                         'contact': ["unknown", "telephone", "cellular"],
                         'day': [],
                         'month': ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                         'duration': [],
                         'campaign': [],
                         'pdays': [],
                         'previous': [],
                         'poutcome': ["unknown", "other", "failure", "success"]}
bank_attribute_types = {'age': "numeric",
                        'job': "categorical",
                        'marital': "categorical",
                        'education': "categorical",
                        'default': "categorical",
                        'balance': "numeric",
                        'housing': "categorical",
                        'loan': "categorical",
                        'contact': "categorical",
                        'day': "numeric",
                        'month': "categorical",
                        'duration': "numeric",
                        'campaign': "numeric",
                        'pdays': "numeric",
                        'previous': "numeric",
                        'poutcome': "categorical"}

bank_labels = ['yes', 'no']
