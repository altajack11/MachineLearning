import json
import pandas as pd

# LABELS = [
#     "unacc", "acc", "good", "vgood"
# ]

# ATTRIBUTES = [
#     "buying": ["vhigh", "high", "med", "low"],
#     "maint":    ["vhigh", "high", med, low],
#     "doors":    [2, 3, 4, 5more],
#     "persons":  [2, 4, more],
#     "lug_boot": [small, med, big],
#     "safety":   [low, med, high]
# ]

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train = pd.read_csv('car/train.csv')

print(train)
