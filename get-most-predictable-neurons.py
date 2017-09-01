# Simple script to get ordered list of "predictable" neurons
# Frequently returns interpretable ones, since information that multiple networks
# learn is frequently significant.
import json

data = json.load(open(sys.argv[1]))

models = data.keys()

result = {}

for target in models:
    overall_error = []

    for x in range(500):
        avg = 0
        for source in models:
            avg += data[source][target][0][x] / len(models)
        overall_error.append(avg)

    sorted_indices = list(sorted(enumerate(overall_error), key = lambda x: x[1]))
    result[target] = sorted_indices

json.dump(result, open(sys.argv[2]), 'w'), indent = 2)
