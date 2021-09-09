import numpy as np
import sys
import os

assert len(sys.argv) == 5, 'A dataset name (digits, office-home or office31), a method (src | dann | m3sda | mdan | mdmn | darn | tar), a projection functions (L2 or dynamic) and a file prefix must be passed.'

dataset = sys.argv[1]
method  = sys.argv[2]
search  = sys.argv[3]
path = os.path.join('results', dataset, method, search)

prefix = sys.argv[4]

filenames = os.listdir(path)
filenames = list(filter(lambda x: x.startswith(prefix) and x.endswith('test'), filenames))
filenames = sorted(filenames, key=lambda x: int(x[-6]))

results = []
for f in filenames:
	r = open(os.path.join(path, f), 'r')
	values = r.readlines()
	values = list(map(lambda x: float(x.strip()), values))
	results.append(values)
	r.close()

results = np.array(results)
print(results)

print()
print('domain mean: ', np.average(results, axis=0))
print('std err: ', np.std(results, axis=0)/np.sqrt(results.shape[0]))
print()

mean = np.average(results, axis=1)
print(mean)
print('Average: {}, stderr: {}'.format(np.average(mean), np.std(mean)/np.sqrt(mean.shape[0])))

	

