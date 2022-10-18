import pickle as pkl


# Open pickle results from demo script
paths = {
	"proba": "demo/results/probabilities.pkl",
	"predictions": "demo/results/predictions.pkl"
}

f_propba = open(paths["proba"], 'rb')
proba = pkl.load(f_propba)

f_predictions = open(paths["predictions"], 'rb')
f_predictions = pkl.load(f_predictions)


print(proba)
print()
print(f_predictions)
