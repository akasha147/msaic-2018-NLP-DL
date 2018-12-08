
original = open('data.tsv','r')
oversampleddata = open('oversampled_data.tsv','w')
for line in original:
	tokens = line.strip().lower().split("\t")
	label = int(tokens[3])
	oversampleddata.write(line)
	if label == 1:
		for _ in range(8):
			oversampleddata.write(line)

