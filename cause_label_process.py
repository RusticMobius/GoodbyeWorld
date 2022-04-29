
import csv

with open('data2/train_data.tsv', 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    output = []
    labels = []
    index = 0
    for r in reader:

        if r[1] == "label":
            continue
        if r[1] not in labels:
            labels.append(r[1])
            record = [index,r[1],"content"]
            output.append(record)
            index += 1
    with open("cause_label.tsv",'w') as f:
        writer = csv.writer(f,delimiter='\t')
        for r in output:
            writer.writerow(r)