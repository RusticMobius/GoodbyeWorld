import csv
f1 = open("data_segment", 'r', encoding="utf-8")
f2 = open("segment_train", 'w', encoding="utf-8")
f3 = open("segment_test", 'w', encoding="utf-8")

reader1 = csv.reader(f1, delimiter=' ')
writer2 = csv.writer(f2, delimiter=' ')
writer3 = csv.writer(f3, delimiter=' ')

data_list = []
for item in reader1:
    data_list.append(item)

for data in data_list[:2100]:
    writer2.writerow(data)

for data in data_list[2100:]:
    writer3.writerow(data)