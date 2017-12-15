import csv

files = ['test.csv', 'train.csv', 'train0.csv', 'train1.csv', 'train2.csv', 'train3.csv', 'train4.csv']
for file in files:
    with open(file, 'r') as f, open(file + '.home.data', 'w') as h, open(file + '.away.data', 'w') as w:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            parts = line.split(',')
            home_result = 1
            if parts[len(parts)-2] == "False":
                home_result = 0
            away_result = 1
            if parts[len(parts)-1].strip() == "False":
                away_result = 0
            features = ""
            for index in range(0,100):
                result = 1
                if parts[index] < parts[index+100]:
                    result = 0
                features += " " + str(index+1) + ":" + str(result)
            h.write(str(home_result) + features + "\n")
            w.write(str(away_result) + features + "\n")
                
