import csv

# Variables
initPath = 'UNSW_2018_IoT_Botnet_Dataset_'
fileType = '.csv'
delimiter = ','
outputPath = 'IoT_Botnet_Dataset_Normal_Traffic.csv'
attackPos = 32


def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(filter(lambda x: x[attackPos] == '0', list(reader)))


with open(outputPath, mode='w') as oFile:
    writer = csv.writer(oFile)

    print('Init loop')
    for i in range(1, 75):
        fileName = initPath + str(i) + fileType
        print('Loading file ... ' + fileName)
        data = load_csv(fileName)
        len(data)
        writer.writerows(data)
        print('data filtered and pushed')

print('File saved in: ', outputPath)
