import csv
import sys

# Variables
initPath = 'cDataSet/UNSW_2018_IoT_Botnet_Dataset_'
fileType = '.csv'
delimiter = ','
outputPath = 'grouped/'
outputName = 'IoT_Botnet_Dataset_$_Traffic.csv'
categoryPos = 33
subCategoryPos = categoryPos + 1


def load_csv(filename, col):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(filter(lambda x: x[col] == attack, list(reader)))


if len(sys.argv) != 3:
    raise ValueError('Please provide the attack name and if is in cat or subCat')

attack = sys.argv[1]
p2 = sys.argv[2]
if p2 not in ['cat', 'subCat']:
    raise ValueError('invalid value for cat')

p2 = p2 == 'cat'

print('Init loop')
for i in range(1, 75):
    fullOPath = outputPath + attack + '/' + outputName.replace('$', attack + '_' + str(i))
    with open(fullOPath, mode='w') as oFile:
        writer = csv.writer(oFile)  # mirror but with just one attack
        fileName = initPath + str(i) + fileType
        print('Loading file ... ' + fileName)
        data = load_csv(fileName, categoryPos if p2 else subCategoryPos)
        writer.writerows(data)
        print('data filtered and pushed to ' + fullOPath)

print('Files saved in: ', outputPath)
print('compiling full file')
with open(outputPath + outputName.replace('$', attack + '_FULL'), 'w') as fullFile:
    writer = csv.writer(fullFile)
    for i in range(1, 75):
        fullOPath = outputPath + attack + '/' + outputName.replace('$', attack + '_' + str(i))
        with open(fullOPath, 'r') as actFile:
            partialAttack = list(csv.reader(actFile))
            print('Found ' + str(len(partialAttack)) + ' records on file ' + fullOPath)
            writer.writerows(partialAttack)
            print('saved, moving to next file')

print('Attack ' + attack + ' compilation file saved')
