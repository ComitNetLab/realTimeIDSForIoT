import csv

# Variables
initPath = 'E:/Estudios/Maestría/2022-10/Tesis1Local/bot-iot/Entire Dataset/UNSW_2018_IoT_Botnet_Dataset_'
fileType = '.csv'
catPos = 33
subCatPos = 34
DDos = 0
DoS = 0
OSfingerprinting = 0
serviceScanning = 0
keylogging = 0
dataExfiltration = 0


print('Init loop')
for i in range(1, 75):
    fileName = initPath + str(i) + fileType
    print('Loading file ... ' + fileName)
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        DoS += len(list(filter(lambda x: x[catPos] == 'DoS', reader)))
        DDos += len(list(filter(lambda x: x[catPos] == 'DDoS', reader)))
        keylogging += len(list(filter(lambda x: x[subCatPos] == 'Keylogging', reader)))
        OSfingerprinting += len(list(filter(lambda x: x[subCatPos] == 'OS_Fingerprint', reader)))
        serviceScanning += len(list(filter(lambda x: x[subCatPos] == 'Service_Scan', reader)))
        dataExfiltration += len(list(filter(lambda x: x[subCatPos] == 'Data_Exfiltration', reader)))

print('Results:')
print(str(DDos))
print(str(DoS))
print(str(keylogging))
print(str(OSfingerprinting))
print(str(serviceScanning))
print(str(dataExfiltration))
