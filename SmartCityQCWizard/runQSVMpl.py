import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report
import re
import pennylane as qml
from pennylane.templates import AngleEmbedding, AmplitudeEmbedding
import itertools
from multiprocessing import Pool
import os

# datasetFile = "particulate_matter/PM_from_01-08-T-00-00_to_10-08-T-21-0@1hr.csv"
# xColumns = ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
# yColumns = ['PM2.5_Sensor1']

# datasetFile = "weather/weather-from-2024-08-01 00:00:00-to-2024-10-08 21:10:00.csv"
# xColumns = ['hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg', 'clouds_all', 'ave_temp']
# yColumns = ['cod_weather']

datasetFile = "final_ds.csv"
xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg', 'clouds_all', 'max_temp', 'min_temp', 'ave_temp']
# xColumns = ['rain_1h']
# xColumns = ['clouds_all']
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12']
# xColumns = ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'ave_temp']
yColumns = ['cod_weather']


# windowSizeX = 48
windowSizeX = 2
windowSizeY = 24

splitSeed = 42

usePrecomputedKernel = True
parallelizeKernel = True

singleSplit = True

useAmplitude = True


df = pd.read_csv(datasetFile)

df = df.loc[:, df.columns.intersection(set(xColumns + yColumns))]

data = np.vstack([pd.concat([pd.concat([df.iloc[i-windowSizeX:i][k] for k in xColumns]), pd.concat([df.iloc[i:i+windowSizeY][k] for k in yColumns])]).reset_index(drop=True).to_numpy() for i in range(windowSizeX, len(df)-windowSizeY+1)])

X = data[:,:-windowSizeY*len(yColumns)]
y = data[:,-windowSizeY*len(yColumns):]

# y = np.mean(y, axis=1)
y = y.astype(int).astype(str)
r = re.compile('[235].*')
vmatch = np.vectorize(lambda x:bool(r.match(x)))
y = vmatch(y)
y = np.logical_or.reduce([y[:,i] for i in range(y.shape[1])]).astype(int)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=splitSeed, shuffle=True)

np.savez(open("windowDataset.npy", 'wb'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

if useAmplitude:
    n_qubits = np.ceil(np.log2(len(X_train[0]))).astype(int).item()

    dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

    projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    projector[0, 0] = 1


    @qml.qnode(dev_kernel)
    def kernel(x1, x2):
        """The quantum kernel."""
        AmplitudeEmbedding(x1, wires=range(n_qubits), pad_with=0., normalize=True)
        qml.adjoint(AmplitudeEmbedding)(x2, wires=range(n_qubits), pad_with=0., normalize=True)
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

else:
    n_qubits = len(X_train[0])

    dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

    projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    projector[0, 0] = 1

    @qml.qnode(dev_kernel)
    def kernel(x1, x2):
        """The quantum kernel."""
        AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

print(X_train.shape, X_test.shape)
exit()
if usePrecomputedKernel:
    precomputedKernelFilename = f"precomputedKernel_seed-{splitSeed}_amp-{useAmplitude}_columns-{len(xColumns)}_wSizeX-{windowSizeX}.npy"
    if os.path.isfile(precomputedKernelFilename):
        precomputedKernel = np.load(open(precomputedKernelFilename, 'rb'))
        kernel_train = precomputedKernel['kernel_train']
        kernel_test = precomputedKernel['kernel_test']

    else:
        if parallelizeKernel:
            # Train kernel
    
            kernel_train = np.zeros([len(X_train), len(X_train)])
    
            indexList = list(itertools.combinations(range(len(X_train)), 2))
            def fTrain(indices):
                i,j=indices
                return kernel(X_train[i], X_train[j])
    
            with Pool(None) as p:
                # resultsList = p.map(fTrain, indexList)
                resultsList = list(tqdm.tqdm(p.imap(fTrain, indexList), total=len(indexList)))
    
            for (i, j),r in zip(indexList, resultsList):
                kernel_train[i, j] = kernel_train[j, i] = r
    
            # Test kernel
            kernel_test = np.zeros([len(X_test), len(X_train)])
    
            indexList = list(itertools.product(range(len(X_test)), range(len(X_train))))
            def fTest(indices):
                i, j = indices
                return kernel(X_test[i], X_train[j])
    
            with Pool(None) as p:
                # resultsList = p.map(fTest, indexList)
                resultsList = list(tqdm.tqdm(p.imap(fTest, indexList), total=len(indexList)))
    
            for (i, j), r in zip(indexList, resultsList):
                kernel_test[i, j] = r
    
        else:
            kernel_train = np.zeros([len(X_train), len(X_train)])
            for i,j in tqdm.tqdm(list(itertools.combinations(range(len(X_train)), 2))):
                kernel_train[i,j] = kernel_train[j,i] = kernel(X_train[i], X_train[j])
    
            kernel_test = np.zeros([len(X_test), len(X_train)])
            for i,j in tqdm.tqdm(list(itertools.product(range(len(X_test)), range(len(X_train))))):
                kernel_test[i,j] = kernel(X_test[i], X_train[j])
    
        np.savez(open(precomputedKernelFilename, 'wb'), kernel_train=kernel_train, kernel_test=kernel_test)

    model = SVC(kernel='precomputed')

else:
    def kernel_matrix(A, B):
        """Compute the matrix whose entries are the kernel
           evaluated on pairwise data from sets A and B."""
        # return np.array([[kernel(a, b) for b in B] for a in A])
        myKernel = np.zeros([len(A), len(B)])
        if A is B:
            for i,j in tqdm.tqdm(list(itertools.combinations(range(len(A)), 2))):
                myKernel[i,j] = myKernel[j,i] = kernel(A[i], B[j])
        else:
            for i,j in tqdm.tqdm(list(itertools.product(range(len(A)), range(len(B))))):
                myKernel[i,j] = kernel(A[i], B[j])

        return myKernel


    model = SVC(kernel=kernel_matrix)

if singleSplit:
    if not usePrecomputedKernel:
        model.fit(X_train, y_train)

        p_train = model.predict(X_train)
        p_test = model.predict(X_test)
    else:
        model.fit(kernel_train, y_train)

        p_train = model.predict(kernel_train)
        p_test = model.predict(kernel_test)

    # print(mean_absolute_error(y_test, y_pred))

    print("Train metrics:")
    print(classification_report(y_train, p_train))

    print("Test metrics:")
    print(classification_report(y_test, p_test))

else:
    if not usePrecomputedKernel:
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True), scoring='f1_macro') # TODO: Work only with callable
        print(f"10-fold CV F1-macro: {scores.mean():0.2f} +- {scores.std():0.2f}")
    else:
        print("No")

