import pickle
from time import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report
# import re
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
import tqdm

import itertools
import pennylane as qml
from pennylane.templates import AngleEmbedding, AmplitudeEmbedding
from multiprocessing import Pool
from autoencoder import *
from torch_ds import UnsupervisedDS
from random import seed, randint

from sklearn.decomposition import PCA
# datasetFile = "particulate_matter/PM_from_01-08-T-00-00_to_10-08-T-21-0@1hr.csv"
# xColumns = ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
# yColumns = ['PM2.5_Sensor1']

# datasetFile = "weather/weather-from-2024-08-01 00:00:00-to-2024-10-08 21:10:00.csv"
# xColumns = ['hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg', 'clouds_all', 'ave_temp']
# yColumns = ['cod_weather']

datasetFile = "labels_final_ds.csv"
xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg', 'clouds_all', 'max_temp', 'min_temp', 'ave_temp']
# xColumns = ['rain_1h']
# xColumns = ['clouds_all']
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12']
# xColumns = ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'ave_temp']
yColumns = ['cod_weather']

random_seed = [randint(0, 1000000) for _ in range(1)]
n_components = [2**q for q in range(5, 6)]
windowSizeX = [12]# , 24, 48]
windowSizeY = [i for i in range(2, 24, 2)]

combinations = list(itertools.product(random_seed, n_components, windowSizeX, windowSizeY))
for i, (rseed, ncomp, wsX, wsY) in enumerate(combinations):
    try:
        print(f"Running for random seed {rseed}, n_components {ncomp}, windowSizeX {wsX}, windowSizeY {wsY}, experiment {i+1}/{len(combinations)}")
        try:
            print("Loading the dataset...")
            X_train, X_test, y_train, y_test = np.load(open(f"saved_data/windowDataset_{wsX}_{wsY}_{rseed}_autoencoder_{ncomp}.npy", 'rb')).values()
        except:
            print("Dataset not found. Creating the dataset...")
            torch.manual_seed(rseed)

            df = pd.read_csv(datasetFile)

            df = df.loc[:, df.columns.intersection(set(xColumns + yColumns))]

            data = np.vstack([pd.concat([pd.concat([df.iloc[i-wsX:i][k] for k in xColumns]), pd.concat([df.iloc[i:i+wsY][k] for k in yColumns])]).reset_index(drop=True).to_numpy() for i in range(wsX, len(df)-wsY+1)])

            X = data[:,:-wsY*len(yColumns)]
            y = data[:,-wsY*len(yColumns):]

            # y = np.mean(y, axis=1)
            # y = y.astype(int).astype(str)
            # r = re.compile('[235].*')
            # vmatch = np.vectorize(lambda x: bool(r.match(x)))
            # y = vmatch(y)
            y = (y + np.ones_like(y)) / 2 
            y = np.logical_or.reduce([y[:,i] for i in range(y.shape[1])]).astype(int)
            scaler = StandardScaler()

            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rseed)
            torch_X_train, torch_X_test = UnsupervisedDS(X_train), UnsupervisedDS(X_test)
            ae = AutoEncoder(in_size=X_train.shape[1], latent_size=ncomp)
            print("Training the autoencoder...")
            train(ae, torch_X_train, epochs=100)

            fitted_X_train = predict(ae, torch_X_train).numpy()
            fitted_X_test = predict(ae, torch_X_test).numpy()
            # pca = PCA(n_components=n_components)
            # pca.fit(X_train)
            # print("Explained variance by the PCA:", sum(pca.explained_variance_ratio_))

            # X_train, X_test = pca.transform(X_train), pca.transform(X_test)

            X_train, X_test = fitted_X_train, fitted_X_test
            np.savez(open(f"saved_data/windowDataset_{wsX}_{wsY}_{rseed}_autoencoder_{ncomp}.npy", 'wb'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        try:
            print("Loading the precomputed kernel...")
            kernel_train, kernel_test = np.load(open(f"kernels/precomputedKernel_{rseed}_{ncomp}_{wsX}_{wsY}.npy", 'rb')).values()
        except:
            print("Precomputed kernel not found. Computing the kernel...")
            # n_qubits = len(X_train[0])
            n_qubits = np.ceil(np.log2(len(X_train[0]))).astype(int).item()

            dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

            projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
            projector[0, 0] = 1

            @qml.qnode(dev_kernel)
            def kernel(x1, x2):
                """The quantum kernel."""
                AmplitudeEmbedding(x1, wires=range(n_qubits), pad_with=0, normalize=True)
                qml.adjoint(AmplitudeEmbedding)(x2, wires=range(n_qubits), pad_with=0, normalize=True)
                return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


            parallelizeKernel = True

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

            np.savez(open(f"kernels/precomputedKernel_{rseed}_{ncomp}_{wsX}_{wsY}.npy", 'wb'), kernel_train=kernel_train, kernel_test=kernel_test)

        model = SVC(kernel='precomputed')

        print("Fitting the model...")   
        model.fit(kernel_train, y_train)
        # qsvc_score = qsvc.score(X_test, y_test)

        # print(f"QSVC classification test score: {qsvc_score}")
        p_train = model.predict(kernel_train)
        p_test = model.predict(kernel_test)

        # print(mean_absolute_error(y_test, y_pred))

        print("Train metrics:")
        print(classification_report(y_train, p_train))

        print("Test metrics:")
        print(classification_report(y_test, p_test))


        # with open(f'trained_models/qsvc_{rseed}_{ncomp}_{wsX}_{time:.2f}.pkl','wb') as f:
        #     pickle.dump(model, f)
    except Exception as e:
        print(f"Error: {e} @ combination {combinations[i]}")
        continue