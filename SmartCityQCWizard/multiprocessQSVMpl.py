import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report, f1_score
import re
import pennylane as qml
from pennylane.templates import AngleEmbedding, AmplitudeEmbedding
import numpy as np
import itertools
from multiprocessing import Pool
# from multiprocess import Pool
import os

# datasetFile = "final_ds.csv"
datasetFile = "final_ds_agg.csv"
# xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5',
#             'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10',
#             'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
#             'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
#             'clouds_all', 'max_temp', 'min_temp', 'ave_temp']
xColumns = ['AttendanceAreaTot', 'PM2.5_avg', 'PM10_avg', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg', 'clouds_all', 'ave_temp']
yColumns = ['cod_weather']

windowSizeX = 1
# windowSizeX = 1
# windowSizeY = 1

splitSeed = 42

usePrecomputedKernel = True
parallelizeKernel = True

singleSplit = True

# circuit = "amplitude"
# circuit = "ampHad"
# circuit = "angle"
circuit = "angle2nd"

returnMF1 = False

predictionsWindows = [1] + [2 * i for i in range(1, 13)] #[1,2,4,6,8,10,12,14,16,18,20,22,24,3,5,7,9,11,13,15,17,19,21,23]
# seeds = [528, 491, 25, 73]
seeds = [528, 491]
# seeds = [4, 7, 12, 17]
circuits = ["angle", "angle2nd", "amplitude"]
# circuits = ["amplitude"]

hyperparameters = list(itertools.product(predictionsWindows, seeds, circuits))

# for windowSizeY in predictionsWindows:
# for windowSizeY in [24]:
for run_number, (windowSizeY, splitSeed, circuit) in enumerate(hyperparameters):
    print("======================================================")
    print(f"Creating for window {windowSizeY}, seed {splitSeed}, circuit {circuit}, run {run_number+1}/{len(hyperparameters)}")

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
    # np.savez(open("windowDataset.npy", 'wb'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    if circuit == "amplitude":
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
    elif circuit == "ampHad":
        n_qubits = np.ceil(np.log2(len(X_train[0]))).astype(int).item()

        dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

        projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
        projector[0, 0] = 1

        @qml.qnode(dev_kernel)
        def kernel(x1, x2):
            x1 = x1 / np.linalg.norm(x1)
            x2 = x2 / np.linalg.norm(x2)

            x1 = np.pad(x1, (int(np.floor(((2**n_qubits)-len(x1)) /2)), int(np.ceil(((2**n_qubits)-len(x1)) /2))))
            x2 = np.pad(x2, (int(np.floor(((2**n_qubits)-len(x2)) /2)), int(np.ceil(((2**n_qubits)-len(x2)) /2))))

            """The quantum kernel."""
            for q in range(n_qubits):
                qml.Hadamard(q)
            qml.MottonenStatePreparation(x1, wires=range(n_qubits))
            for q in range(n_qubits):
                qml.Hadamard(q)
            qml.MottonenStatePreparation(x1, wires=range(n_qubits))

            qml.adjoint(qml.MottonenStatePreparation)(x2, wires=range(n_qubits))
            for q in range(n_qubits):
                qml.Hadamard(q)
            qml.adjoint(qml.MottonenStatePreparation)(x2, wires=range(n_qubits))
            for q in range(n_qubits):
                qml.Hadamard(q)

            return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))


    elif circuit == "angle":
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
    elif circuit == "angle2nd":
        n_qubits = len(X_train[0])

        dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

        projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
        projector[0, 0] = 1


        # def kernel_angle_embedding_second_order(x1, x2, rot: str = 'X'):
        @qml.qnode(dev_kernel)
        def kernel(x1, x2, rot: str = 'X'):
            if rot not in {'X', 'Y', 'Z'}:
                raise ValueError(f"Invalid rotation gate {rot}")
            AngleEmbedding(x1, wires=range(n_qubits), rotation=rot) # forst order term
            # theta = np.sin(np.pi - x1) * np.sin(np.pi - x2) # custom feature map with bounded rotation
            
            # second order term is R_ij with theta sin(pi -x1) sin(pi-xj)
            theta = np.sin(np.pi - x1) * np.sin(np.pi - x2)
            for i in range(n_qubits):
                qml.MultiRZ(theta[i], wires=[i, (i+1) % n_qubits])
            return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))
            



    else:
        raise ValueError(f"Circuit {circuit} not valid")

    if usePrecomputedKernel:
        precomputedKernelFilename = f"kernels/precomputedKernel_seed-{splitSeed}_circ-{circuit}_columns-{len(xColumns)}_wSizeX-{windowSizeX}_wSizeY-{windowSizeY}.npy"
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

                # classic linear kernel for test
                # kernel_train = X_train @ X_train.T
                # kernel_test = X_test @ X_train.T

            else:
                kernel_train = np.zeros([len(X_train), len(X_train)])
                for i,j in tqdm.tqdm(list(itertools.combinations(range(len(X_train)), 2))):
                    kernel_train[i,j] = kernel_train[j,i] = kernel(X_train[i], X_train[j])

                kernel_test = np.zeros([len(X_test), len(X_train)])
                for i,j in tqdm.tqdm(list(itertools.product(range(len(X_test)), range(len(X_train))))):
                    kernel_test[i,j] = kernel(X_test[i], X_train[j])

            np.savez(open(precomputedKernelFilename, 'wb'), kernel_train=kernel_train, kernel_test=kernel_test)

        model = SVC(kernel='precomputed')

    # else:
    #     def kernel_matrix(A, B):
    #         """Compute the matrix whose entries are the kernel
    #            evaluated on pairwise data from sets A and B."""
    #         # return np.array([[kernel(a, b) for b in B] for a in A])
    #         myKernel = np.zeros([len(A), len(B)])
    #         if A is B:
    #             for i,j in tqdm.tqdm(list(itertools.combinations(range(len(A)), 2))):
    #                 myKernel[i,j] = myKernel[j,i] = kernel(A[i], B[j])
    #         else:
    #             for i,j in tqdm.tqdm(list(itertools.product(range(len(A)), range(len(B))))):
    #                 myKernel[i,j] = kernel(A[i], B[j])

    #         return myKernel


    #     model = SVC(kernel=kernel_matrix)

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

        if returnMF1:
            print("No")
            # return f1_score(y_test, p_test, average='macro')
        else:
            train_report = classification_report(y_train, p_train) 
            # print("Train metrics:")
            # print(train_report)
            train_report_dict = classification_report(y_train, p_train, output_dict=True)

            test_report = classification_report(y_test, p_test)
            # print("Test metrics:")
            # print(train_report)
            test_report_dict = classification_report(y_test, p_test, output_dict=True)

            if not os.path.isfile("results.csv"):
                df_result = pd.DataFrame(columns=['seed', 'circuit', 'len_ds', 'Tx', 'Ty',
                                'split', 
                                '0_precision', '0_recall', '0_f1-score', '0_support',
                                '1_precision', '1_recall', '1_f1-score', '1_support',
                                'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1-score', 'macro_avg_support',
                                'weighted_avg_precision', 'weighted_avg_recall', 'weighted_avg_f1-score', 'weighted_avg_support'])

            else:
                df_result = pd.read_csv("results.csv")
                # df = df.append({"windowSizeY": windowSizeY, "train_report": train_report_dict, "test_report": test_report_dict}, ignore_index=True)
            # df_result = df_result.append({"seed": splitSeed, "circuit": circuit, 'len_ds': len(xColumns), 'Tx': windowSizeX, 'Ty': windowSizeY,
            #                 'split': 'train', 
            #                 '0_precision': train_report_dict['0']['precision'], '0_recall': train_report_dict['0']['recall'], '0_f1-score': train_report_dict['0']['f1-score'], '0_support': train_report_dict['0']['support'],
            #                 '1_precision': train_report_dict['1']['precision'], '1_recall': train_report_dict['1']['recall'], '1_f1-score': train_report_dict['1']['f1-score'], '1_support': train_report_dict['1']['support'],
            #                 'macro_avg_precision': train_report_dict['macro avg']['precision'], 'macro_avg_recall': train_report_dict['macro avg']['recall'], 'macro_avg_f1-score': train_report_dict['macro avg']['f1-score'], 'macro_avg_support': train_report_dict['macro avg']['support'],
            #                 'weighted_avg_precision': train_report_dict['weighted avg']['precision'], 'weighted_avg_recall': train_report_dict['weighted avg']['recall'], 'weighted_avg_f1-score': train_report_dict['weighted avg']['f1-score'], 'weighted_avg_support': train_report_dict['weighted avg']['support']
            #                 }, ignore_index=True)
            train_df = pd.DataFrame([{"seed": splitSeed, "circuit": circuit, 'len_ds': len(xColumns), 'Tx': windowSizeX, 'Ty': windowSizeY,
                            'split': 'train', 
                            '0_precision': train_report_dict['0']['precision'], '0_recall': train_report_dict['0']['recall'], '0_f1-score': train_report_dict['0']['f1-score'], '0_support': train_report_dict['0']['support'],
                            '1_precision': train_report_dict['1']['precision'], '1_recall': train_report_dict['1']['recall'], '1_f1-score': train_report_dict['1']['f1-score'], '1_support': train_report_dict['1']['support'],
                            'macro_avg_precision': train_report_dict['macro avg']['precision'], 'macro_avg_recall': train_report_dict['macro avg']['recall'], 'macro_avg_f1-score': train_report_dict['macro avg']['f1-score'], 'macro_avg_support': train_report_dict['macro avg']['support'],
                            'weighted_avg_precision': train_report_dict['weighted avg']['precision'], 'weighted_avg_recall': train_report_dict['weighted avg']['recall'], 'weighted_avg_f1-score': train_report_dict['weighted avg']['f1-score'], 'weighted_avg_support': train_report_dict['weighted avg']['support']
                            }], index=[0])  
            df_result = pd.concat([df_result, train_df], ignore_index=True)
            test_df = pd.DataFrame({"seed": splitSeed, "circuit": circuit, 'len_ds': len(xColumns), 'Tx': windowSizeX, 'Ty': windowSizeY,
                            'split': 'test', 
                            '0_precision': test_report_dict['0']['precision'], '0_recall': test_report_dict['0']['recall'], '0_f1-score': test_report_dict['0']['f1-score'], '0_support': test_report_dict['0']['support'],
                            '1_precision': test_report_dict['1']['precision'], '1_recall': test_report_dict['1']['recall'], '1_f1-score': test_report_dict['1']['f1-score'], '1_support': test_report_dict['1']['support'],
                            'macro_avg_precision': test_report_dict['macro avg']['precision'], 'macro_avg_recall': test_report_dict['macro avg']['recall'], 'macro_avg_f1-score': test_report_dict['macro avg']['f1-score'], 'macro_avg_support': test_report_dict['macro avg']['support'],
                            'weighted_avg_precision': test_report_dict['weighted avg']['precision'], 'weighted_avg_recall': test_report_dict['weighted avg']['recall'], 'weighted_avg_f1-score': test_report_dict['weighted avg']['f1-score'], 'weighted_avg_support': test_report_dict['weighted avg']['support']
            }, index=[0])
            # df_result.append({"seed": splitSeed, "circuit": circuit, 'len_ds': len(xColumns), 'Tx': windowSizeX, 'Ty': windowSizeY,
            #                 'split': 'test', 
            #                 '0_precision': test_report_dict['0']['precision'], '0_recall': test_report_dict['0']['recall'], '0_f1-score': test_report_dict['0']['f1-score'], '0_support': test_report_dict['0']['support'],
            #                 '1_precision': test_report_dict['1']['precision'], '1_recall': test_report_dict['1']['recall'], '1_f1-score': test_report_dict['1']['f1-score'], '1_support': test_report_dict['1']['support'],
            #                 'macro_avg_precision': test_report_dict['macro avg']['precision'], 'macro_avg_recall': test_report_dict['macro avg']['recall'], 'macro_avg_f1-score': test_report_dict['macro avg']['f1-score'], 'macro_avg_support': test_report_dict['macro avg']['support'],
            #                 'weighted_avg_precision': test_report_dict['weighted avg']['precision'], 'weighted_avg_recall': test_report_dict['weighted avg']['recall'], 'weighted_avg_f1-score': test_report_dict['weighted avg']['f1-score'], 'weighted_avg_support': test_report_dict['weighted avg']['support']
            #                 }, ignore_index=True)
            df_result = pd.concat([df_result, test_df], ignore_index=True)
            df_result.to_csv("results.csv", index=False)
            print(test_report_dict.keys())

    # else:
    #     if not usePrecomputedKernel:
    #         scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True), scoring='f1_macro') # TODO: Work only with callable
    #         print(f"10-fold CV F1-macro: {scores.mean():0.2f} +- {scores.std():0.2f}")
    #     else:
    #         print("No")


