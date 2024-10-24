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

from autoencoder import *
from torch_ds import UnsupervisedDS

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

random_seed = 42
n_components = 6
windowSizeX = 12
windowSizeY = 24

try:
    print("Loading the dataset...")
    X_train, X_test, y_train, y_test = np.load(open(f"saved_data/windowDataset_{windowSizeX}_{random_seed}_autoencoder_{n_components}.npy", 'rb')).values()
except:
    print("Dataset not found. Creating the dataset...")
    torch.manual_seed(random_seed)

    df = pd.read_csv(datasetFile)

    df = df.loc[:, df.columns.intersection(set(xColumns + yColumns))]

    data = np.vstack([pd.concat([pd.concat([df.iloc[i-windowSizeX:i][k] for k in xColumns]), pd.concat([df.iloc[i:i+windowSizeY][k] for k in yColumns])]).reset_index(drop=True).to_numpy() for i in range(windowSizeX, len(df)-windowSizeY+1)])

    X = data[:,:-windowSizeY*len(yColumns)]
    y = data[:,-windowSizeY*len(yColumns):]

    # y = np.mean(y, axis=1)
    # y = y.astype(int).astype(str)
    # r = re.compile('[235].*')
    # vmatch = np.vectorize(lambda x: bool(r.match(x)))
    # y = vmatch(y)
    y = (y + np.ones_like(y)) / 2 
    y = np.logical_or.reduce([y[:,i] for i in range(y.shape[1])]).astype(int)
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    torch_X_train, torch_X_test = UnsupervisedDS(X_train), UnsupervisedDS(X_test)
    ae = AutoEncoder(in_size=X_train.shape[1], latent_size=n_components)
    print("Training the autoencoder...")
    train(ae, torch_X_train, epochs=100)

    fitted_X_train = predict(ae, torch_X_train).numpy()
    fitted_X_test = predict(ae, torch_X_test).numpy()
    # pca = PCA(n_components=n_components)
    # pca.fit(X_train)
    # print("Explained variance by the PCA:", sum(pca.explained_variance_ratio_))

    # X_train, X_test = pca.transform(X_train), pca.transform(X_test)

    np.savez(open(f"saved_data/windowDataset_{random_seed}_autoencoder.npy", 'wb'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


adhoc_feature_map = ZZFeatureMap(feature_dimension=n_components, reps=2, entanglement="linear")

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

model = QSVC(quantum_kernel=adhoc_kernel, verbose=True)

print("Fitting the model...")   
start = time()
model.fit(X_train, y_train)
fit_time = time() - start
# qsvc_score = qsvc.score(X_test, y_test)

# print(f"QSVC classification test score: {qsvc_score}")
print(f"Model fitted in {fit_time} seconds: Testing...")
p_train = model.predict(X_train)
p_test = model.predict(X_test)

print(f"Fit time: {fit_time}")
# print(mean_absolute_error(y_test, y_pred))

print("Train metrics:")
print(classification_report(y_train, p_train))

print("Test metrics:")
print(classification_report(y_test, p_test))


with open(f'trained_models/qsvc_{random_seed}_{n_components}_{windowSizeX}_{time:.2f}.pkl','wb') as f:
    pickle.dump(model, f)