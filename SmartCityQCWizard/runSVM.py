import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score,StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report, f1_score
import re

def run(datasetFile, xColumns, yColumns, windowSizeX, windowSizeY, singleSplit, returnMF1, kernel):
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42, shuffle=True)

    # np.savez(open("windowDataset.npy", 'wb'), X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    # model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    # model = SVC(kernel='linear')
    # model = SVC(kernel='poly', degree=4)
    # model = SVC(kernel='rbf')
    # model = SVC(kernel='sigmoid')

    if kernel.startswith("poly"):
        model = SVC(kernel="poly", degree=int(kernel[4:]))
    else:
        model = SVC(kernel=kernel)

    if singleSplit:
        model.fit(X_train, y_train)

        p_train = model.predict(X_train)
        p_test = model.predict(X_test)

        # print(mean_absolute_error(y_test, y_pred))

        if returnMF1:
            return f1_score(y_test, p_test, average='macro')
        else:
            print("Train metrics:")
            print(classification_report(y_train, p_train))

            print("Test metrics:")
            print(classification_report(y_test, p_test))

    else:
        scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=43), scoring='f1_macro')
        if returnMF1:
            return scores.mean(), scores.std()
        else:
            print(f"10-fold CV F1-macro: {scores.mean():0.2f} +- {scores.std():0.2f}")


if __name__ == "__main__":
    # datasetFile = "particulate_matter/PM_from_01-08-T-00-00_to_10-08-T-21-0@1hr.csv"
    # xColumns = ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
    # yColumns = ['PM2.5_Sensor1']

    # datasetFile = "weather/weather-from-2024-08-01 00:00:00-to-2024-10-08 21:10:00.csv"
    # xColumns = ['hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg', 'clouds_all', 'ave_temp']
    # yColumns = ['cod_weather']

    # datasetFile = "final_ds.csv"
    datasetFile = "final_ds_agg.csv"
    # xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5',
    #             'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10',
    #             'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
    #             'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
    #             'clouds_all', 'max_temp', 'min_temp', 'ave_temp']
    xColumns = ['AttendanceAreaTot', 'PM2.5_avg', 'PM10_avg', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
                'clouds_all', 'ave_temp']

    # xColumns = ['rain_1h']
    # xColumns = ['clouds_all']
    # xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
    # xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12']
    # xColumns = ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3']
    # xColumns = ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5', 'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10', 'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3', 'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'ave_temp']
    yColumns = ['cod_weather']

    # windowSizeX = 48
    windowSizeX = 1
    windowSizeY = 24

    singleSplit = True
    # singleSplit = False

    returnMF1 = False

    # kernel='linear'
    # kernel='poly2'
    # kernel='poly4'
    kernel='sigmoid'
    # kernel='rbf'

    run(datasetFile, xColumns, yColumns, windowSizeX, windowSizeY, singleSplit, returnMF1, kernel)
