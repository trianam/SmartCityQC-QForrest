import runSVM
import runQSVMpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    datasetFile = "final_ds.csv"

    xColumns = {
        "Attendance": ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5',
                'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10',
                'AttendanceArea11', 'AttendanceArea12'],
        "Pollutants": ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
                'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3'],
        "Weather": ['hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
                'clouds_all', 'max_temp', 'min_temp', 'ave_temp'],
        "Poll. + Att.": ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
                       'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5',
                       'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10',
                       'AttendanceArea11', 'AttendanceArea12'],
        "Weat. + Poll.": ['PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
                       'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
                    'clouds_all', 'max_temp', 'min_temp', 'ave_temp'],
        "All": ['AttendanceArea1', 'AttendanceArea2', 'AttendanceArea3', 'AttendanceArea4', 'AttendanceArea5',
                'AttendanceArea6', 'AttendanceArea7', 'AttendanceArea8', 'AttendanceArea9', 'AttendanceArea10',
                'AttendanceArea11', 'AttendanceArea12', 'PM2.5_Sensor1', 'PM2.5_Sensor2', 'PM2.5_Sensor3',
                'PM10_Sensor1', 'PM10_Sensor2', 'PM10_Sensor3', 'hum', 'pres', 'rain_1h', 'wind_speed', 'wind_deg',
                'clouds_all', 'max_temp', 'min_temp', 'ave_temp'],
    }

    keys = xColumns.keys()

    yColumns = ['cod_weather']
    windowSizeX = 48
    windowSizeY = 24
    singleSplit = False
    returnMF1 = True
    kernel = "rbf"


    means = []
    stds = []

    for k in keys:
        print(f"Processing {k}")
        mean,std = runSVM.run(datasetFile, xColumns[k], yColumns, windowSizeX, windowSizeY, singleSplit, returnMF1, kernel)
        means.append(mean)
        stds.append(std)

    plt.figure(figsize=(8,5))
    plt.title(f"Classical RBF kernel, Macro F1 score, 10-fold KV, input {windowSizeX}h, output {windowSizeY}h")

    plt.bar(keys, means, yerr=stds, align='center')
    # plt.legend()
    plt.ylabel('Macro F1 score')
    plt.xlabel('Features set')
    # plt.xticks(wList)
    plt.ylim([0.5, 1])
    plt.grid(axis='y')
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.savefig('plot_features.png', bbox_inches='tight')
    plt.show(block=True)

