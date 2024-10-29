import runSVM
import runQSVMpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__ == "__main__":
    datasetFile = "final_ds.csv"

    allColumns = {
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

    xColumns = allColumns['All']

    df = pd.read_csv(datasetFile)[:24]




    plt.figure(figsize=(8,2.5))
    # plt.title(f"Classical RBF kernel, Macro F1 score, 10-fold KV, input {windowSizeX}h, output {windowSizeY}h")

    for column in xColumns:
        normVal = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
        plt.plot(normVal, label=column)

    # plt.legend()
    plt.ylabel('Norm value')
    plt.xlabel('h')
    # plt.xticks(wList)
    plt.ylim([0, 1])
    # plt.grid(axis='y')
    # plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.savefig('plot_data.png', bbox_inches='tight')
    plt.show(block=True)

