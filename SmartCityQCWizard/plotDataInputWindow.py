import runSVM
import runQSVMpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

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

    yColumns = ['cod_weather']

    # allWindowSizeX = range(2,25,2)
    # allWindowSizeY = range(2,25,2)
    allWindowSizeX = range(2, 5, 2)
    allWindowSizeY = range(2, 5, 2)

    singleSplit = False
    returnMF1 = True
    kernel = "rbf"

    X, Y = np.meshgrid(allWindowSizeX, allWindowSizeY)
    Z = np.zeros_like(X)
    E = np.zeros_like(X)

    for i, windowSizeX in enumerate(allWindowSizeX):
        for j, windowSizeY in enumerate(allWindowSizeY):
            print(f"Processing {windowSizeX}, {windowSizeY}")
            mean,std = runSVM.run(datasetFile, xColumns, yColumns, windowSizeX, windowSizeY, singleSplit, returnMF1, kernel)
            Z[i,j] = mean
            E[i,j] = std

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title(f"Classical RBF kernel, Macro F1 score, 10-fold KV, variable window sizes")

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1.)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel("Input window size")
    ax.set_ylabel("Prediction window size")

    plt.savefig('plot_dataInputWindow.png', bbox_inches='tight')
    plt.show()


