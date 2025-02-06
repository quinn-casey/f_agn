import pandas as pd
import numpy as np
from find_f import calc
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

def kauf_line():
    x=np.linspace(-2, 0, 100)
    y = (0.61/(x - 0.05) + 1.3)
    return x, y

def kewl_line():
    x=np.linspace(-2, 0.4, 100)
    y = (0.61/(x - 0.47) + 1.19)
    return x, y


def main():

    x_points = np.linspace(-2, 0.5, 100)
    y_points = np.linspace(-1.5, 1.5, 100)

    X, Y = np.meshgrid(x_points, y_points)

    mean_values_025 = np.zeros_like(X)
    mean_values_05 = np.zeros_like(X)
    mean_values_075 = np.zeros_like(X)
    mean_values_1 = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i, j]
            y_val = Y[i, j]
            # calling the function
            result_025 = calc(x_val, y_val, grid_name="r025")
            # getting the mean value (can use f_median, f_std, or f_var)
            mean_values_025[i, j] = result_025['f_mean']

            result_05 = calc(x_val, y_val, grid_name="r05")
            mean_values_05[i, j] = result_05['f_mean']

            result_075 = calc(x_val, y_val, grid_name="r075")
            mean_values_075[i, j] = result_075['f_mean']

            result_1 = calc(x_val, y_val, grid_name="r1")
            mean_values_1[i, j] = result_1['f_mean']

    mean_values = [mean_values_025, mean_values_05, mean_values_075, mean_values_1]

    for i in mean_values:
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Kaufmann and Kewley lines
        plt.plot(kauf_line()[0], kauf_line()[1], color='r')
        plt.plot(kewl_line()[0], kewl_line()[1], color='r')

        # Scatter the points based on the condition
        plt.scatter(X, Y, c=i)
        plt.colorbar(label=r'Fractional AGN Contribution')
        #plt.title(str(i))
        plt.text(-1.5, -0.5, '0% AGN', color='w')
        plt.text(0.05, 1, '100% AGN', color='k')

        plt.xlim(-2, 0.5)
        plt.ylim(-1.25, 1.25)
        plt.xlabel(r'[NII] / H$\alpha$')
        plt.ylabel(r'[OIII] / H$\beta$')
        #plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
