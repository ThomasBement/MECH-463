# ---------------------------------------- #
# DataAnalysis [Python File]
# Written By: Thomas Bement
# Created On: 2021-10-23
# ---------------------------------------- #

"""
IMPORTS
"""
import math

import numpy as np
import matplotlib.pyplot as plt

from csv import reader

"""
CONSTANTS
"""
L = (56*2)/100
W = (2.54)/100
I = (W*(L**3))/12
A = L*W
R = math.sqrt(I/A) 
g = 9.81

"""
READ IN DATA
"""
# Function to read specific CSV format
def read_csv(path):
    ans = {}
    headers = []
    with open(path, 'r') as read_file:
        csv_reader = reader(read_file)
        for i, row in enumerate(csv_reader):
            if (i == 1):
                for j in range(len(row)):
                    ans[row[j]] = []
                    headers.append(row[j])
            elif (i > 1):
                for j in range(len(row)):
                    ans[headers[j]].append(float(row[j]))
    return ans, path.split(' ')[0].split('\\')[-1]

# Generating file names to read
fil_list = []
for i in range(1, 8):
    fil_list.append(r'.\DATA\6.%i Export.csv' %i)

# Reading all data from all files into one dictonary
all_dat = {}
for fil in fil_list:
    ans, key = read_csv(fil)
    all_dat[key] = ans

# Converting lists into numpy arrays for statistics
for section in all_dat:
    for key in all_dat[section]:
        # Factor of 2 to convert from half frequency to full frequency, 1/X factor to convert to Hz
        all_dat[section][key] = 1/(2*np.array(all_dat[section][key]))

"""
SECTION 6.2
"""
section_lis = ['6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7']

plt_dat = {'D': [], 'Mean': [], 'STD_Err': []}
section = '6.2'
for test in all_dat[section]:
    plt_dat['D'].append(float(test.split('=')[-1]))
    plt_dat['Mean'].append(np.mean(all_dat[section][test]))
    plt_dat['STD_Err'].append(np.std(all_dat[section][test])/np.sqrt(np.shape(all_dat[section][test])[0]))

plt.errorbar(plt_dat['D'], plt_dat['Mean'], yerr=plt_dat['STD_Err'], marker='.')
plt.title('Section 6.2: Frequency vs. String Seperation')
plt.ylabel('Frequency of Oscillation [Hz]')
plt.xlabel('String Seperation D [cm]')
plt.savefig('Seperation.png', format='png', bbox_inches='tight')
plt.show()
plt.close()

"""
SECTION 6.3
"""
D = 50/100
L_string = 70.5/100
plt_dat = {'S/D': [], 'Mean': [], 'STD_Err': []}
section = '6.3'
for test in all_dat[section]:
    plt_dat['S/D'].append(float(test.split('=')[-1])/(D*100))
    plt_dat['Mean'].append(np.mean(all_dat[section][test]))
    plt_dat['STD_Err'].append(np.std(all_dat[section][test])/np.sqrt(np.shape(all_dat[section][test])[0]))

x_range = np.linspace(min(plt_dat['S/D']), max(plt_dat['S/D']), 256)
y_range = np.ones_like(x_range)
for i in range(len(y_range)):
    y_range[i] = (D/(2*3.141592*R))*math.sqrt(g/L_string)*math.sqrt((1/4)-(x_range[i])**2)
plt.errorbar(plt_dat['S/D'], plt_dat['Mean'], yerr=plt_dat['STD_Err'], marker='.', label='Measured')
plt.plot(x_range, y_range, label='Theoretical')
plt.legend()
plt.title('Section 6.3: Frequency vs. C.G. Offset')
plt.ylabel('Frequency of Oscillation [Hz]')
plt.xlabel('Normalized C.G. Offset s/D')
plt.savefig('CG_Off.png', format='png', bbox_inches='tight')
plt.show()
plt.close()

"""
SECTION 6.5
"""

# From MatLab
LR = [0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25,1.30]
f = [0.5696,0.5687,0.5674,0.5656,0.5629,0.5588,0.5528,0.5451,0.5361,0.5265,0.5169,0.5075,0.4984]


plt_dat = {'L2/L1': [], 'Mean': [], 'STD_Err': []}
section = '6.5'
for test in all_dat[section]:
    plt_dat['L2/L1'].append(float(test.split('=')[-1]))
    plt_dat['Mean'].append(np.mean(all_dat[section][test]))
    plt_dat['STD_Err'].append(np.std(all_dat[section][test])/np.sqrt(np.shape(all_dat[section][test])[0]))

plt.errorbar(plt_dat['L2/L1'], plt_dat['Mean'], yerr=plt_dat['STD_Err'], marker='.', label='Measured')
plt.plot(LR, f, label='Theoretical')
plt.legend()
plt.title('Section 6.5: Frequency vs. Length Ratio')
plt.ylabel('Frequency of Oscillation [Hz]')
plt.xlabel('String Length Ratio L2/L1')
plt.savefig('Len_Ratio.png', format='png', bbox_inches='tight')
plt.show()
plt.close()