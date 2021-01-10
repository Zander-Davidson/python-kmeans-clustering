import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import numpy as np
import sys
from datetime import datetime


#--------------------------------------------------------------------------------
# functions
#--------------------------------------------------------------------------------

# params:
#   x: list of x-coordinates of data
#   y: list of y-coordinates of data
#   Cx: list of x-coordinates of centroids 
#   Cy: list of y-coordinates of centroids
# returns:
#   list of indices corresponding to Cx and Cy
def nearestCentroids(x, y, Cx, Cy):
    # nearest_c[i] = j is the index of the closest centroid (Cx[j], Cy[j]) to the
    # ith coordinate (x[i], y[i]) in the data
    nearest_c = [None] * len(x)

    # for each data coordinate (x[i], y[i]), find nearest centroid
    for i in range(len(x)):
        d = np.array((x[i], y[i]))  # data vector

        nearest_c_index = -1
        min_distance = sys.maxsize

        # for each centroid (Cx[j], Cy[j]), calculate the distance
        # to (x[i], y[i]), storing the centroid with the minimum distance
        for j in range(len(Cx)):
            cj = np.array((Cx[j], Cy[j]))  # centroid vector

            # calculate Euclidean distance between 2 vectors
            distance = np.linalg.norm(d - cj) 

            if distance < min_distance:
                nearest_c_index = j
                min_distance = distance
        
        # Cx/Cy indices are stored as strings to be used as dictionary keys
        nearest_c[i] = nearest_c_index

    return nearest_c

# params:
#   x: list of x-coordinates of data
#   y: list of y-coordinates of data
#   nearest_c: nearest_c[i] is an index j for which (Cx[j], Cy[j]) 
#              is the nearest centroid to (x[i], y[i])
#   K: number of clusters
#   Cx: list of x-coordinates of centroids 
#   Cy: list of y-coordinates of centroids
# returns: {
#   'Cx': list of x-coordinates of next centroids,
#   'Cy': list of y-coordinates of next centroids
# }
def nextCentroids(x, y, nearest_c, K, Cx, Cy):
    CxTmp = []
    CyTmp = []

    # for each centroid, gather the data points of its cluster, and 
    # calculate its next location
    for i in range(K):
        xp = []
        yp = []

        # gather x and y coordinates for the cluster of centroid i
        for j in range(len(nearest_c)):
            if nearest_c[j] == i:
                xp.append(x[j])
                yp.append(y[j])

        # case when centroid i has no data in its cluster. make no change
        if (len(xp) == 0):
            CxTmp.append(Cx[i])
            CyTmp.append(Cy[i])
        # case when centroid i has at least one datum in its cluster.
        # the next centroid i is the midpoint of all data in its cluster.
        else:
            CxTmp.append(np.mean(xp))
            CyTmp.append(np.mean(yp))
    
    return {'Cx': CxTmp, 'Cy': CyTmp}

# function to unpack data from rows of a .data file (modified from Assignment 4)
def unpackData(attributes, data):
    numAttrs = len(attributes)
    numData = len(data)
    dataDict = {}

    # initialize the data dictionary with attribute keys and empty list values
    for attr in attributes:
        dataDict[attr] = []

    # populate the data dictionary attribute lists
    for i in range(numData):
        obj = data[i].split(",")
            
        if len(obj) == numAttrs:
            # add values for the current data object
            for j in range(numAttrs):
                attrName = attributes[j]
                attrVal = obj[j]

                if attrName == 'class' :
                    dataDict[attrName].append(attrVal)
                else:
                    dataDict[attrName].append(float(attrVal))

        elif (len(obj) == 1) & (obj[0] == ''):
            print("Warning: Ignored empty data row") 
        else:
            print("Warning: Ignored incomplete data object")
            print(obj)

    return dataDict


#--------------------------------------------------------------------------------
# main program
#--------------------------------------------------------------------------------

# specific to iris data
data_file_path = "data/iris.data"
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# key: centroid index, value: display color. # of entries = # of classes in the data
color_map = {-1: 'black', 0: 'red', 1: 'royalblue', 2: 'green'}

# key: data class, value: display shape
symbol_map = {'centroid': 'x', 'Iris-setosa': 'circle', 'Iris-versicolor': 'square', 'Iris-virginica': 'diamond'}

try:
    # open the file in read ("r") mode
    data_file = open(data_file_path, "r")
except:
    print("File not found or incorrect path.")

# each element is an entire row with comma-delimited attributes
raw_data = data_file.read().split('\n')
data = unpackData(attributes, raw_data)

# initialize K-Means clustering for Iris data

iterations = 7
K = len(class_names)

x = data['petal_length']
random.seed(datetime.now())
Cx = [random.uniform(1, 7) for i in range(K)]  # randomly-initialized Cx
# Cx = [1.0, 3.76, 6.9]  # min, mean, max values of petal_length
# Cx = [1.5, 4.0, 7.0]   # semi-arbitrary guess upon viewing the data

y = data['petal_width']
random.seed(datetime.now())
Cy = [random.uniform(0, 3) for i in range(K)]  # randomly-initialized Cy
# Cy = [0.1, 1.20, 2.5]  # min, mean, max values of petal_width
# Cy = [1.5, 1.5, 1.5]   # semi-arbitrary guess upon viewing the data

# nearest_centroids[i] is an index j for which (Cx[j], Cy[j]) is the nearest centroid to (x[i], y[i])
nearest_centroids = [-1 for i in range(len(x))]
centroids = range(K)  # centroids[i] = i (each centroid is nearest to itself)

# size of data points on the scatter plots
size = [8 for i in range(len(x))] + [25]*K

fig = make_subplots(
    rows=iterations+1, cols = 2,
    subplot_titles=[("Iteration " + str(i)) for i in range(iterations + 1)]
)

# perform K-Means clustering
for i in range(iterations+1):
    row = 1+int(i/2)  # subplot row
    col=1+(i%2)       # subplot column

    # decode colors to make the plots look pretty
    color = []
    symbol = []
    for j in range(len(nearest_centroids)):
        color.append(color_map[nearest_centroids[j]])
        symbol.append(symbol_map[data['class'][j]])
    for j in range(K):
        color.append(color_map[centroids[j]])
        symbol.append(symbol_map['centroid'])

    # add the subplot to the plotly figure
    fig.append_trace(go.Scatter(
        x=x+Cx, 
        y=y+Cy,
        mode='markers',
        marker_symbol=symbol,
        marker=dict(
            size=size, 
            color=color, 
            # line=dict(
            #     color=color,
            #     width=1
            # )
        ),
    ), row=row, col=col)

    fig.update_xaxes(title_text="Petal Length (cm)", row=row, col=col)
    fig.update_yaxes(title_text="Petal Width (cm)", row=row, col=col)

    fig.add_annotation(text="\u2716: Centroid<br>\u25C6: Iris-Virginica<br>\u25A0: Iris-Versicolor<br>\u25CF: Iris-Setosa<br>", showarrow=False, row=row, col=col)

    # reassign the data to their nearest centroids
    nearest_centroids = nearestCentroids(x, y, Cx, Cy)

    # calculate the next centroids
    C = nextCentroids(x, y, nearest_centroids, K, Cx, Cy)
    Cx = C['Cx']
    Cy = C['Cy']
    
# display the subplots
fig.update_layout(
    height=700*(iterations+1), width=1300, 
    title_text="K-Means Clustering for Iris Dataset<br>(Random Initial Centroids)", showlegend=False
)
fig.show()