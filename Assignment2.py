# Import relevant libraries
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

# Set the working directory path to the location with the data
path = '...'

##############################################################################
# Open using pandas
animals = pd.read_csv(path + "animals", sep = " ", header = None)
countries = pd.read_csv(path + "countries", sep = " ", header = None)
fruits = pd.read_csv(path + "fruits", sep = " ", header = None)
veggies = pd.read_csv(path + "veggies", sep = " ", header = None)

# Add the cluster category in a new column
animals['Category'] = 'animals'
countries['Category'] = 'countries'
fruits['Category'] = 'fruits'
veggies['Category'] = 'veggies'

# Join all data together
data = pd.concat([animals, countries, fruits, veggies], ignore_index = True)

# Change all class labels to numbers starting from 0
labels = (pd.factorize(data.Category)[0]+1) - 1 # 0=animals, 1=countries, 2=fruits, 3=veggies
x = data.drop([0, 'Category'], axis = 1).values

# Save the maximum index for each category for the P/R/F
maxAni = data.index[data['Category'] == 'animals'][-1]
maxCount = data.index[data['Category'] == 'countries'][-1]
maxFruit = data.index[data['Category'] == 'fruits'][-1]
maxVeg = data.index[data['Category'] == 'veggies'][-1]

##############################################################################
"""
kmeans_clustering allows the user to calculate the k-means clustering of 
    their data. x refers to the data they wish to cluster, and k is the number 
    of clusters the user wishes to specify.

Formatting:
    x - numpy.darray
    k - integer
    distance_measure - string (Euclidean, Manhattan or Cosine)
    trueOrFalseNorm - boolean
    trueorFalsePRF - boolean

Returns:
    If trueOrFalseNorm = False, it will print the number of clusters, the 
    number of updates required the final clusters and centroid locations.
    If trueOrFalseNorm = True, it will also prints the Precision (P), Recall 
    (R) and F-Score (F). The function also returns the P, R and F.
"""
def kmeans_clustering(x, k, distance_measure, trueOrFalseNorm, trueOrFalsePRF):
 
    # Normalise the data using L2
    if trueOrFalseNorm == True:
        x = x / np.linalg.norm(x)
    
    # Randomly initialise the first centroids
    centroids = []
    temp = np.random.randint(x.shape[0], size = k)
    while (len(temp) > len(set(temp))):
        temp = np.random.randint(x.shape[0], size = k)
    for i in temp:
        centroids.append(x[i])
    # Create copies of the centroids for updating
    centroids_old = np.zeros(np.shape(centroids))
    centroids_new = deepcopy(centroids)

    # Create a blank distance and cluster assignment object to hold results
    clusters = np.zeros(x.shape[0])
    # Create an error object
    error = np.linalg.norm(centroids_new - centroids_old)
    num_errors = 0

    # Whilst there is an error value:
    while error != 0:
        #print(error)
        dist = np.zeros([x.shape[0], k])
        # Add one to the number of errors
        num_errors += 1
        # Calculate the Euclidean distance from each point to each centroid    
        if distance_measure == "Euclidean":
            for j in range(len(centroids)):
                dist[:, j] = np.linalg.norm(x - centroids_new[j], axis=1)
        # Calculate the Manhattan distance from each point to each centroid   
        elif distance_measure == "Manhattan":
            for j in range(len(centroids)):
                dist[:, j] = np.sum(np.abs(x - centroids_new[j]), axis=1)
        # Calculate the Cosine Similarity distance from each point to each 
        # centroid          
        elif distance_measure == "Cosine":
             for j in range(len(centroids)):
                dist[:, j] = 1 - (np.dot(x, centroids_new[j]) / 
                                  (np.linalg.norm(x, axis = 1) * 
                                   np.linalg.norm(centroids_new[j])))       

        # Calculate the cluster assignment
        clusters = np.argmin(dist, axis = 1)

        # Assign the new copy of centroids to the old centroids object
        centroids_old = deepcopy(centroids_new)

        # Calculate the mean to re-adjust the cluster centroids
        for m in range(k):
            centroids_new[m] = np.mean(x[clusters == m], axis = 0)

        # Re-calculate the error
        error = np.linalg.norm(np.array(centroids_new) - np.array(centroids_old))

    #Assign the final clusters and centroids to new objects
    predicted_clusters = clusters
    final_centroids = np.array(centroids_new)

    # If the user does not want P, R and F to be calculated, then display
    # the results of the k-means clustering
    if trueOrFalsePRF == False:
        print("\nFinal Results of K-Means Clustering with", distance_measure, 
              "Distance Measurement\n")
        print("\tNumber of Clusters:", k)
        print("\tNumber of Updates:", num_errors)
        print ("\tFinalised Clusters:\n", predicted_clusters)
        print ("\tFinalised Centroid Locations:\n", final_centroids)
        print("-----------------------------------------------------------------------------")
    
    # If the user wants P, R and F to be calculated then continue
    else:
        # Create objects of the index positioning of the different classes
        animal_pos = predicted_clusters[:maxAni+1]
        countries_pos = predicted_clusters[maxAni+1:maxCount+1]
        fruit_pos = predicted_clusters[maxCount+1:maxFruit+1]
        veggies_pos = predicted_clusters[maxFruit+1:maxVeg+1]
        
        # Create objects for contingency calculations
        # True Positives
        TP = 0
        # False Negatives
        FN = 0
        # True Negatives
        TN = 0
        # False Positives
        FP = 0
        #--------------------------------------------------------
        # For every row in animal_pos
        for i in range(len(animal_pos)):
            # For every row in animal_pos
            for j in range(len(animal_pos)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TP
                    if(animal_pos[i] == animal_pos[j]):
                        TP += 1
                    # Otherwise add 1 to FN
                    else:
                        FN += 1
            #For every row in countries_pos                
            for j in range(len(countries_pos)):
                # If i is equal to j then add 1 to FP
                if(animal_pos[i] == countries_pos[j]):
                    FP += 1
                # Otherwise add 1 to TN
                else:
                    TN += 1
            # For every row in fruit_pos
            for j in range(len(fruit_pos)):
                # If i is equal to j then add 1 to FP
                if(animal_pos[i]==fruit_pos[j]):
                    FP += 1
                # Otherwise add 1 to TN
                else:
                    TN += 1
            # For every row in veggies_pos
            for j in range(len(veggies_pos)):
                # If i is equal to j then add 1 to FP
                if(animal_pos[i] == veggies_pos[j]):
                    FP += 1
                # Otherwise add 1 to TN
                else:
                    TN += 1
        #--------------------------------------------------------    
        #For every row in countries_pos 
        for i in range(len(countries_pos)):
            #For every row in countries_pos 
            for j in range(len(countries_pos)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TP
                    if(countries_pos[i] == countries_pos[j]):
                        TP += 1
                    # Otherwise add 1 to FN
                    else:
                        FN += 1
            # For every row in fruit_pos
            for j in range(len(fruit_pos)):
                # If i is equal to j then add 1 to FP
                if(countries_pos[i] == fruit_pos[j]):
                    FP += 1
                # Otherwise add 1 to TN
                else:
                    TN += 1
            # For every row in veggies_pos
            for j in range(len(veggies_pos)):
                # If i is equal to j then add 1 to FP
                if(countries_pos[i] == veggies_pos[j]):
                    FP += 1
                # Otherwise add 1 to TN
                else:
                    TN += 1     
        #--------------------------------------------------------
        # For every row in fruit_pos
        for i in range(len(fruit_pos)):
            # For every row in fruit_pos
            for j in range(len(fruit_pos)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TP
                    if(fruit_pos[i] == fruit_pos[j]):
                        TP += 1
                    # Otherwise add 1 to FN
                    else:
                        FN += 1
            # For every row in veggies_pos
            for j in range(len(veggies_pos)):
                # If i is equal to j then add 1 to FP
                if(fruit_pos[i] == veggies_pos[j]):
                    FP += 1
                # Otherwise add 1 to TN
                else:
                    TN += 1    
        #--------------------------------------------------------
        # For every row in veggies_pos
        for i in range(len(veggies_pos)):       
            # For every row in veggies_pos
            for j in range(len(veggies_pos)):
                # If i and j are not the same, and j > i
                if (i != j & j>i):
                    # If i is equal to j then add 1 to TP
                    if(veggies_pos[i] == veggies_pos[j]):
                        TP += 1
                    # Otherwise add 1 to FN
                    else:
                        FN += 1       
        # Calculate the Precision (P), Recall (R), and F-Score (F) and round
        # to 2 decimal places
        P = round((TP / (TP + FP)), 2)
        R = round((TP / (TP + FN)), 2)
        F = round((2 * (P * R) / (P + R)), 2)
        
        # If the data was normalised, then print the distance measurement and
        # normalisation
        if trueOrFalseNorm == True:
            print("\nFinal Results of K-Means Clustering with", distance_measure, 
              "Distance Measurement and L2 Normalisation")
        # Otherwise just print the distance measurement
        else: 
            print("\nFinal Results of K-Means Clustering with", distance_measure, 
              "Distance Measurement")
        # Print the results
        print("\tNumber of Clusters:", k)
        print("\tNumber of Updates:", num_errors)
        print("\tP:", P, ", R:", R, ", F:", F)
        
        # Return the P, R and F values for plotting4
        return P, R, F
        
##############################################################################
"""
plotting allows the user to plot the results of the Precision (P), 
    Recall (R) and F-Scores (F) acquired from the K-Means clustering.

Formatting:
    k - list
    P - list
    R - list
    F - list
    distance_measure - string
    l2 - string

Returns:
    A line graph comparing the P, R and F across the number of clusters (k)
    
"""
def plotting(k, P, R, F, distance_measure, l2):
    # Plot K against P
    plt.plot(K_list, P_list, label="Precision")
    # Plot K against R
    plt.plot(K_list, R_list, label="Recall")
    # Plot K against F
    plt.plot(K_list, F_list, label="F-Score")
    # Plot the title
    plt.title("K-Means Clustering with " + distance_measure + l2, loc="left")
    # Plot the x and y axis labels
    plt.xlabel('Number of Clusters')
    plt.ylabel("Score")
    # Display the legend
    plt.legend()
    # Display the plot
    plt.show()    

##############################################################################
# Question 1
kmeans_clustering(x, 4, "Euclidean", False, False)

#Questions 2-6
for question in range(2,7):
    #Create an empty list for P, R, F and K
    P_list = []
    R_list = []
    F_list = []
    K_list = []
    # Create an empty string for the distance method
    distance_measure = ""
    
    # Question 2
    if question == 2:
        distance_measure = "Euclidean"
        normalisation = False
    # Question 3
    elif question == 3:
        distance_measure = "Euclidean"
        normalisation = True
    # Question 4
    elif question == 4:
        distance_measure = "Manhattan"
        normalisation = False
    # Question 5
    elif question == 5:
        distance_measure = "Manhattan"
        normalisation = True
    # Question 6
    else:
        distance_measure = "Cosine"
        normalisation = False
    
    # Define k between 1 - 10
    for k in range(1,11):
        # Append k to a list for plotting
        K_list.append(k)
        # Save the Precision, Recall and F-Scores
        P,R,F = kmeans_clustering(x, k, distance_measure, normalisation, True)
        # Append the Precision, Recall and F-Score to each list for plotting
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)
    # If the data is normalised, edit the title to include 'and Normalisation'    
    if normalisation:
        plotting(K_list, P_list, R_list, F_list, distance_measure, " and Normalisation")
    # If not normalised, then do not include additional title
    else:
        plotting(K_list, P_list, R_list, F_list, distance_measure, "")