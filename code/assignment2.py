import os
import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def get_correct_false(predictions,real):
    corrects=list()
    false=list()
    N=len(predictions)
    for i in range(N):
        if predictions[i]==real[i]:
            corrects.append(i) #get point and folder
        else:
            false.append(i)
    return corrects,false


def accuracy(predicted, real):
    n = len(predicted)
    counter = 0
    for i in range(n):
        if predicted[i] == real[i]:  # if the prediction is the same as the real then count
            counter += 1

    return counter / n


def get_tiny_image(image):
    image = cv2.resize(image, (16, 16))
    image = (image - np.mean(image)) / np.std(image)
    # Normalization work well because when we do this means can be near to zero but the variance will be the same

    image = image.flatten()  # to obtain 1d image array
    return image


def create_histograms(array, kmean_algo):  # histogram is the bag of visual words
    histogram_list = []
    for descriptor in array:
        if descriptor is not None:
            histogram = np.zeros(len(kmean_algo.cluster_centers_))
            cluster_result = kmean_algo.predict(descriptor)
            for i in cluster_result:
                histogram[i] += 1.0
            histogram_list.append(histogram)
    return histogram_list


def tiny_image_with_KNN_SVM(data_set):
    X_train, X_test, y_train, y_test, file_train, file_test = train_test_split([x[0] for x in data_set],
                                                                               [x[1] for x in data_set],
                                                                               [x[2] for x in data_set],
                                                                               test_size=0.2, random_state=42)
    # Linear SVM and tiny image
    clf = svm.LinearSVC(max_iter=10000, dual=False)  # In order to avoid convergence
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("My Accuracy of tiny image with Linear SVM {0:.2f}".format(accuracy(y_pred, y_test)))
    print("Accuracy of accuracy_score function", accuracy_score(y_pred, y_test))
    print("Classification report:\n", classification_report(y_pred, y_test))
    print("Confusion matrix:\n", confusion_matrix(y_pred, y_test))
    print("--------------------------------------------------")
    plot_confusion_matrix(clf,X_test,y_test)
    plt.title('Confusion matrix of the tiny image-KNN')

    # KNN and tiny image
    classifier = KNeighborsClassifier(n_neighbors=19)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("My Accuracy of tiny image with 19-NN {0:.2f}".format(accuracy(y_pred, y_test)))
    print("Accuracy of accuracy_score function", accuracy_score(y_pred, y_test))
    print("Classification report:\n", classification_report(y_pred, y_test))
    print("Confusion matrix:\n", confusion_matrix(y_pred, y_test))
    plot_confusion_matrix(classifier, X_test, y_test)
    plt.title('Confusion matrix of the tiny image-LinearSVM')

def boVW_with_KNN_SVM(data_set):
    X_train, X_test, y_train, y_test, file_train, file_test = train_test_split([x[0] for x in data_set],
                                                                               [x[1] for x in data_set],
                                                                               [x[2] for x in data_set],
                                                                               test_size=0.2, random_state=42)
    descriptor_list = list()
    for x in X_train:
        descriptor_list.extend(x)
    descriptor_list = np.array(descriptor_list)
    kmeans = KMeans(n_clusters=100, n_init=10)
    kmeans.fit(descriptor_list)

    train_histograms = np.array(create_histograms(X_train, kmeans))
    test_histograms = np.array(create_histograms(X_test, kmeans))

    """# pickles are created with the code above 
    #If you use the pickle code below code gives the results in 1 minute.
    #Before using the pickle comment the upper code lines espacially the K means. It takes too much time 
    #
    p_in = open("test_h", "rb")
    test_histograms = pickle.load(p_in)
    p_in = open("train_h", "rb")
    train_histograms = pickle.load(p_in)
    p_in = open("y_test", "rb")
    y_test = pickle.load(p_in)
    p_in = open("y_train", "rb")
    y_train = pickle.load(p_in)
    p_in = open("file_test", "rb")
    file_test = pickle.load(p_in)
    p_in = open("file_train", "rb")
    file_train = pickle.load(p_in)"""

    y_pred = list()
    k = 19
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_histograms, y_train)  # pickle.load(pickle_in5)
    for x in test_histograms:
        y_pred.append(classifier.predict([x]))
    print("My Accuracy of BoVW with 19-NN {0:.2f}".format(accuracy(y_pred, y_test)))
    print("Accuracy of accuracy_score function", accuracy_score(y_pred, y_test))
    print("Classification report:\n", classification_report(y_pred, y_test))
    print("Confusion matrix:\n", confusion_matrix(y_pred, y_test))
    print("--------------------------------------------------")
    plot_confusion_matrix(classifier,test_histograms,y_test)
    plt.title('Confusion matrix of the BoVW-KNN')

    y_pred = list()
    clf = svm.LinearSVC(max_iter=10000, dual=False)  # In order to avoid convergence (More on report)
    clf.fit(train_histograms, y_train)
    for x in test_histograms:
        y_pred.append(clf.predict([x]))
    print("My Accuracy of BoVW with Linear SVM {0:.2f}".format(accuracy(y_pred, y_test)))
    print("Accuracy of accuracy_score function", accuracy_score(y_pred, y_test))
    print("Classification report:\n", classification_report(y_pred, y_test))
    print("Confusion matrix\n", confusion_matrix(y_pred, y_test))
    plot_confusion_matrix(clf, test_histograms, y_test)
    plt.title('Confusion matrix of the BoVW-LinearSVM')
    plt.show()

    c,f=get_correct_false(y_pred,y_test)
    c2=list()
    f2=list()
    for i in c:
        c2.append([y_test[i],file_test[i]])
    for i in f:
        f2.append([y_test[i],file_test[i]])
    c2.sort(key=lambda a:a[0])
    f2.sort(key=lambda a:a[0])
    return c2,f2

scene_path = "C:\\Users\\AtakanAYYILDIZ\\PycharmProjects\\Vision2\\SceneDataset"
tiny_data_set = list()  # consist of data and its label(bedroom or highwar or ...)
bow_data_set = list()
sift = cv2.SIFT_create()

for root, directories, files in os.walk(scene_path):
    for folder in directories:
        folder_path = scene_path + "\\" + folder
        for filename in os.listdir(folder_path):
            img = cv2.imread(folder_path + "\\" + filename, 0)  # read as gray image
            if img is not None:
                tiny_data_set.append(
                    [get_tiny_image(img), folder, filename])  # out is the tiny data and folder is the label

                # bow
                kp, descriptor = sift.detectAndCompute(img, None)
                bow_data_set.append([descriptor, folder, filename])

print("                                   Tiny image Implementation                                                   ")
tiny_image_with_KNN_SVM(tiny_data_set)
print("---------------------------------------------------------------------------------------------------------------")
print("                                       Bow Implementation                                                      ")
correct_images,false_images=boVW_with_KNN_SVM(bow_data_set)




"""This is created just for the correct and false image
counter=0
l=["Bedroom","Highway","Kitchen","Mountain","Office"]
j=0
for i in false_images:
    if (j>5):
        break
    if counter<5 and i[0]==l[j]:
        img=cv2.imread(scene_path+"\\"+i[0]+"\\"+i[1])
        cv2.imshow(i[0]+"\\"+i[1],img)
        counter+=1
    elif (counter==5):
        j+=1
        counter=0
cv2.waitKey(0)"""
"""p=open("train_h","wb")
pickle.dump(train_histograms,p)
p=open("test_h","wb")
pickle.dump(test_histograms,p)
p=open("y_train","wb")
pickle.dump(y_train,p)
p=open("y_test","wb")
pickle.dump(y_test,p)
p=open("file_train","wb")
pickle.dump(file_train,p)
p=open("file_test","wb")
pickle.dump(file_test,p)"""

