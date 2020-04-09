import csv
import urllib.request
import numpy as np

def saveData():
    train_data = []
    # open file to read
    with open("data/data.csv", 'r') as csvfile:
        # iterate on all lines
        i = 0
        for line in csvfile:
            splitted_line = line.split(',')
            # check if we have an image URL
            if splitted_line[1] != '' and splitted_line[1] != "\n" and splitted_line[1] != "url":
                if (splitted_line[0] == '17037'):
                    # this url contains a ',' inside the url, hence need to take care of it
                    splitted_line[1] = splitted_line[1][1:] + ',' + splitted_line[2][:-1]
                    del splitted_line[2]
                # if (i >= 15264):
                urllib.request.urlretrieve(splitted_line[1], "train/" + str(i) + ".png")
                print("Image saved for {0}".format(splitted_line[0]))
                train_data.append(splitted_line)
                i += 1
            else:
                print("No result for {0}".format(splitted_line[0]))
        print(str(i) + " datapoints retrieved")

    train_data = np.array(train_data)
    np.save("data/train_data.npy", train_data)

if __name__ == "__main__":
    saveData()