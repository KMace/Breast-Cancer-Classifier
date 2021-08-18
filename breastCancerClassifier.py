import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Cleaning the data
def classifier(value):
    if int(value) == 2:
        return 0
    elif int(value) == 4:
        return 1
    else:
        return False

data = pd.read_csv("formattedData.csv")
lengthDuplicates = len(data[data.duplicated(subset=['idNumber'])])

data = data.drop_duplicates(subset=['idNumber']).reset_index()
data = data.drop(columns=['index', 'idNumber'])

data['diagnosis'] = data['diagnosis'].apply(classifier)

# Assigning test and training data
# 80% training & 20% testing --> 516 training & 129 testing
trainDiagnosis = np.array([np.array(data[i:i+1]).flatten()[0] for i in range(516)])
trainData = np.array([np.array(data[i:i+1]).flatten()[1:] for i in range(516)])

testDiagnosis = np.array([np.array(data[i:i+1]).flatten()[0] for i in range(516, 645)])
testData = np.array([np.array(data[i:i+1]).flatten()[1:] for i in range(516, 645)])


# Creating the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(9,)),
    keras.layers.Dense(81, activation = "relu"),
    keras.layers.Dense(2, activation = "softmax")
])

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model.fit(trainData, trainDiagnosis, epochs=5)
print()
testLoss, testAcc = model.evaluate(testData, testDiagnosis)
print("Overall Model Accuracy:", testAcc)

#Saving the model
model.save("breastCancerNeuralNetwork.h5")

# Using the model to predict the entire set and ouput then some of those predictions
classes = ["Benign", "Malignant"]
predictions = model.predict(trainData)

for i in range(6):
    #print(predictions[i])
    prediction = np.argmax(predictions[i])

    if str(prediction) == str(trainDiagnosis[i]):
        print("Correct prediction!")
    else:
        print("Incorrect prediction")
    
    print("Prediction: " + classes[prediction] + " with a confidence of: " + str(round(max(predictions[i]) * 100, 4)) + "%")
    print()
