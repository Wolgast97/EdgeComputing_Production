import matplotlib.pyplot as plt
import requests
import json
import numpy as np
from skimage.transform import resize

#Upload new Image
new_image = plt.imread('/Users/Dominik/Desktop/CNN-Server/image001.jpg')

#Plot image
#img = plt.imshow(new_image)

#Resize new image
resized_image = resize(new_image, (32,32,3))
#img = plt.imshow(resized_image)

#Reshape Image from (32,32,3) to (1,32,32,3)
resized_image = resized_image.reshape((1, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2]))

#Rename 
x_test = resized_image

#Set server URL
url = 'http://localhost:8501/v1/models/img_classifier_22_12:predict'

#Function for prediction
def make_prediction(instances):
   data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
   headers = {"content-type": "application/json"}
   json_response = requests.post(url, data=data, headers=headers)
   predictions = json.loads(json_response.text)
   return predictions

#Make prediction with new image
predictions = make_prediction(x_test)

#Sorted labels
classification = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
list_index = [0,1,2,3,4,5,6,7,8,9]

#Convert dict in list in array
x = predictions
x = x['predictions']
x = np.asarray(x)


for i in range(10):
  for j in range(10):
    if x[0][list_index[i]] > x[0][list_index[j]]:
      temp = list_index[i]
      list_index[i] = list_index[j]
      list_index[j] = temp

print(list_index)

for i in range(5):
  print(classification[list_index[i]], ':', round(x[0][list_index[i]] * 100, 2), '%')