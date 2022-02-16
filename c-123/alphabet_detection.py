import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import os, ssl, time
from PIL import Image
import PIL.ImageOps

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('image.npz')['arr_0']
y = pd.read_csv('https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=7500, test_size=2500, random_state=26)
X_train_scaled = X_train/255
X_test_scaled = X_test/255
clf = LogisticRegression(solver='saga', multi_class= 'multinomial').fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
print(accuracy_score(y_pred, y_test))

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)
        roi = gray[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
        im_pil = Image.fromarray(roi)

        # convert to gray scale image
        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

        # Invert the image
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20

        # Coverting to scaler quantiy
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)

        # Using clip to limit the values between 0,255
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0 ,255)
        max_pixel = np.max(image_bw_resized_inverted)

        # Conveting into an array
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled) / max_pixel

        # Creating a test sample and making prediction
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print('Predicted class is: ',test_pred)
        # Showing frame and then releasing it
        cv2.imshow('Frame', gray)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break
    except Exception as e:
        pass   
cap.release()
cv2.destroyAllWindows()