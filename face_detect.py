import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

from fr_utils import *
from inception_blocks_v2 import *

from keras.models import load_model
model = load_model("frmodel.h5",compile=False)
#model.summary()

#function to convert image to encodings
def img_to_encodings(image_path,model):
  img = load_img(image_path,target_size=(96,96,3),color_mode='rgb')
  #img1 = cv2.imread(image_path, 1)#load the image
  plt.imshow(img)
  img = img_to_array(img)
  print(type(img))
  img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)#convert ot numpy array
  image = np.expand_dims(img,axis=0)
  #print(image.shape)#expand the batch dimension
  enc = model.predict_on_batch(image) #convert the img to encoding
  #print(enc)
  return enc  #return the encodings

def encodings(img,model):
  #img = load_img(image_path,target_size=(96,96,3),color_mode='rgb')
  #img1 = cv2.imread(image_path, 1)#load the image
  #plt.imshow(img)
  img = img_to_array(img)
  print(img.shape)
  img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)#convert ot numpy array
  image = np.expand_dims(img,axis=0)
  #print(image.shape)#expand the batch dimension
  enc = model.predict_on_batch(image) #convert the img to encoding
  #print(enc)
  return enc

database={}
#database['kasi'] = img_to_encodings(image_path="kasi_new.jpg",model=model)
database['Kasi_Perumal'] = img_to_encodings(image_path="Kasi_Crop.jpg",model=model)


#database['Hasmitha'] = img_to_encodings(image_path="Hasmi.jpg",model=model)
database['Donald_Trump'] = img_to_encodings(image_path="Trump.jpg",model=model)
database['Barack_Obama'] = img_to_encodings(image_path="obama.jpg",model=model)
#database['Anjan'] = img_to_encodings(image_path="Anjan.jpg",model=model)



#database['kasi2'] = img_to_encodings(image_path="Kasi_2.jpg",model=model)
#database['kasi3'] = img_to_encodings(image_path="Kasi_3.jpg",model=model)

#print(database['kasi'])

def verify_video(image_path, identity, database, model):

  encoding = encodings(image_path, model)

  # Step 2: Compute distance with identity's image (≈ 1 line)
  dist = np.linalg.norm(encoding - database[identity])
  print(dist)
  # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
  if dist < 0.7:
    #print("It's " + str(identity) + ", welcome in!")
    door_open = "It is kasi"
  else:
    #print("It's not " + str(identity) + ", please go away")
    door_open = 'It is not kasi'
  return dist, door_open


def verify(image_path, identity, database, model):


  ### START CODE HERE ###

  # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
  encoding = img_to_encodings(image_path, model)

  # Step 2: Compute distance with identity's image (≈ 1 line)
  dist = np.linalg.norm(encoding - database[identity])
  print(dist)
  # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
  if dist < 0.7:
    print("It's " + str(identity) + ", welcome in!")
    door_open = True
  else:
    print("It's not " + str(identity) + ", please go away")
    door_open = False
  return dist, door_open
  ### END CODE HERE ###


def who_is_it(image_path, database, model):

    ### START CODE HERE ###

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = encodings(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
        dist = np.linalg.norm(db_enc - encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    ### END CODE HERE ###

    if min_dist > 0.8:
        print("Not in the database.")
        identity = "Not in the database."
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

#verify(image_path='kasi_new.jpg',identity="Kasi_Perumal",database=database,model=model)


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = frame[y-30:y+h+30, x-30:x+w+30]

        resized_img = cv2.resize(img, dsize=(96,96), interpolation = cv2.INTER_AREA)
        cv2.imshow('Face',resized_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #print(resized_img)
        #image = Image.open(img)



        dist, door_open = who_is_it(image_path=resized_img, database=database, model=model)

        cv2.putText(frame, door_open, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        #print(img.shape)
        #cv2.resize


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyWindows()
# #
