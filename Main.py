import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pyodbc

path='image_ai'  #folder where training images are present.
images=[]      # list of training image.
imgLabel=[]    # list of labels given to the images.
mylst=os.listdir(path)   #os.listdir() method is used to get the list of all files and directories in the specified directory.

for cl in mylst:
    curimg=cv2.imread(f'{path}\\{cl}')   #all the image read one by one.
    images.append(curimg)               #images append one by one in the images list.
    imgLabel.append(os.path.splitext(cl)[0])   #image labels append in imgLabel list.


def findEncodings(images):
    encodLst=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0] # this face_encoding function encodes the image present in the training data.
        encodLst.append(encode)
    return encodLst

encodlstKnowFaces=findEncodings(images)



def markAttendance2(name,inTime,InDate):
    conn = pyodbc.connect('Driver={SQL Server};'
                      'Server=LAPTOP-PS8KSBQB\MSSQLSERVER01;'
                      'Database=attendance;'
                      'Trusted_Connection=yes;')


    cursor = conn.cursor() # create connection between database and python.
        
    sql='''insert into attendance.dbo.tbl_attendance (Name,InDate,InTime) values(?, ?, ?)''' #query for inserting data into the the attendance table.

    val=(name,InDate,inTime)
    cursor.execute(sql,val) # for executing the query.
    conn.commit() # used to permanently save the changes done in the transaction in tables/databases.



webcam=cv2.VideoCapture(0)  #object of webcam to capture  the live image.
nm="a"  #it is a temporary variable for validating a attendance.

while True: # it is a infinite loop.
    success, img=webcam.read() #read the current image from webcam and store it in img variable.
    imgS=cv2.resize(img,(0,0),None,0.25,0.25) #change the size of image here it is 1/4th of image.
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB) #actual image is in BGR but we have to convert it into RGB.

    faceCurFrm= face_recognition.face_locations(imgS) # face current frame: here we use face location to find location of face it is from face recognition library. 
    encodeCurFrm=face_recognition.face_encodings(imgS,faceCurFrm) # encoding current frame : It is a face for us. But, for our algorithm, it is only an array of RGB values(Here we use numpy) — that matches a pattern that the it has learnt from the data samples we provided to it.Here it encode the current image.

    for encodFace, faceLocation in zip(encodeCurFrm,faceCurFrm):  # zip is used for mapping the encodeCurFrm and faceCurFrm.
        maches=face_recognition.compare_faces(encodlstKnowFaces,encodFace) #compare face from training set  and current face.
        faceDis=face_recognition.face_distance(encodlstKnowFaces,encodFace)  #for check distance between the faces.
        
        machesIndex=np.argmin(faceDis) #returns the index of the minimum element from a given array along the given axis.

        if maches[machesIndex]:
            name = imgLabel[machesIndex].upper() #store image label in name.
            # print(name)
            y1,x2,y2,x1=faceLocation       #take a coordinate of face from facelocation for drawing the box.
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4  #earlier we change the size so we need to modify it for making the box over the image.
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)  # to draw the rectangle (img , pointer one ,pointer two ,color , thickness).
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) # to fill the box.
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX ,1,(255,255,255),2)  #for writting a text on it .
            
            crTime=datetime.now().time() #current time.
            crDate=datetime.now().date() #current date.
            if name!=nm:
                markAttendance2(name,str(crTime),str(crDate))
                nm=name
    
    
    cv2.imshow('Frame',img)   # is used to display an image in a window.
    if cv2.waitKey(1) & 0xFF == ord('q'):  #to terminate the while loop we need to click q.
        break

webcam.release()  #for releasing the webcam. 
cv2.destroyAllWindows()  #simply destroys all the windows we created.