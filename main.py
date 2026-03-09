import pickle
import cv2
import os
import face_recognition
import numpy as np
import cvzone
from pymongo import MongoClient
from datetime import datetime


# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["project-FR"]
students_col = db["students"]
image_col = db["Studentimg"]


cap = cv2.VideoCapture(0)
imgBackground = cv2.imread("resources/background.jpeg")

folderModePath = 'resources/modes'
modePathList = os.listdir(folderModePath)
imgModeList = []

for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))


# camera setting
CAM_WIDTH = 605
CAM_HEIGHT = 430
X_OFFSET = 53
Y_OFFSET = 167

# modes setting
MODE_WIDTH = 464
MODE_HEIGHT = 590
MODE_X = 748
MODE_Y = 60


# load the encoding file
print("Loading Encoded File....")
file = open("EncodeFile.p", "rb")
encodeListKnownWithIds = pickle.load(file)
file.close()

encodeListKnown, studentIds = encodeListKnownWithIds
print(studentIds)
print("Encode File Loaded")


modeType = 0
counter = 0
id = -1
imgStudent = []
attendanceMarked = False


while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    img = cv2.resize(img, (CAM_WIDTH, CAM_HEIGHT))
    imgBackground[
        Y_OFFSET:Y_OFFSET + CAM_HEIGHT,
        X_OFFSET:X_OFFSET + CAM_WIDTH
    ] = img

    imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
    imgBackground[
        MODE_Y:MODE_Y + MODE_HEIGHT,
        MODE_X:MODE_X + MODE_WIDTH
    ] = imgMode

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                id = studentIds[matchIndex]

                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                    cv2.imshow("Face Attendance", imgBackground)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

        if counter != 0:
            if counter == 1:
                studentInfo = students_col.find_one({"_id": id})
                image_profile = image_col.find_one({"_id": id})

                imgStudent = cv2.imread(image_profile["image_path"])
                imgStudent = cv2.resize(imgStudent, (240, 192))

                datetimeObject = datetime.strptime(studentInfo['last_attandance_time'],"%Y-%m-%d %H:%M:%S")
                secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                print(secondsElapsed)

                if secondsElapsed > 30:
                    if not attendanceMarked:
                        students_col.update_one(
                            {"_id": id},
                            {
                                "$inc": {"total_attandance": 1},
                                "$set": {
                                    "last_attandance_time": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                }
                            }
                        )
                        attendanceMarked = True
                else:
                    modeType = 3
                    counter = 0

                    imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
                    imgBackground[ MODE_Y:MODE_Y + MODE_HEIGHT, MODE_X:MODE_X + MODE_WIDTH] = imgMode

            if modeType != 3:
                if 10 < counter < 20:
                    modeType = 2

                imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
                imgBackground[ MODE_Y:MODE_Y + MODE_HEIGHT,MODE_X:MODE_X + MODE_WIDTH ] = imgMode

                if counter <= 10:
                    cv2.putText( imgBackground,str(studentInfo.get('total_attandance', 0)),
                        (810, 135),cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255),1)

                    cv2.putText(imgBackground,studentInfo.get('name', ''),
                        (1006, 540), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255),1 )

                    cv2.putText(imgBackground,str(id),
                        (1000, 448), cv2.FONT_HERSHEY_COMPLEX,0.5,(255, 255, 255), 1)

                    cv2.putText( imgBackground,studentInfo.get('major', ''),
                        (1000, 498),cv2.FONT_HERSHEY_COMPLEX,0.5,(255, 255, 255),1)

                    cv2.putText(imgBackground,studentInfo.get('standing', ''),
                        (880, 578),cv2.FONT_HERSHEY_COMPLEX,0.6,(100, 100, 100),1)

                    cv2.putText(imgBackground,str(studentInfo.get('year', '')),
                        (990, 578),cv2.FONT_HERSHEY_COMPLEX,0.6,(100, 100, 100),1)

                    cv2.putText(imgBackground,str(studentInfo.get('starting_year', '')),
                        (1105, 578),cv2.FONT_HERSHEY_COMPLEX,0.6,(100, 100, 100),1)

                    (w, h), _ = cv2.getTextSize(studentInfo['name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
                    offset = (414 - w) // 2
                    cv2.putText( imgBackground, str(studentInfo['name']),
                        (775 + offset, 410),cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1 )
                    
                    imgBackground[175:175 + 192, 859:859 + 240] = imgStudent

                counter += 1

                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []

                    imgMode = cv2.resize(imgModeList[modeType],(MODE_WIDTH, MODE_HEIGHT))
                    imgBackground[MODE_Y:MODE_Y + MODE_HEIGHT,MODE_X:MODE_X + MODE_WIDTH] = imgMode
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()





# ======================================
# import pickle
# import cv2
# import os
# import face_recognition  
# import numpy as np
# import cvzone
# from pymongo import MongoClient
# import numpy as np
# from datetime import datetime

# # MongoDB connection
# # conString = "mongodb://localhost:27017/"
# client = MongoClient("mongodb://localhost:27017/")

# db = client["project-FR"]
# students_col = db["students"]
# image_col = db["Studentimg"]

# # bucket = storage.bucket()


# cap = cv2.VideoCapture(0)
# imgBackground = cv2.imread("resources/background.jpeg")
# folderModePath = 'resources/modes'
# modePathList = os.listdir(folderModePath)
# imgModeList = []

# for path in modePathList:
#     imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

# # print(imgModeList)


# # camera setting
# CAM_WIDTH = 605
# CAM_HEIGHT = 430
# X_OFFSET = 53
# Y_OFFSET = 167

# # modes setting

# MODE_WIDTH = 464
# MODE_HEIGHT = 590
# MODE_X = 748
# MODE_Y = 60


# #load the encoding file 
# print("Loading Encoded File....")
# file = open("EncodeFile.p" ,"rb")
# encodeListKnownWithIds = pickle.load(file)
# file.close()
# encodeListKnown,studentIds = encodeListKnownWithIds
# print(studentIds)
# print("Encode File Loaded")


# modeType = 0
# counter = 0
# id = -1
# imgStudent = []
# attendanceMarked = False


# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     imgS = cv2.resize(img,(0,0),None,0.25,0.25)
#     imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

#     faceCurFrame = face_recognition.face_locations(imgS)
#     encodeCurFrame =face_recognition.face_encodings(imgS,faceCurFrame)



#     img = cv2.resize(img, (CAM_WIDTH, CAM_HEIGHT))
#     imgBackground[Y_OFFSET:Y_OFFSET + CAM_HEIGHT,
#                   X_OFFSET:X_OFFSET + CAM_WIDTH] = img
    
   
#     imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
#     imgBackground[MODE_Y:MODE_Y + MODE_HEIGHT,
#                   MODE_X:MODE_X + MODE_WIDTH] = imgMode

#     if faceCurFrame:
#         for encodeFace ,faceLoc in zip(encodeCurFrame, faceCurFrame):
#             matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#             faceDis = face_recognition.face_distance(encodeListKnown , encodeFace)
#             # print("matches" , matches)
#             # print("faceDis", faceDis)

#             matchIndex = np.argmin(faceDis)
#             # print("Match Index", matchIndex)

#             if matches[matchIndex]:
#                 # print("Known Face detected")
#                 # print(studentIds[matchIndex])
#                 y1,x2,y2,x1=faceLoc
#                 y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
#                 bbox = 55+x1, 162+y1, x2-x1 , y2-y1
#                 imgBackground =  cvzone.cornerRect(imgBackground,bbox,rt=0)
#                 id = studentIds[matchIndex]

#                 if counter == 0:
#                     cvzone.putTextRect(imgBackground,"Loading",(275,400))
#                     cv2.imshow("Face Attendance",imgBackground)
#                     cv2.waitKey(1)
#                     counter = 1
#                     modeType = 1

#         if counter!= 0:
            
#             if counter == 1:
#                 #get the data 
#                 studentInfo =  students_col.find_one({"_id": id})
#                 #get the image
#                 blob = image_col.find_one({"_id":id})
#                 imgStudent = cv2.imread(blob["image_path"])
#                 imgStudent = cv2.resize(imgStudent,(240,192))
                
#                 #  Attendance Update (ONCE PER APP START)
#                 datetimeObject = datetime.strptime(studentInfo['last_attandance_time'],
#                                                 "%Y-%m-%d %H:%M:%S")
#                 secondsElapsed = (datetime.now()-datetimeObject).total_seconds()
#                 # print(secondsElapsed)
#                 if secondsElapsed > 30:
#                     if not attendanceMarked:
#                         students_col.update_one(
#                             {"_id": id},
#                             {"$inc": {"total_attandance": 1},
#                             "$set": { "last_attandance_time":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
#                             }
#                             )
#                         attendanceMarked = True
#                 else:
#                     modeType =3
#                     counter = 0
#                     imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
#                     imgBackground[MODE_Y:MODE_Y + MODE_HEIGHT,MODE_X:MODE_X + MODE_WIDTH] = imgMode
#             if modeType != 3:
                    
#                 if 10<counter<20:
#                     modeType = 2 
#                 imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
#                 imgBackground[MODE_Y:MODE_Y + MODE_HEIGHT,MODE_X:MODE_X + MODE_WIDTH] = imgMode


#                 if counter <=10:
#                     cv2.putText( imgBackground,str(studentInfo.get('total_attandance', 0)),
#                         (810, 135),cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255),1)

#                     cv2.putText(imgBackground,studentInfo.get('name', ''),
#                         (1006, 540), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255),1 )

#                     cv2.putText(imgBackground,str(id),
#                         (1000, 448), cv2.FONT_HERSHEY_COMPLEX,0.5,(255, 255, 255), 1)

#                     cv2.putText( imgBackground,studentInfo.get('major', ''),
#                         (1000, 498),cv2.FONT_HERSHEY_COMPLEX,0.5,(255, 255, 255),1)

#                     cv2.putText(imgBackground,studentInfo.get('standing', ''),
#                         (880, 578),cv2.FONT_HERSHEY_COMPLEX,0.6,(100, 100, 100),1)

#                     cv2.putText(imgBackground,str(studentInfo.get('year', '')),
#                         (990, 578),cv2.FONT_HERSHEY_COMPLEX,0.6,(100, 100, 100),1)

#                     cv2.putText(imgBackground,str(studentInfo.get('starting_year', '')),
#                         (1105, 578),cv2.FONT_HERSHEY_COMPLEX,0.6,(100, 100, 100),1)

#                     (w, h), _ = cv2.getTextSize(studentInfo['name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
#                     offset = (414 - w) // 2
#                     cv2.putText( imgBackground, str(studentInfo['name']),
#                         (775 + offset, 410),cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1 )
                    
#                     imgBackground[175:175 + 192, 859:859 + 240] = imgStudent

#                 counter+=1


#                 if counter>=20:
#                     counter = 0
#                     modeType = 0
#                     studentInfo = []
#                     imgStudent = []
#                     imgMode = cv2.resize(imgModeList[modeType], (MODE_WIDTH, MODE_HEIGHT))
#                     imgBackground[MODE_Y:MODE_Y + MODE_HEIGHT,MODE_X:MODE_X + MODE_WIDTH] = imgMode
#     else:
#         modeType = 0
#         couter = 0




 
# # _id:"25256"


#     # cv2.imshow("Webcam", img)
#     cv2.imshow("Face Attendance", imgBackground)

#     if cv2.waitKey(1) & 0xFF == 27:  
#         break



# cap.release()
# cv2.destroyAllWindows()
