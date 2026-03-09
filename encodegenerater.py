import cv2
import face_recognition  
import pickle
import os 
from pymongo import MongoClient


# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["project-FR"]
students_image = db["Studentimg"]
print("MongoDB connected successfully")

folderPath = 'images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds =[]
imagePaths = [] 

for path in pathList:
    fullPath = os.path.join(folderPath, path)

    img = cv2.imread(fullPath)
    if img is None:
        print(f"❌ Failed to load image: {fullPath}")
        continue

    imgList.append(img)
    studentIds.append(os.path.splitext(path)[0])
    imagePaths.append(fullPath) 

    
    # print(path)
    # print(os.path.splitext(path)[0])
print("Student IDs:", studentIds)


def findEncodings(imagesList):
    encodeList =[]
    for idx,img in enumerate(imagesList):
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        # encodeList.append(encode)
        if encode:
            encodeList.append(encode[0])
        else:
            print(f"⚠️ No face detected in image index {idx}")
            encodeList.append(None)
    return encodeList

print("Encoding Started......")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds =[encodeListKnown,studentIds]
# print(encodeListKnown)
print("Encoding Complete")

for student_id, encoding, image_path in zip(studentIds, encodeListKnown, imagePaths):
    if encoding is None:
        print(f"No face found for {student_id}, skipped")
        continue

    students_image.update_one(
        {"_id": student_id},
        {
            "$set": {
                "student_id": student_id,
                "image_path": image_path,
                "encoding": encoding.tolist() 
            }
        },
        upsert=True
    )
    
print("Encodings stored in MongoDB successfully")

encodeListKnownWithIds = [encodeListKnown, studentIds]
file = open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File saves")
