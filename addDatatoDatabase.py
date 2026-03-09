from pymongo import MongoClient
import json

client = MongoClient("mongodb://localhost:27017/")

db = client["project-FR"]
students_col = db["students"]

print("MongoDB connected successfully")

with open("StudentData.json", "r") as f:
    data = json.load(f)

#Insert or update student data safely
for student_id, student_data in data.items():
    students_col.update_one(
        {"_id": student_id},
        {"$set": student_data},
        upsert=True
    )

print("Student data inserted successfully")
