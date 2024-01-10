import cv2
import face_recognition as fr
import numpy as np
import os
from datetime import datetime

path = "images"
images = []
class_names = []
my_list = os.listdir(path)

for cl in my_list:
    cur_img = cv2.imread(f"{path}/{cl}")
    images.append(cur_img)
    class_names.append(os.path.splitext(cl)[0])

def find_encoding(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode_img = fr.face_encodings(img)[0]
        encode_list.append(encode_img)
    return encode_list

def mark_attendance(name):
    with open("Attendance.csv", "r+") as f:
        data_list = f.readlines()
        name_list = []
        for line in data_list:
            entry = line.split(",")
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            data_str = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{data_str}")


encode_list_known = find_encoding(images)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # img = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    results_faces = fr.face_locations(frame)
    results_encode = fr.face_encodings(frame)

    for encode_face, face_location in zip(results_encode, results_faces):
        matches = fr.compare_faces(encode_list_known, encode_face)
        face_distance = fr.face_distance(encode_list_known, encode_face)
        match_index = np.argmin(face_distance)
        if matches[match_index]:
            name = class_names[match_index].upper()
            mark_attendance(name)
            y1, x2, y2, x1 = face_location
            if name == "KRISHA":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 153, 153), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 153, 153), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (222, 100, 56), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (222, 100, 56), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
    cv2.imshow("test", frame)
    cv2.waitKey(1)


# # Get images
# img_elon = fr.load_image_file("images/elon_musk.jpg")
# img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)
# elon_w, elon_h, elon_c = img_elon.shape
#
# img_test = fr.load_image_file("images/jack_ma.jpg")
# img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
# test_w, test_h, test_c = img_test.shape
#
# # Resize Images
# sz = .5
# img_elon = cv2.resize(img_elon, (int(elon_h), int(elon_w)))
# img_test = cv2.resize(img_test, (int(test_h * sz), int(test_w * sz)))

