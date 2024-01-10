import cv2
import numpy as np
import face_recognition

# Load Images and convert to RGB
img_elon = face_recognition.load_image_file("images/elon_musk.jpg")
img_elon = cv2.cvtColor(img_elon, cv2.COLOR_BGR2RGB)
img_test = face_recognition.load_image_file("images/bill gates.jpg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# Get shapes for changing size if image
img_test_h, img_test_w, img_chan_c = img_test.shape
img_height, img_width, img_chan = img_elon.shape

a = .8 # change size of image
dim = (int(img_width * a), int(img_height * a))
dim_test = (int(img_test_w * a), int(img_test_h * a))

# Resize Image
img_elon = cv2.resize(img_elon, dim)
img_test = cv2.resize(img_test, dim_test)

# Get face locations and encodings
face_loc = face_recognition.face_locations(img_elon)[0]
encode_elon = face_recognition.face_encodings(img_elon)[0]

# (0 = T,  1 = R, 2 = B, 3 = L)
point_1 = (face_loc[3], face_loc[0])
point_2 = (face_loc[1], face_loc[2])

bound_color = (255, 0, 255)
bound_thickness = 2
cv2.rectangle(img_elon, point_1, point_2, bound_color, bound_thickness)

face_loc_test = face_recognition.face_locations(img_test)[0]
encode_elon_test = face_recognition.face_encodings(img_test)[0]
point_1_test = (face_loc_test[3], face_loc[0])
point_2_test = (face_loc_test[1], face_loc[2])
cv2.rectangle(img_test, point_1_test, point_2_test, bound_color, bound_thickness)

results = face_recognition.compare_faces([encode_elon], encode_elon_test)[0]
face_distance = face_recognition.face_distance([encode_elon], encode_elon_test)[0]
cv2.putText(img_test, f"{results} {round(face_distance * 100, 2)}", (point_1_test[0], point_1_test[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)


cv2.imshow("Elon Test", img_test)
cv2.imshow("Elon Musk", img_elon)
cv2.waitKey(0)