"""
dlib은 python으로 만들어진 페키지가 아니다
https://stackoverflow.com/questions/54719496/how-to-install-dlib-for-python-on-mac
여기를 참고하자
"""
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow._api.v2.compat.v1 as tf
# import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../model/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('../img/02.jpg')
# plt.figure(figsize=(8,5))
# plt.imshow(img)
# plt.show()

img_result = img.copy()
dets = detector(img)
if len(dets) == 0:
    print('no detection')
else:
    fig, ax = plt.subplots(1, figsize=(8, 5))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=2,
                                 edgecolor='g',
                                 facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()

fig, ax = plt.subplots(1, figsize=(8, 5))
objs = dlib.full_object_detections()
for detection in dets:
    s = sp(img, detection)
    objs.append(s)
    for point in s.parts():
        circle = patches.Circle((point.x, point.y),
                                radius=3,
                                edgecolor='r',
                                facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)
plt.show()

faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
fig, axes = plt.subplots(1, len(faces)+1, figsize=(10, 8))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)
plt.show()



# def align_faces(img):
#     dets = detector(img, 1)
#     objs = dlib.full_object_detections()
#     for detection in dets:
#         s = sp(img, detection)
#         objs.append(s)
#     faces = dlib.