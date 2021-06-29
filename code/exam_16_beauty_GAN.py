"""
dlib은 python으로 만들어진 페키지가 아니다
https://stackoverflow.com/questions/54719496/how-to-install-dlib-for-python-on-mac
여기를 참고하자
"""
import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow._api.v2.compat.v1 as tf
import numpy as np

# 페이스 얼라인(Align Faces) : 5개의 랜드마크를 기준으로 얼굴의 수평을 맞춰준다.
def align_faces(img):
    dets = detector(img, 1)
    if len(dets) == 0:
        print('no detection')
    else:
        objs = dlib.full_object_detections()
        for detection in dets:
            s = sp(img, detection)
            objs.append(s)
        faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
        return faces

# 페이스 디텍션(Face Detection) : 사진에서 얼굴 부분을 찾아낸다.
def face_detection(img):
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
        plt.close()

# 랜드마크 디텍션(Landmark Detection) : 페이스 디텍션한 부분에서 눈 양쪽끝과 코끝에 총 5개의 점을 찾아준다.
def landmark_detection(img):
    img_result = img.copy()
    dets = detector(img)
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
    plt.close()

# 페이스 얼라인(Align Faces) : 5개의 랜드마크를 기준으로 얼굴의 수평을 맞춰준다.
def align_face(img):
    img_result = img.copy()
    dets = detector(img)
    objs = dlib.full_object_detections()
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
    fig, axes = plt.subplots(1, len(faces)+1, figsize=(10, 8))
    axes[0].imshow(img)
    for i, face in enumerate(faces):
        axes[i+1].imshow(face)
    plt.show()
    plt.close()

# BeautyGAN
def beauty_gan(img1, img2):
    with tf.Session() as sess:  # 위 코드를 통해 즉시실행했기 때문에 with문으로 감싸줘야한다.
        sess.run(tf.global_variables_initializer())

    sess = tf.Session()
    saver = tf.train.import_meta_graph("../models/model.meta")
    saver.restore(sess, tf.train.latest_checkpoint("../models"))
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name('X:0')  # source
    Y = graph.get_tensor_by_name('Y:0')  # reference
    Xs = graph.get_tensor_by_name('generator/xs:0')  # output

    # align_faces : Load img1(no_makeup) , img2(makeup) images
    img1_faces = align_faces(img1)
    img2_faces = align_faces(img2)

    # 실행(Run) : x: source (no-makeup) + y:reference (makeup) = xs: output
    src_img = img1_faces[0]  # 소스 이미지
    ref_img = img2_faces[0]  # 레퍼런스 이미지

    X_img = preprocess(src_img)
    X_img = np.expand_dims(X_img, axis=0)  # np.expand_dims() : 배열에 차원을 추가한다. 즉, (256,256,2) -> (1,256,256,3)

    Y_img = preprocess(ref_img)
    Y_img = np.expand_dims(Y_img, axis=0)  # 텐서플로에서 0번 axis는 배치 방향

    output = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})

    output_img = postprocess(output[0])

    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].set_title('Source')
    axes[0].imshow(src_img)
    axes[1].set_title('Reference')
    axes[1].imshow(ref_img)
    axes[2].set_title('Result')
    axes[2].imshow(output_img)

    plt.show()
    plt.close()

def preprocess(img):
    return img.astype(np.float32) / 127.5 - 1  # 0 ~ 255 -> -1 ~ 1

def postprocess(img):
    return ((img + 1.) * 127.5).astype(np.uint8)  # -1 ~ 1 -> 0 ~ 255

tf.disable_v2_behavior()
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('../models/shape_predictor_5_face_landmarks.dat')
img = dlib.load_rgb_image('../img/02.jpg')


# # align_faces test
# test_faces = align_faces(img)
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(10, 8))
# axes[0].imshow(img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
# plt.show()
# plt.close()

# 이미지 다운로드(Load Images) : Load img1(no_makeup) , img2(makeup) images
img1 = dlib.load_rgb_image("../img/me2.jpg")
img2 = dlib.load_rgb_image("../img/06.jpg")

face_detection(img1)
beauty_gan(img1, img2)