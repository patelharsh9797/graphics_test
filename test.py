import os
import csv
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.spatial import distance
import cv2
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

embedder = FaceNet()
def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

def resize_image(img, target_size):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    if w > h:
        new_w = target_w
        new_h = int(h * (new_w / w))
    else:
        new_h = target_h
        new_w = int(w * (new_h / h))
    
    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img

def find_matching_photos(query_embedding, embeddings, filenames, threshold=0.33):
    distances = np.array([distance.cosine(embedding, query_embedding) for embedding in embeddings])
    print(distances)
    matches = np.where(distances <= threshold)[0]
    return [filenames[i] for i in matches]


detector = MTCNN()
data_folder_name = root_dir+'/album'
def find_embeddings(data_folder_name):
    photographer_folder_file = os.listdir(data_folder_name)
    csv_file_path = root_dir+'/photos.csv'
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['filename', 'embedding'])
        for idx, file_name in enumerate(photographer_folder_file):
            print(idx)
            img_path = os.path.join(data_folder_name, file_name)
            img = cv2.imread(img_path)
            if img.shape[0]*img.shape[1] > 20000000:
                img = resize_image(img,[4500,3000])
            results = detector.detect_faces(img)
            for result in results:
                if result['confidence'] >= 0.98:
                    x,y,w,h = result['box']
                    my_face = img[y:y+h, x:x+w]
                    face_arr = cv2.resize(my_face, (160,160))
                    embedding = get_embedding(face_arr)
                    embedding_str = ','.join(map(str, embedding))
                    csv_writer.writerow([file_name, embedding_str])



selfie_path = root_dir+'/album/ajay.jpeg'
def selfie_embeddings(selfie_path):
    # Load facial embeddings from CSV file to match with selfie
    embeddings = []
    filenames = []
    with open(root_dir+'/photos.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            embedding = np.array([float(val) for val in row['embedding'].split(',')])
            embeddings.append(embedding)
            filenames += [i for i in row['filename'].split(',')]

    selfie_image = cv2.imread(selfie_path)
    matching_photo_filenames = []
    selfie_image = cv2.cvtColor(selfie_image, cv2.COLOR_BGR2RGB)
    if selfie_image.shape[0]*selfie_image.shape[1] > 20000000:
        selfie_image = resize_image(selfie_image,[6000,4000])
    results = detector.detect_faces(selfie_image)
    face_num = []
    for result in results:
        if result['confidence'] >= 0.98:
            face_num.append(result)
    if len(face_num) == 1:
        x,y,w,h = face_num[0]['box']
        my_face = selfie_image[y:y+h, x:x+w]
        face_arr = cv2.resize(my_face, (160,160))
        selfie_embedding = get_embedding(face_arr)
        matching_photo_filenames = find_matching_photos(selfie_embedding, embeddings, filenames)
        print('Done')
    elif len(face_num) == 1:
        print("There is no face detected in Selfie...")
    else:
        print("Selfie has more than 1 face...")

    return matching_photo_filenames

find_embeddings(data_folder_name)
matching_photo_filenames = selfie_embeddings(selfie_path)
print(matching_photo_filenames)

