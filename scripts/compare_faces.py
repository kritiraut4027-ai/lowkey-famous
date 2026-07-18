import os
import cv2
import numpy as np
from insightface .app import FaceAnalysis

def cosine_similarity(embedding1,embedding2):
    similarity = np.dot(embedding1,embedding2)/(np.linalg.norm(embedding1)*np.linalg.norm(embedding2))

    return similarity


app = FaceAnalysis()
app.prepare(ctx_id=0)

image_path = "test_images/Alia.jpg"
image = cv2.imread(image_path)

if image is None:
   print("Image could not be loaded")
   print("please upload valid image")
   exit()
#detect the image
faces = app.get(image)
if len(faces) == 0:
   print("No face detected!")
   exit()

if len(faces)>1:
   print("Please upload image containing one face")
   exit()

#generate the embedding
user_embedding = faces[0].embedding
print("faces detected successfully!")
print(user_embedding.shape)

embedding_folder = "embedding"
saved_embeddings = {}

for file in os.listdir(embedding_folder):
    if file.endswith(".npy"):

        file_path= os.path.join(embedding_folder,file)
        embedding = np.load(file_path)
        celebrity_name =os.path.splitext(file)[0]

        saved_embeddings[celebrity_name]=embedding

print(saved_embeddings.keys())

best_match = None
highest_similarity = -1

for celebrity_name,celebrity_embedding in saved_embeddings.items():

  similarity = cosine_similarity(user_embedding,celebrity_embedding) 
print(f"{celebrity_name} :{similarity :4f}")

if similarity>highest_similarity:
   highest_similarity =similarity
   best_match = celebrity_name

percentage = highest_similarity * 100

if percentage <40:
   print("\n No close match found")
   exit()

print("\n..............")
print(f"Your Hamshakal is :{best_match}")
print(f"Match Percentage:{percentage:2f}%")

celeb_image_path = os.path.join("dataset",best_match+".jpg")
print(celeb_image_path)
celeb_image = cv2.imread(celeb_image_path)

if celeb_image is None:
   
   print("Celebrity image is not found!")

else:
   cv2.imshow("Your hamshakal",celeb_image)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
      
