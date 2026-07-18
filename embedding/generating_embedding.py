import os
import cv2
import numpy as np
from insightface .app import FaceAnalysis

app = FaceAnalysis()
app.prepare(ctx_id=0)

dataset_path = "dataset"
embedding_path = "embedding"

images = os.listdir(dataset_path)

for image_name in images:
    image_path = os.path.join(dataset_path, image_name)

    image= cv2.imread(image_path)
    if image is None:
        print(f"Could not read{image_path}")
        continue

    faces = app.get(image)
    print(f" Faces detected:" ,len(faces))
    
    if len(faces)==0:
        print("No face detected")
        continue

    embedding= faces[0].embedding
    filename = os.path.splitext(image_name)[0]

    embedding_file = os.path.join(
        embedding_path,
        filename +".npy"
    )
    np.save(embedding_file, embedding)

    print(f"Saved embedding:{embedding_file}")

    print("Embedding shape:", embedding.shape)
    print(f"Embedding generated for {image_name}")