import os

prototxt_path = '/Users/aayushpande/MULTIPLE_FACES/Models/deploy.prototxt'
model_path = '/Users/aayushpande/MULTIPLE_FACES/Models/res10_300x300_ssd_itr_140000.caffemodel'

print("Prototxt Exists:", os.path.exists(prototxt_path))
print("Model Exists:", os.path.exists(model_path))
