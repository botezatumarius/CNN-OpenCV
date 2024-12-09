import os
import cv2
import pandas as pd

directories = ['train/true', 'train/false', 'val/true', 'val/false', 'test/true', 'test/false']
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

df = pd.read_csv('labels.csv')

trn_ct = 31
val_ct = 10
tst_ct = 7

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    image_path = row['new_path']
    label = row['label']
    img = cv2.imread(image_path)
    
    # Extract the image filename from the path
    image_name = os.path.basename(image_path)
    
    # Check the label and save the image accordingly
    if label:
        if trn_ct != 0:
            cv2.imwrite(f'train/true/{image_name}', img)
            trn_ct -= 1
        elif val_ct != 0:
            cv2.imwrite(f'val/true/{image_name}', img)
            val_ct -= 1
        elif tst_ct != 0:
            cv2.imwrite(f'test/true/{image_name}', img)
            tst_ct -= 1
    else:
        if trn_ct != 0:
            cv2.imwrite(f'train/false/{image_name}', img)
            trn_ct -= 1
        elif val_ct != 0:
            cv2.imwrite(f'val/false/{image_name}', img)
            val_ct -= 1
        elif tst_ct != 0:
            cv2.imwrite(f'test/false/{image_name}', img)
            tst_ct -= 1

print("Image categorization complete!")
