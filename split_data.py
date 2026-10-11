import os 
import splitfolders 

DATA_SET_FOLDER="dataset_annote"
DATA_SET_IMAGE="dataset_annote/images"
DATA_LABEL="dataset_annote/labels"
TARGET_ROOT_PATH="./newdata/"
TARGET_IMAGES=TARGET_ROOT_PATH+"/images/"
TARGET_LABELS=TARGET_ROOT_PATH+"/labels/"

TRAINING_RATIO = 0.70
EVAL_RATIO = 0.15
TEST_RATIO = 0.15

splitfolders.ratio(input=DATA_SET_IMAGE, output=TARGET_IMAGES, ratio=(TRAINING_RATIO,EVAL_RATIO,TEST_RATIO))
splitfolders.ratio(input=DATA_LABEL, output=TARGET_LABELS, ratio=(TRAINING_RATIO,EVAL_RATIO,TEST_RATIO))





