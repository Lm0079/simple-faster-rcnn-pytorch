import os
import sys
import json
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
config_file = "/user/work/gh18931/diss/datasets/task_simple-frcnn-data/annotations/instances_default.json"

#1 - image namd and location
#2 - bounding box with image id and name
# 3 - image label and class names
def get_images(data):
    images = {}
    for image in data:
        id = image["id"]
        w,h = image["width"], image["height"]
        filename = image["file_name"]
        images[id]= {"w":w,"h":h,"name":filename}
    return images
def convert_bbox(bbox):
    new_bbox = [bbox[0],(bbox[0]+bbox[2]),bbox[1], (bbox[1]+bbox[3])]
    return new_bbox
## faster rcnn format is ( fname_path, xmin, xmax, ymin, y_max, label)
def get_bboxes( data, image_info):
    bboxes = {}
    for annotation in data:      
        id = annotation["id"]
        image_id = annotation["image_id"]
        bbox = convert_bbox(annotation["bbox"])
        image_name = image_info[image_id]["name"]
        bboxes[id] = {"name":image_name,"bbox":bbox,"imageid":image_id,"class":annotation["category_id"]}

    return bboxes
def write_anno():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # for training
    with open("annotation.txt", "w+") as f:
      for idx, row in train_df.iterrows():
          img = cv2.imread('train/' + row['FileName'])

          x1 = int(row['XMin'] )
          x2 = int(row['XMax'] )
          y1 = int(row['YMin'] )
          y2 = int(row['YMax'] )
          fileName = os.path.join("train", row['FileName'])
          className = row['ClassName']
          f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(className)+ '\n')



    # for test
    with open("test_annotation.txt", "w+") as f:
      for idx, row in test_df.iterrows():
          sys.stdout.write(str(idx) + '\r')
          sys.stdout.flush()
          img = cv2.imread('test/' + row['FileName'])

          x1 = int(row['XMin'] )
          x2 = int(row['XMax'] )
          y1 = int(row['YMin'] )
          y2 = int(row['YMax'] )


          fileName = os.path.join("test", row['FileName'])
          className = row['ClassName']
          f.write(fileName + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' +  str(className)+ '\n')

def split_data(data,ids):
    X_train, X_test,  = train_test_split( ids,  test_size=0.2, random_state=42)

    train_dict = make_split_dicts(data, X_train)
    test_dict = make_split_dicts(data, X_test)

    save_split_data(train_dict,True)
    save_split_data(test_dict,False)
    train_df = make_df(train_dict,True)
    test_df = make_df(test_dict,False)
    train_df.to_csv('train.csv')
    test_df.to_csv('test.csv')

def save_split_data(data, train=True):
    
    if train:
       # print("-------------------------train------------------")
        output_dir = "/user/work/gh18931/diss/datasets/task_simple-frcnn-data/train/"
       # print(data.keys())
        for id in data:

            name = data[id]["name"]
            img = Image.open(f"/user/work/gh18931/diss/datasets/task_simple-frcnn-data/images/{name}")
            
            img = img.save(f"{output_dir}{name}")
    else:
      #  print("-------------------------test------------------")
        output_dir = "/user/work/gh18931/diss/datasets/task_simple-frcnn-data/test/"
        for id in data:

            name = data[id]["name"]
            img = Image.open(f"images/{name}")
            #print(f"{output_dir}{name}")
            img = img.save(f"{output_dir}{name}")
  

def make_split_dicts(data,labels):
    new_dict= {}
    for label in labels:
        for d in data:
            if data[d]["imageid"] == label:
                new_dict[d] = data[d]
    return new_dict

def make_df(dataset,train = True):
    val_df = pd.DataFrame(columns=['FileName', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
    if train:
        output = "/user/work/gh18931/diss/datasets/task_simple-frcnn-data/train/"
    else:
        output = "/user/work/gh18931/diss/datasets/task_simple-frcnn-data/test/"
    for id in dataset:
        data = dataset[id]
        val_df = val_df.append({'FileName':output+ data["name"],
                                            'XMin': data["bbox"][0],
                                            'XMax':  data["bbox"][1],
                                            'YMin':  data["bbox"][2],
                                            'YMax':  data["bbox"][3],
                                            'ClassName': data["class"]},
                                           ignore_index=True)
    return val_df

def main():
    print("Hello World!")
    with open(config_file,"r") as f:
        data = json.load(f)
     
        image_info = get_images(data["images"])
        bboxes = get_bboxes(data["annotations"],image_info)
        
        ids = list(image_info.keys())
        
        #split data inot train test and move them to dictories
        split_data(bboxes,ids)
        # make a train and test df and same it
        write_anno()
    # coco json format is xmin,ymix, width, height


if __name__ == "__main__":
    main()
