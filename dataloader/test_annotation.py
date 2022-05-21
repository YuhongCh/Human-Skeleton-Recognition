import cv2
import json
from utils import convert_xywh_to_xyxy, convert_odgt_to_json


def test_annotation(filename: str, annotation: json):
    img = cv2.imread(filename)
    for tag in annotation['gtboxes']:
        if tag['tag'] == 'person':
            box = convert_xywh_to_xyxy(tag['hbox'])
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0 ,0), 2)
        else:
            box = convert_xywh_to_xyxy(tag['hbox'])
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imwrite("examples_images/hbox_example_image.jpg", img)
    return


def main():
    # generate the example images
    filename = "../dataset/Images/273271,1a02900084ed5ae8.jpg"
    with open("../dataset/annotation_train.odgt", 'r') as annotation:
        line = annotation.readlines()[9366]
        annot = json.loads(line)
        test_annotation(filename, annot)

    # convert json file
    convert_odgt_to_json("../dataset/annotation_train.odgt", "ID", "gtboxes")

if __name__ == "__main__":
    main()