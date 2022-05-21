import json
import os

def convert_xywh_to_xyxy(xywh: list[int]) -> list[int]:
    """metric conversion from [x,y,w,h] to [x,y,x,y]"""
    xyxy = []
    xyxy.append(xywh[0])
    xyxy.append(xywh[1])
    xyxy.append(xywh[0] + xywh[2])
    xyxy.append(xywh[1] + xywh[3])
    return xyxy


def convert_odgt_to_json(filename: str, key: str, val: str):
    """ filename should be odgt file which will be turned into json file. """
    file_content = {}
    with open(filename, 'r') as file:
        for line in file.readlines():
            out = json.loads(line)
            file_content[out[key]] = out[val]

    filename = os.path.splitext(filename)[0]
    with open(f"{filename}.json", 'w+') as output_file:
        json.dump(file_content, output_file, indent = 4, sort_keys = False)