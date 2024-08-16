import os
import argparse
import cv2
import numpy as np

def read_yolo_labels(label_path, image_width, image_height):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    boxes = []
    for line in lines:
        class_id, x, y, w, h = map(float, line.strip().split())
        x1 = int((x - w/2) * image_width)
        y1 = int((y - h/2) * image_height)
        x2 = int((x + w/2) * image_width)
        y2 = int((y + h/2) * image_height)
        boxes.append((int(class_id), (x1, y1, x2, y2)))
    return boxes

def write_yolo_label(file_path, boxes, image_width, image_height):
    with open(file_path, 'w') as f:
        for class_id, (x1, y1, x2, y2) in boxes:
            x = (x1 + x2) / (2 * image_width)
            y = (y1 + y2) / (2 * image_height)
            w = (x2 - x1) / image_width
            h = (y2 - y1) / image_height
            f.write(f"{max(class_id-1, 0)} {x} {y} {w} {h}\n")

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou

def process_dataset(input_dir, person_output_dir, ppe_output_dir):
    classes = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]
    
    for img_file in os.listdir(input_dir + '/images'):
        if img_file.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, "images", img_file)
            label_path = os.path.join(input_dir, "labels", img_file.rsplit('.', 1)[0] + '.txt')
            
            if not os.path.exists(label_path):
                continue
            
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            boxes = read_yolo_labels(label_path, width, height)
            
            person_boxes = [box for box in boxes if classes[box[0]] == "person"]
            if person_boxes:
                person_img_path = os.path.join(person_output_dir, "images", img_file)
                person_label_path = os.path.join(person_output_dir, "labels", img_file.rsplit('.', 1)[0] + '.txt')
                cv2.imwrite(person_img_path, img)
                write_yolo_label(person_label_path, [(0, box[1]) for box in person_boxes], width, height)
            
            for i, (class_id, (x1, y1, x2, y2)) in enumerate(person_boxes):
                person_crop = img[y1:y2, x1:x2]
                crop_height, crop_width = person_crop.shape[:2]
                
                ppe_boxes = []
                for box in boxes:
                    if classes[box[0]] != "person":
                        ppe_x1, ppe_y1, ppe_x2, ppe_y2 = box[1]
                        if calculate_iou((x1, y1, x2, y2), (ppe_x1, ppe_y1, ppe_x2, ppe_y2)) > 0:
                            # coordinates of PPE item relative to the person crop
                            crop_ppe_x1 = max(0, ppe_x1 - x1)
                            crop_ppe_y1 = max(0, ppe_y1 - y1)
                            crop_ppe_x2 = min(crop_width, ppe_x2 - x1)
                            crop_ppe_y2 = min(crop_height, ppe_y2 - y1)
                            
                            ppe_boxes.append((
                                box[0],
                                (crop_ppe_x1, crop_ppe_y1, crop_ppe_x2, crop_ppe_y2)
                            ))
                
                if ppe_boxes:
                    ppe_img_path = os.path.join(ppe_output_dir, "images", f"{img_file.rsplit('.', 1)[0]}_{i}.{img_file.rsplit('.', 1)[1]}")
                    ppe_label_path = os.path.join(ppe_output_dir, "labels", f"{img_file.rsplit('.', 1)[0]}_{i}.txt")
                    cv2.imwrite(ppe_img_path, person_crop)
                    write_yolo_label(ppe_label_path, ppe_boxes, crop_width, crop_height)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate dataset into person detection and PPE kit detection datasets')
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing images and YOLO labels')
    parser.add_argument('person_output_dir', type=str, help='Path to the output directory for person detection dataset')
    parser.add_argument('ppe_output_dir', type=str, help='Path to the output directory for PPE kit detection dataset')
    args = parser.parse_args()

    os.makedirs(args.person_output_dir, exist_ok=True)
    os.makedirs(args.ppe_output_dir, exist_ok=True)

    process_dataset(args.input_dir, args.person_output_dir, args.ppe_output_dir)
    print("Dataset separation completed.")