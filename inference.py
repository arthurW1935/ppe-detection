import os
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def run_inference(person_model, ppe_model, input_dir, output_dir, ppe_class_names):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Run inference on each image
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_dir, image_file)
        
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Run person detection
        person_results = person_model(image, verbose=False)
        
        for result in person_results:
            for bbox in result.boxes.xyxy:
                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, bbox)
                
                # Ensure coordinates are within image bounds
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img_width, x_max), min(img_height, y_max)
                
                # Crop the person image
                person_crop = image[y_min:y_max, x_min:x_max]
                
                if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                    # Run PPE detection on the cropped image
                    ppe_results = ppe_model(person_crop, verbose=False)
                    
                    for ppe_result in ppe_results:
                        for ppe_bbox, ppe_class_id in zip(ppe_result.boxes.xyxy, ppe_result.boxes.cls):
                            ppe_x_min, ppe_y_min, ppe_x_max, ppe_y_max = map(int, ppe_bbox)
                            
                            # Adjust PPE bounding box coordinates relative to the original image
                            ppe_x_min += x_min
                            ppe_y_min += y_min
                            ppe_x_max += x_min
                            ppe_y_max += y_min
                            
                            # Ensure adjusted coordinates are within the original image bounds
                            ppe_x_min, ppe_y_min = max(0, ppe_x_min), max(0, ppe_y_min)
                            ppe_x_max, ppe_y_max = min(img_width, ppe_x_max), min(img_height, ppe_y_max)
                            
                            # Draw bounding box for PPE detection on the original image
                            cv2.rectangle(image, (ppe_x_min, ppe_y_min), (ppe_x_max, ppe_y_max), (0, 255, 0), 2)

                            # Annotate the image with the class name
                            class_name = ppe_class_names[int(ppe_class_id)]
                            cv2.putText(image, class_name, (ppe_x_min, ppe_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the image with detections
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

def main():
    parser = argparse.ArgumentParser(description="Run PPE detection inference on images.")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_dir", help="Directory to save output images")
    args = parser.parse_args()

    # Load the trained models
    person_model = YOLO('person.pt')
    ppe_model = YOLO('ppe.pt')

    # Define PPE class names (update this list according to your model's classes)
    ppe_class_names = ["hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]

    run_inference(person_model, ppe_model, args.input_dir, args.output_dir, ppe_class_names)

if __name__ == "__main__":
    main()