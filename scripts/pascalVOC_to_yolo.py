import os
import xml.etree.ElementTree as ET
import argparse

def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_pascalvoc_to_yolo(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(input_dir):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(input_dir, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        output_file = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))

        with open(output_file, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult')
                if difficult is not None:
                    difficult = int(difficult.text)
                else:
                    difficult = 0 
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_coordinates((w, h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PascalVOC format to YOLO v8 format')
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing PascalVOC XML files')
    parser.add_argument('output_dir', type=str, help='Path to the output directory for YOLO v8 format files')
    args = parser.parse_args()

    # Define your classes here
    classes = ["person", "hard-hat", "gloves", "mask", "glasses", "boots", "vest", "ppe-suit", "ear-protector", "safety-harness"]  # Replace with your actual classes

    convert_pascalvoc_to_yolo(args.input_dir, args.output_dir)
    print(f"Conversion completed. YOLO format files saved in {args.output_dir}")