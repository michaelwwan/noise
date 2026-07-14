# Testing github new local directory 2.4.26

import os, sys
import glob
from ultralytics import YOLO
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint
import torch
import torchvision
import argparse
import math
import numpy as np
import json
from skspatial.measurement import area_signed
from shapely.geometry import Polygon, box as shp_box
from shapely.ops import unary_union

# um/pixel length
UM_PER_PIXEL = 0.7784
UM_PER_PATCH = UM_PER_PIXEL * 832
#0.5945 µm2/pixel
MAX_DET = 30000
Image.MAX_IMAGE_PIXELS = 1000000000

DEVICE = torch.device('cuda:0')


# ---- Detection aggregation: overlap-normalized mask IoU with fragment bridging ----
# Tiles overlap ~50%, so a cell can appear across tiles either as near-duplicates or,
# when larger than a tile, as boundary-split fragments. We walk tiles in raster order
# accumulating detections in global coords. For each new tile we merge a new detection
# Y into every accepted detection X it matches, scoring the match by IoU normalized to
# the co-observed region I = (new tile) intersect (already-visited tiles):
#     rel_iou(X, Y) = area(X ∩ Y) / area((X ∪ Y) ∩ I)
# Normalizing by the shared region means fragment parts lying outside I do not suppress
# the match. All matching X are bridged with Y into a single union, so a cell spanning
# several tiles is reconstructed as one detection.

REL_IOU_THRESHOLD = 0.5

def _largest_polygon(geom):
    """Reduce any geometry to its largest single Polygon component, or None.

    Handles Polygon, MultiPolygon, and GeometryCollection (e.g. results of
    buffer(0)/unary_union that split a self-intersecting shape into pieces).
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return geom
    polys = [g for g in getattr(geom, 'geoms', [])
             if g.geom_type == 'Polygon' and not g.is_empty]
    if not polys:
        return None
    return max(polys, key=lambda g: g.area)

def _poly_from_xy(xy):
    """Build a valid single shapely Polygon from an (N,2) array of global coords, or None."""
    if xy is None or len(xy) < 3:
        return None
    p = Polygon(xy)
    if not p.is_valid:
        p = p.buffer(0)          # repair self-intersections; may yield Multi/empty
    p = _largest_polygon(p)      # normalize to a single Polygon
    if p is None or p.is_empty or p.area == 0.0:
        return None
    return p

def rel_iou(X, Y, I):
    """Overlap-normalized IoU: area(X ∩ Y) / area((X ∪ Y) ∩ I); 0 if undefined."""
    inter = X.intersection(Y).area
    if inter == 0.0:
        return 0.0
    denom = X.union(Y).intersection(I).area
    if denom == 0.0:
        return 0.0
    return inter / denom

def inference(model, img, img_filename, size, out_dir):

    stride = size // 2
    accepted = []         # list of {"poly": Polygon, "score": float, "cls": int}, global coords
    processed_tiles = []  # shapely boxes of tiles already visited

    for y0 in range(0, img.size[1], stride):
        for x0 in range(0, img.size[0], stride):

            tile_rect = shp_box(x0, y0, x0 + size, y0 + size)

            # I: the part of this tile already observed by earlier overlapping tiles.
            # Every accepted X lies within prior coverage, so X ∩ Y ⊆ I always holds --
            # that keeps rel_iou's numerator inside its denominator (ratio in [0, 1]).
            prior = [t for t in processed_tiles if t.intersects(tile_rect)]
            I = tile_rect.intersection(unary_union(prior)) if prior else None

            # Crop onto white (PIL's default black fill causes false detections)
            img_crop = Image.new('RGB', (size, size), (255, 255, 255))
            img_crop.paste(img, (-x0, -y0))

            results = model(img_crop, verbose=False, device=DEVICE)

            for r in results:
                if r.masks is None:
                    continue
                boxes = r.boxes.data  # (N, 6): x1,y1,x2,y2,conf,cls in crop-local coords
                for det_i, poly_xy in enumerate(r.masks.xy):
                    xy = np.asarray(poly_xy, dtype=float).copy()
                    if len(xy) < 3:
                        continue
                    xy[:, 0] += x0  # crop-local -> global
                    xy[:, 1] += y0
                    Y = _poly_from_xy(xy)
                    if Y is None:
                        continue
                    conf = float(boxes[det_i, 4])
                    cls = int(boxes[det_i, 5])

                    # Match Y against accepted detections. intersects(tile_rect) is a
                    # cheap prefilter so we only run rel_iou on X's that reach into this
                    # tile, instead of scanning the whole accumulated list every time.
                    matches = []
                    if I is not None and not I.is_empty:
                        for k, X in enumerate(accepted):
                            if X["poly"].intersects(tile_rect) and \
                               rel_iou(X["poly"], Y, I) > REL_IOU_THRESHOLD:
                                matches.append(k)

                    if matches:
                        # Bridge every matched detection with Y into one union. Y overlaps
                        # each match, so the union is normally a single connected Polygon;
                        # fall back to Y if it ever degenerates to no polygon component.
                        merged = unary_union([accepted[k]["poly"] for k in matches] + [Y])
                        merged = _largest_polygon(merged) or Y
                        score = max([accepted[k]["score"] for k in matches] + [conf])
                        for k in sorted(matches, reverse=True):
                            accepted.pop(k)
                        accepted.append({"poly": merged, "score": score, "cls": cls})
                    else:
                        accepted.append({"poly": Y, "score": conf, "cls": cls})

            processed_tiles.append(tile_rect)

    # ---- write detections: box, objectness, flattened global mask coords ----
    with open("{f}/{id}".format(f=out_dir, id=img_filename[:-4] + ".txt"), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["box_x1", "box_y1", "box_x2", "box_y2", "objectness_score",
                         "mask_x1", "mask_y1", "mask_x2", "mask_y2", "..."])
        if len(accepted) == 0:
            return f.write("No osteoclasts detected")
        for det in accepted:
            minx, miny, maxx, maxy = det["poly"].bounds
            mask = np.asarray(det["poly"].exterior.coords).flatten().tolist()
            writer.writerow([minx, miny, maxx, maxy, det["score"]] + mask)

    # ---- annotated image ----
    img1 = ImageDraw.Draw(img, 'RGBA')
    for i, det in enumerate(accepted):
        minx, miny, maxx, maxy = [int(v) for v in det["poly"].bounds]
        shape = [(max(0, minx), max(0, miny)),
                 (min(maxx, img.size[0] - 1), min(maxy, img.size[1] - 1))]
        img1.rectangle(shape, outline="red", width=3)
        coords = [(int(px), int(py)) for px, py in det["poly"].exterior.coords]
        if len(coords) >= 3:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            img1.polygon(coords, fill=color + (125,), outline="blue")
    img.save("{f}/{id}".format(f=out_dir, id=img_filename))

    return accepted

def count_ocls_from_output(img_dir, out_dir):
    
    # This script will count each newline for the files in the output directory

    #This will save the output_files to a list from the output directory and only include the txt files
    output_files = glob.glob((out_dir) + "*.txt")

    #Add the image directory name to output file
    image_dir = (img_dir.rsplit("/")) # split usr give image directory into a list split by /

    # This will make sure that a blank name is not written as part of the ocl_count output folder name
    if len(image_dir[-1]) > 0:
        output_name = image_dir[-1]
    else:
        output_name = image_dir[-2]
    
    col_name = ["Image_Name", "Ocl_Count"] # Add the column names to top of csv.

    csv_file_name = "ocl_counts_" + str(output_name) +".csv" # file name

    with open(csv_file_name, "a", newline = '') as csvfile:
        if os.stat(csv_file_name).st_size == 0: # only put the column name row in when the file is empty (at start)
            writer = csv.writer(csvfile)
            writer.writerow(col_name)
        
    split_dir = len(out_dir)
    #To iterate over each file in that output directory
    for file in output_files:
        counts_list = []
        with open(file, "r") as f: # f is now the object of each file
            as_string = str(f.read())
            split_string = as_string.split("\n")
            counts_list.append("{id}".format(id=f.name[split_dir:-4]))
            counts_list.append(str(len(split_string[1:-1])))
            with open(csv_file_name, "a", newline = '') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(counts_list)
        csvfile.close
    
#Below functions are required for area calculations
def masking_coordinates_to_list(out_dir):

    # This function will count each newline for the files in the output directory

    #This will save the output_files to a list from the output directory and only include the txt files
    output_files = glob.glob((out_dir) + "*.txt")

    dir_list = [] # List will contain each .txt file name containing the masking coordinates. 
    for file in os.listdir(out_dir):
        if file.endswith('.txt'):
            dir_list.append(file)

    length_files_in_dir = (len(dir_list)) # Save how many files are in the directory
    
    counter = 0
    coordinate_dict = {}  # Save each file name as a key to the masking coordinate value
    #To iterate over each file in that output directory
    while len(coordinate_dict) != length_files_in_dir:
        for file in (output_files):
            with open(file, "r") as f: # f is now the object of each file
                as_string = str(f.read())
                split_string = as_string.split("\n") # Split string has each osteoclast masking coordinates in an element of a list.
    
                coordinate_dict[dir_list[counter]] = split_string
                counter += 1
    #print(coordinate_dict)
    return coordinate_dict # The coordinate dict will have each file name as a key and the masking coordinates as a value.
              
def calculate_pixel_area(coordinate_list_as_floats):
    '''This function will create a 2d array of the masking coordinates
    as input for the area_signed function which utilizes the shoelace
    formula to calculate area of an irregular polygon. 

    Returns the pixel area of each osteoclast. '''

    for i in coordinate_list_as_floats:
         if isinstance(i, float):      
            array_2d = np.array(coordinate_list_as_floats).reshape((len(coordinate_list_as_floats))//2,2) #Create numpy array

            pixel_area = (abs(area_signed(array_2d))) # Output Pixel area of each osteoclast using shoelace formula

            return (pixel_area)

def total_area_per_well(area_list):

    # Each area of an ocl will be added to the total area 
    total_area_per_well_sum = 0

    for i in area_list:
         total_area_per_well_sum += i

    return round(total_area_per_well_sum, 3)


def percent_ocl_area_per_well(total_area, well_area_in_pixels):

    '''Function will calculate the percent of osteoclast area on each well.
    The well_area_in_pixels is set to the user entered area.'''
    perecent_area = (total_area/well_area_in_pixels)*100

    return round(perecent_area, 3)

def write_area_to_output(img_dir, total_area_per_well_sum, percent_area, out_dir ,key):

    '''Function will write each area to a csv.'''

    image_dir = (img_dir.rsplit("/"))

    if len(image_dir[-1]) > 0:
        output_name = image_dir[-1]
    else:
        output_name = image_dir[-2]
    # Add a row for column names to top of file

    col_name = ["Image_Name", "Total_Area", "%_Area"] # First row of csv with column names

    csv_file_name = "ocl_area_" + str(output_name) + ".csv" # name of file to store data

    with open(csv_file_name, "a", newline = '') as csvfile:
        if os.stat(csv_file_name).st_size == 0: # only put the column name row in when the file is empty (at start)
            writer = csv.writer(csvfile)
            writer.writerow(col_name)

    area_output_tuple = [key[:-4], str(total_area_per_well_sum), str(percent_area)] # tuple to store the row to write to csv
    with open(csv_file_name, "a", newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(area_output_tuple)
    csvfile.close

def write_individual_ocl_area_to_output(img_dir, all_individual_ocl_areas_dict):
    '''Going to output a file with all ocl areas'''

    image_dir = (img_dir.rsplit("/"))

    if len(image_dir[-1]) > 0:
        output_name = image_dir[-1]
    else:
        output_name = image_dir[-2]

    col_name = ["Image_Name"]

    csv_file_name = "ocl_individual_areas_" + str(output_name) + ".csv"

    with open(csv_file_name, "a", newline = '') as csvfile:
        if os.stat(csv_file_name).st_size == 0: # only put the column name row in when the file is empty (at start)
            writer = csv.writer(csvfile)
            writer.writerow(col_name)

    with open (csv_file_name,'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in all_individual_ocl_areas_dict.items():
            writer.writerow([key[:-4], *value]) # write a row as image name, then the individual areas. 

    csvfile.close()


def main(argv):
    
    # move the sys args into parser
    parser = argparse.ArgumentParser()

    #Add option to utilize params.json file
    parser.add_argument("--params", type=str, default=None) #should be the params file path

    parser.add_argument("--img_foldername", type=str, default="img")
    parser.add_argument("--out_foldername", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="out")
    parser.add_argument("--ratio", type=float, default=0.7784) #um per pixel
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--total_well_area_in_pixels", type = int, default = 0)
    

    args = parser.parse_args()

    json_parameter = args.params

    if json_parameter != None: # if params file given
       with open(json_parameter) as params_file: # open the params file
           data = json.load(params_file) # store params as dict in data

    for param,argument in data.items(): # parse params dict, assigning each usr argument to correct variable
        
        if param == "model_path":
            model_path = argument # the model path is equal to json given argument
            
        if param == "img_foldername":
            img_dir = argument

        if param ==  "out_foldername":
            out_dir = argument

        if param == "ratio":
            um_per_pixel = argument
            patch_size = int( UM_PER_PATCH/um_per_pixel )

        if param == "total_well_area_in_pixels":
            well_area_in_pixels = argument 

        if "total_well_area_in_pixels" not in data:
            well_area_in_pixels = 0 # sets the well_area to 0, will return None for % area

        if param == "device":
            usr_device = argument

        if "device" not in data:
            usr_device = "cpu" # sets the usr_device to default cpu if it's not provided 

    if json_parameter == None: # If no json params file provided, will use user arguments to run the command
    
        um_per_pixel = args.ratio
        patch_size = int( UM_PER_PATCH/um_per_pixel )
        
        out_dir = args.out_foldername
        img_dir = args.img_foldername

        well_area_in_pixels = args.total_well_area_in_pixels

        model_path = args.model_path
        model = YOLO(model_path)

        usr_device = args.device

    global DEVICE
    DEVICE = torch.device(usr_device)
    
    if out_dir == img_dir:
        print("Error: Input directory equals output directory. Please specify a unique output directory.")
        return
    
    # check if out_dir exists and create if it doesn't
    if not os.path.exists( out_dir ):
        os.makedirs( out_dir )
        
    model = YOLO(model_path)
        
    img_files = [ file for file in os.listdir(img_dir) if not file.startswith(".") ]
    for img_filename in img_files:
        print(img_filename)
        
        img = Image.open( os.path.join(img_dir, img_filename) )
        
        pred = inference(model, img, img_filename, patch_size, out_dir)

    
    count_ocls_from_output(img_dir, out_dir)

    split_string = masking_coordinates_to_list(out_dir) # Split string variable is now a dictionary, where each each key is a txt and value is a set of masking coordinates.
    
    # Store all individual areas in dict

    all_individual_ocl_areas_dict = {} # key = image_name, value = list of individual ocl areas. 

    for keys,values in (split_string.items()):
        pixel_area_list = [] # The list containing the pixel area calculated by the shoelace formula
        file_key = []
        for i in range(len(values)):
            if i != 0 and i != (len(values)-1):
                coordinate_list = values[i].split(',')     
                coordinate_list_as_floats = [] # New list is now a list of coordinates as a float
                for i in coordinate_list[5:]: # This will exclude the box coordinates and object score
                    if len(coordinate_list[5:]) >= 6: #This will make sure that the area is at least a three sided polygon. 
                        flt = float(i)
                        coordinate_list_as_floats.append(flt)
                    else:
                        continue

                # This conditional statement will make sure that every coordinate list has floats
                if len(coordinate_list_as_floats) >= 6 :
                    pixel_area = calculate_pixel_area(coordinate_list_as_floats)
                pixel_area_list.append(pixel_area)

        # create dict of image_name as keys and pixel area of all the individual ocl areas as values. 
        all_individual_ocl_areas_dict[keys] = pixel_area_list

        # Below will determine the total area of ocls in each well in pixels.
        # This will calculate total area of ocls in pixels
        total_area = (total_area_per_well(pixel_area_list))

        # Below was utilized to provide user with total area in pixels.
        if well_area_in_pixels != 0:
            percent_ocl_each_well = percent_ocl_area_per_well(total_area, well_area_in_pixels)
        
            write_area_to_output(img_dir, total_area, percent_ocl_each_well, out_dir, keys)
        
        else:
            percent_ocl_each_well = "None"
            write_area_to_output(img_dir, total_area, percent_ocl_each_well, out_dir, keys)

            print("If you want total area calculated, please enter pixel area of each well into the --total_well_area_in_pixels argument.")

    
    # function to create csv with all individual areas. 
    write_individual_ocl_area_to_output(img_dir, all_individual_ocl_areas_dict)

    return

if __name__ == '__main__':
    main(sys.argv)
