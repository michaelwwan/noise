import os, sys
import glob
from ultralytics import YOLO
import csv
from PIL import Image, ImageDraw
import numpy as np
from random import randint
import torch
import torchvision
import argparse
import math

# um/pixel length
UM_PER_PIXEL = 0.7784
UM_PER_PATCH = UM_PER_PIXEL * 832
#0.5945 µm2/pixel
MAX_DET = 30000
Image.MAX_IMAGE_PIXELS = 100000000

DEVICE = torch.device('cuda:0')


def scale_boxes(boxes, num_images, img_ind, img_scale):
    boxes[:,(0,2)] = boxes[:,(0,2)] + (img_scale[0]/2)*img_ind[0] # x
    boxes[:,(1,3)] = boxes[:,(1,3)] + (img_scale[1]/2)*img_ind[1] # y
    return boxes
    
def scale_masks(masks, num_images, img_ind, img_scale):
    for m in range(len(masks)):
        masks[m] = masks[m] + (img_scale/2)*img_ind # x
    return masks

# Credit to torchvision/ops/boxes.py
def box_inter_union(boxes1, boxes2):
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
    
def local_nms(box_results, mask_results):

    keep_bool_master = []
    
    # Collate predictions from neighboring crops
    boxes_torch = torch.cat( [ box_results[r][c] for r,c in zip([0,0,0,1,1,2,2,2],[0,1,2,0,2,0,1,2]) ] )
    
    # If no neighboring predictions, skip
    if torch.numel(boxes_torch)==0:
        return torch.tensor([], device=DEVICE), []
    
    for box in box_results[1][1]:
        
        intersection = box_inter_union(box[:4].unsqueeze(0), boxes_torch[:,:4])[0]
        surf_area = torchvision.ops.box_area( box[:4].unsqueeze(0) )[0]
        sig_overlap = ((surf_area - intersection)/surf_area) < 0.1
        
        # If center box significantly overlaps any neighboring box
        if torch.any( sig_overlap ):
            neighboring_areas = torchvision.ops.box_area( boxes_torch[sig_overlap[0],:4] )
            
            # Keep center box if its larger than all boxes
            if torch.all(surf_area > neighboring_areas):
                keep_bool_master.append( True )
            else:
                keep_bool_master.append( False )
        else: # If no significant overlap, keep box
            keep_bool_master.append( True )
        
    # If no predictions remain
    if not any(keep_bool_master):
        return torch.tensor([], device=DEVICE), []
    
    # Collate large non-overlapping predictions
    box_results = box_results[1][1][keep_bool_master]
    mask_results = [ mask_results[1][1][i] for i,val in enumerate(keep_bool_master) if val ]
    return box_results, mask_results
    
def inference(model, img, img_filename, size, out_dir):
    empty_tensor = torch.tensor([], device=DEVICE)
    
    # divide size of image by size of patch/2
    num_patches = ( np.array(img.size) / (size/2) ).astype(int)
    
    # Run inference on each image
    box_results = []
    mask_results = []
    
    row_range = 2 + img.size[1]//(size//2)
    col_range = int( 2 + img.size[0]//(size//2) )
    box_results = [ [empty_tensor for _ in range(col_range)] for _ in range(row_range) ]
    mask_results= [ [[] for _ in range(col_range)] for _ in range(row_range) ]
    
    for yi, y0 in enumerate( range(0, img.size[1], size//2) ):
        for xi, x0 in enumerate( range(0, img.size[0], size//2) ):
            x1, y1 = x0+size, y0+size
            
            # save crops
            img_crop = img.crop((x0,y0,x1,y1))
            yc=math.ceil(y0/(size//2))
            xc=math.ceil(x0/(size//2))
            
            results = model( img_crop, verbose=False, device=DEVICE )
            img_ind = np.array((xc,yc))
            
            # Scale the predictions back to their proper size
            for r in range(len(results)):
                boxes = results[r].boxes.data.clone()
                
                # If osteoclasts detected, scale/translate prediction to original image size
                if boxes.numel() != 0:
                    box_results[yi+1][xi+1] = scale_boxes( boxes, num_patches, img_ind, (size,size) )
                    mask_results[yi+1][xi+1]= scale_masks( results[r].masks.xy, num_patches, img_ind, np.array((size,size)) )
                else:
                    box_results[yi+1][xi+1] = empty_tensor
                    mask_results[yi+1][xi+1]= []
                 
    
    objects_found = True if box_results else False
    
    '''Collate object predictions while removing overlapping duplicates'''
    if objects_found:
    
        new_box_results = []
        new_mask_results = []
        # Loop over 3x3 squares of neighboring crops
        for r in range(1,len(box_results)-1):
            for c in range(1,len(box_results[r])-1):
                
                # If central crop produced no predictions
                if torch.numel(box_results[r][c])==0:
                    continue
                
                output = local_nms( [ b[c-1:c+2] for b in box_results[r-1:r+2] ], [ m[c-1:c+2] for m in mask_results[r-1:r+2] ] )
                
                new_box_results.append(output[0])
                new_mask_results += output[1]

        # If no osteoclasts are detected in image, this will handle the output
        if len(new_box_results) > 0:
            box_results = torch.cat(new_box_results)
            mask_results = new_mask_results
        else:
            box_results = [0]
            mask_results = new_mask_results

    '''Save text output'''
    with open("{f}/{id}".format(f=out_dir, id=img_filename[:-4]+".txt"), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow( ["box_x1","box_y1","box_x2","box_y2","objectness_score","mask_x1","mask_y1","mask_x2","mask_y2","..."] )
        if (len(box_results)) > 1:
            for i in range(len(box_results)):
                writer.writerow( box_results[i].tolist()[:-1] + mask_results[i].flatten().tolist() )
        else:
             return f.write("No osteoclasts detected")

    '''Save visual output'''
    img1 = ImageDraw.Draw(img, 'RGBA')
    
    for i, box in enumerate(box_results):
        box = box[:4].type(torch.int)
        shape = [(box[0], box[1]), (box[2], box[3])]
        img1.rectangle(shape, outline="green", width=3)
        mask = mask_results[i].astype(int).flatten().tolist()
        if len(mask) >= 6:
            color = (randint(0,255),randint(0,255),randint(0,255))
            img1.polygon(mask, fill=color+(125,), outline="blue")
            
    img.save( "{f}/{id}".format(f=out_dir, id=img_filename) )
    
    if objects_found:
        return [{"boxes":box_results[:,:4], "scores":box_results[:,4], "labels":box_results[:,5].int()}]
    else:
        return [{"boxes":[], "scores":[], "labels":[]}]
    
def count_ocls_from_output(out_dir):
    
    # This script will count each newline for the files in the output directory

    #This will save the output_files to a list from the output directory and only include the txt files
    output_files = glob.glob((out_dir) + "*.txt")

    #To iterate over each file in that output directory
    for file in output_files:
        with open(file, "r") as f: # f is now the object of each file
            as_string = str(f.read())
            split_string = as_string.split("\n")
            count_value = (len(split_string[1:-1]))
            with open("ocl_counts.txt", "a") as file:
                file.write("{id}".format(id=f.name[:-4]) + ": " + str(count_value) + "\n")
            file.close


def main(argv):
    
    # move the sys args into parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_foldername", type=str, default="img")
    parser.add_argument("--out_foldername", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="out")
    parser.add_argument("--ratio", type=float, default=0.7784) #um per pixel
    parser.add_argument("--device", type=str, default='cpu')
    
    args = parser.parse_args()
    
    um_per_pixel = args.ratio
    patch_size = int( UM_PER_PATCH/um_per_pixel )
    
    out_dir = args.out_foldername
    img_dir = args.img_foldername

    global DEVICE
    DEVICE = torch.device(args.device)
    
    if out_dir == img_dir:
        print("Error: Input directory equals output directory. Please specify a unique output directory.")
        return
    
    # check if out_dir exists and create if it doesn't
    if not os.path.exists( out_dir ):
        os.makedirs( out_dir )
        
    model_path = args.model_path
    model = YOLO(model_path)
        
    img_files = [ file for file in os.listdir(img_dir) if not file.startswith(".") ]
    for img_filename in img_files:
        print(img_filename)
        
        img = Image.open( os.path.join(img_dir, img_filename) )
        
        pred = inference(model, img, img_filename, patch_size, out_dir)

    
    count_ocls_from_output(out_dir)
        
    
    return
    

if __name__ == '__main__':
    main(sys.argv)
