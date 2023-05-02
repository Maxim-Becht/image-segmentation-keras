import os
from PIL import Image
import numpy as np

clear = lambda: os.system('cls')

folder_dir = "./datasets/CamVid/test/labeled/"
new_dir = "./new_test/"

# key=original rgb value; value=tuple(label_id, label_name, label_pixelcount)
annotations = { 
    '[ 64,128, 64]': [0, "animal", 0],
    '[192,  0,128]': [1, "archway", 0],
    '[  0,128,192]': [2, "bicyclist", 0],
    '[  0,128, 64]': [3, "bridge", 0],
    '[128,  0,  0]': [4, "building", 0],
    '[ 64,  0,128]': [5, "car", 0],
    '[ 64,  0,192]': [6, "cart_luggage_pram", 0],
    '[192,128, 64]': [7, "child", 0],
    '[192,192,128]': [8, "column_pole", 0],
    '[ 64, 64,128]': [9, "fence", 0],
    '[128,  0,192]': [10, "lane_mkgs_driv", 0],
    '[192,  0, 64]': [11, "lane_mkgs_non_driv", 0],
    '[128,128, 64]': [12, "misc_text", 0],
    '[192,  0,192]': [13, "motorcycle_scooter", 0],
    '[128, 64, 64]': [14, "other moving", 0],
    '[ 64,192,128]': [15, "parking_block", 0],
    '[64,64, 0]': [16, "pedestrian", 0],
    '[128, 64,128]': [17, "road", 0],
    '[128,128,192]': [18, "road_shoulder", 0],
    '[  0,  0,192]': [19, "sidewalk", 0],		
    '[192,128,128]': [20, "sign_symbol", 0],	
    '[128,128,128]': [21, "sky", 0],	
    '[ 64,128,192]': [22, "suv_pickup_truck", 0],	
    '[ 0, 0,64]': [23, "traffic_cone", 0],
    '[ 0,64,64]': [24, "traffic_light", 0], #
    '[192, 64,128]': [25, "train", 0],
    '[128,128,  0]': [26, "tree", 0],	
    '[192,128,192]': [27, "truck_bus", 0],
    '[64, 0,64]': [28, "tunnel", 0],		
    '[192,192,  0]': [29, "vegetation_misc", 0],
    '[0,0,0]': [30, "void", 0],
    '[ 64,192,  0]': [31, "wall", 0]
}


i = 0
for img_name in os.listdir(folder_dir):
    if img_name.endswith(".png"):
        i+=1

        if i <= 151:     # x training images already done, skip x
            continue

        # if i <= x:     # x training images already done, skip x
        #     continue
        # create the full input path and read the file
        input_path = os.path.join(folder_dir, img_name)
        img = Image.open(input_path, 'r')
        new_img = Image.new('RGB', (img.width, img.height))
        
        # gather img properties
        width, height = img.size
        assert(img.mode == 'RGB')
        channels = 3

        for y in range(img.height):
            for x in range(img.width):
                pixel = img.getpixel((x, y))
                pixel = np.array2string(np.asarray(pixel), separator=',')

                # if entry doesnt exist (artifacts or ignore label case) Seq05VD_f02610_L.png fails here -> Artifacts!
                try:
                    annotations[pixel][2] += 1  # pixel counter++ for label associated with color
                except KeyError:
                    pixel = '[0,0,0]' # -> "void"
                    annotations[pixel][2] += 1

                
                annotations[pixel][2] += 1
                id, name, counter = annotations[pixel]
                #print("id:{}, name:{}, counter:{}".format(id, name, counter))
                new_img.putpixel((x, y), (id, id, id))

                
                id, name, counter = annotations[pixel]
                #print("id:{}, name:{}, counter:{}".format(id, name, counter))
                new_img.putpixel((x, y), (id, id, id))

        new_img.save("{}{}".format(new_dir, img_name))

    clear()
    print("image: {} number: {}".format(img_name, i))
    # print("+++Counters+++")
    # for anno in annotations:
    #     print("{}: {}, count={}".format(annotations[anno][0], annotations[anno][1], annotations[anno][2]))


print("===END===")
print("+++Counters+++")
for anno in annotations:
    print("{}: {}, count={}".format(annotations[anno][0], annotations[anno][1], annotations[anno][2]))