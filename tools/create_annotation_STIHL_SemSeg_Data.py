import os
import threading
import time
import datetime
from PIL import Image
import PIL
import numpy as np
#from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Pool

clear = lambda: os.system('cls')

#folder_dir = "./data/datasets/STIHL_SemSeg_Data/train/color_mask/"
folder_dir = "./data/datasets/STIHL_SemSeg_Data/train/color_masks_simplyfied/"
#folder_dir = "./data/datasets/STIHL_SemSeg_Data/train/color_masks_simplyfied_3/"

#new_dir = "./new_STIHL_SemSeg_Data_train/"
new_dir = "./new_STIHL_SemSeg_Data_train_simplyfied/"
#new_dir = "./new_STIHL_SemSeg_Data_train_simplyfied_3/"

#key=original rgb value; value=tuple(label_id, label_name, label_pixelcount)

# annotations = { 
#     '(0,0,0)': [0, "Unlabeled", 0],   #TODO: create annotations for all classes
#     '(0,255,0)': [1, "Lawn", 0],   
#     '(0,64,255)': [2, "Obstacle", 0],
# }

annotations = { 
    '(0,0,0)': [0, "Unlabeled", 0],   
    '(0,255,0)': [1, "Lawn", 0],   
    '(0,64,255)': [2, "Obstacle", 0],
}

# annotations = { 
#     '(0,0,0)': [0, "Unlabeled", 0],   
#     '(0,255,0)': [1, "Lawn", 0],   
#     '(0,64,255)': [2, "Obstacle", 0],
#     '(0,32,0)': [3, "Flat", 0],
# }

def operation(img_name):
    if img_name.endswith(".png"):
        # create the full input path and read the file
        input_path = os.path.join(folder_dir, img_name)

        img = Image.open(input_path, 'r').convert('RGB')
        new_img = Image.new('RGB', (img.width, img.height))
        
        assert(img.mode == 'RGB')

        data = img.load()

        for x in range(0, img.size[0]):
            for y in range(0, img.size[1]):
                pixel = data[x,y]
                pixel = '('+','.join(str(x) for x in pixel)+')'

                id = 255 # -> "White pixel(s) indicates failure(s)"
                try:
                    id = annotations[pixel][0]
                    #annotations[pixel][2] += 1 uncomment for class pixel counter (not thread safe -> not 100% accurate, when threading)
                except KeyError:
                    print("KeyError in: ", img_name)
                    
                new_img.putpixel((x, y), (id, id, id))
        new_img.save("{}{}".format(new_dir, img_name))

    #clear()
    #print("image: ", img_name)


def print_results():
    print("===END===")
    print("+++Timers+++")
    print("{} - {} time={}".format(datetime_start, datetime_end, datetime.timedelta(seconds=end_time-start_time)))
    print("+++Counters+++")
    for anno in annotations:
        print("{}: {}, count={}".format(annotations[anno][0], annotations[anno][1], annotations[anno][2]))



def process(items, start, end):                                                 
    for item in items[start:end]:                                               
        try:                                                                    
            operation(item)                                              
        except Exception as e:                                                       
            print('error with item: ', e)                                            


def split_processing(items, num_splits=8):                                      
    split_size = len(items) // num_splits                                       
    threads = []                                                                
    for i in range(num_splits):                                                 
        # determine the indices of the list this thread will handle             
        start = i * split_size                                                  
        # special case on the last chunk to account for uneven splits           
        end = None if i+1 == num_splits else (i+1) * split_size                 
        # create the thread                                                     
        threads.append(                                                         
            threading.Thread(target=process, args=(items, start, end)))         
        threads[-1].start() # start the thread we just created                  

    # wait for all threads to finish                                            
    for t in threads:                                                           
        t.join()                                                                

print(np.version.version)
datetime_start = datetime.datetime.now()
start_time = time.time()
split_processing(os.listdir(folder_dir))
end_time = time.time()
datetime_end = datetime.datetime.now()
print_results()