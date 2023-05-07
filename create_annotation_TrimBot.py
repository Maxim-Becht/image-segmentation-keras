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

folder_dir = "./data/datasets/TrimBot/training/labeled/"
new_dir = "./new_TrimBot_training/"

# key=original rgb value; value=tuple(label_id, label_name, label_pixelcount)
annotations = { 
    '(0,0,0)': [0, "Unknown", 0],       # 2D only - pixels to be ignored
    '(0,204,0)': [1, "Grass", 0],
    '(77,128,0)': [2, "Ground", 0],
    '(179,204,230)': [3, "Pavement", 0],
    '(128,128,0)': [4, "Hedge", 0],
    '(0,179,179)': [5, "Topiary", 0],
    '(230,0,0)': [6, "Rose", 0],
    '(51,51,230)': [7, "Obstacle", 0],
    '(77,179,26)': [8, "Tree", 0],
    '(26,26,26)': [9, "Background", 0], # 2D only - pixels from outside of garden
}

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

                id = 0 # -> "Unknown"
                try:
                    id = annotations[pixel][0]
                    #annotations[pixel][2] += 1
                except KeyError:
                    print("KeyError in: ", img_name)
                    
                new_img.putpixel((x, y), (id, id, id))
        new_img.save("{}{}".format(new_dir, img_name))

    #clear()
    #print("image: ", img_name)


def print_results():
    print("===END===")
    print("+++Timers+++")
    print("{} - {} time={}".format(start_time, end_time, datetime.timedelta(seconds=end_time-start_time)))
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
start_time = time.time()
split_processing(os.listdir(folder_dir))
end_time = time.time()
print_results()