import csv
import requests
import time
import os
import uuid

# import classify_ext_cntr as ct # step1
# import classify_photo_vs_drawing as ct # step2
import classify_step3 as ct # step3
# import classify_step4 as ct # step4

last_request_time = 0.0

def check_time():
    '''Checks when the last request was. If it was less than 3 seconds ago, it sleeps the difference.'''
    time_constraint = 3
    current_time = time.time()
    if (current_time - last_request_time) < time_constraint:
        time.sleep(time_constraint - (current_time - last_request_time))

class Manager:
    '''The class is for input and output manipulation.'''
    def __init__(self):
        self.CSV_IMG_ID = 'item'
        self.CSV_IMG_ADDR = 'imageAddr'
        self.classificatopn_path = r'result\classification'
        self.class_tool = ct.ClassificationTool(self.classificatopn_path)
        self.img_path_to_classify = "img_to_classify.jpeg"
        self.resized_img_path_to_classify = "resized_img_to_classify.jpeg"
        self.dummy_top_dist = [('', 0.0) for _ in range(3)]
        self.create_dir('result')
        self.create_dir(self.classificatopn_path)
        self.create_class_dirs()
        # self.open_result_csv()

    def create_class_dirs(self):
        # classes = ['library stamps', 'misaligned', 'multiple images', 'not images', 'ok', 'too big', 'too small'] # step1
        # classes =  ['painting', 'photo', 'photomontage'] # step2
        classes =  ['03_MONEY + POSTAGE STAMPS', 
            '04_GRAPHIC ORNAMENTS', '05_SCHEMES', '06_BUILDINGS', 
            '08_MODELS OF ARCHUTECTURE', '09_ARCHITECTURAL PLANS', '10_SCULPTURES IN THE ROUND', 
            '11_RELIEF + INTAGLIO', '12_BOOKS', '13_MACHINES, VEHICLES', '19_FURNITURE', '21_COINS, MEDALS', 
            '23_PUPPETS + TOYS', '24_THEATRE', '26_wallpapers, wallpapers designs, fabrics, fabrics designs', 
            '28_JEWELRY', '30_CLOTHING', '32_DESIGN', '34_EXHIBITIONS AND INSTALLATIONS', '35_sheet music', 
            '36_HANDWRITTEN TEXTS', '37_OTHER TEXTS', '38_OTHER', '39_PHOTOGRAMS', '40_ARIST STUDIOS', 
            '44_PHOTOS OF INTERIORS', '46_MURALS', '48_MOSAICS', '50_STAINED GLASS', '52_SCULPTURE + DESIGN', 
            '53_ARCHITECTURE + SCULPTURE', '54_INTERIORS WITH LOT OF ARTWORKS', '55_RELIEF + DESIGN', 
            '57_ other + architecture', '62_maps', '63_clocks', '66_fabric_bags', '67_fans', 
            '68_lamps', '69_vessels_of_all_kinds', 'no_classification'] # step3
        # classes = ['ad', 'not_ad'] # step4
    
        for cl in classes:
            class_path = os.path.join(self.classificatopn_path, cl)
            self.create_dir(class_path)
        return
    def open_result_csv(self, file_name:str):
        '''Opens the CSV file that stores the results of the pipeline.'''
        self.csv_to_write = open(os.path.join('result', f'{file_name}_result.csv'), 'w', encoding='utf-8')
        self.fieldnames = ['item', 'imageAddr', 'class1', 'prob1']#, 'class2', 'prob2', 'class3', 'prob3']
        self.writer = csv.DictWriter(self.csv_to_write, fieldnames=self.fieldnames)
        self.writer.writeheader()
        return
    
    def create_csv_entry(self, item:str, img_addr:str, top_dist_classes:list)->dict:
        '''Creates an ebtry to the result CSV file.'''
        return {
            'item' : item,
            'imageAddr' : img_addr,
            'class1' : top_dist_classes[0][0],
            'prob1' : top_dist_classes[0][1],
            # 'class2' : top_dist_classes[1][0],
            # 'prob2' : top_dist_classes[1][1],
            # 'class3' : top_dist_classes[2][0],
            # 'prob3' : top_dist_classes[2][1],
        }

    def create_dir(self, dir_name:str):
        '''Creates a directory with a given name, if it does not already exist.'''
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return
    
    def delete_img(self):
        '''Deletes the images that were created in the process of processing the input.'''
        if os.path.exists(self.resized_img_path_to_classify):
            os.remove(self.resized_img_path_to_classify)
        if os.path.exists(self.img_path_to_classify):
            os.remove(self.img_path_to_classify)
    

class CSVManager(Manager):
    def __init__(self):
        super().__init__()

    def process_csv(self, csv_path:str):
        '''Processes a single input CSV file and writes the results to the output CSV file.'''
        try:
            self.open_result_csv(os.path.basename(csv_path).split('.')[0])
            with open(csv_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    img_id = row[self.CSV_IMG_ID]
                    img_address = row[self.CSV_IMG_ADDR]

                    if img_address.startswith('http'):
                        database_img_id = img_id.split('Q')[-1]
                        img_path = database_img_id+'_'+self.img_path_to_classify
                        success = self.save_img(img_path, img_address)
                        if success:
                            top_dist_classes = self.class_tool.get_classification_rank(img_path)
                            csv_entry = self.create_csv_entry(img_id, img_address, top_dist_classes)
                            self.writer.writerow(csv_entry)
                        else:
                            csv_entry = self.create_csv_entry(img_id, img_address, self.dummy_top_dist)
                            self.writer.writerow(csv_entry)
                        
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    else:
                        top_dist_classes = self.class_tool.get_classification_rank(img_address)
                        csv_entry = self.create_csv_entry(img_id, img_address, top_dist_classes)
                        self.writer.writerow(csv_entry)
                
            self.csv_to_write.close()
        except Exception as e:
            print("An error occurred...", e)
            self.csv_to_write.close()
        return


    def work_with_csv(self, csvs:list[str]):
        '''Processes a list of input CSV files.'''
        # self.create_dir(r'result\classification_dist')
        for csv in csvs:
            print(csv)
            self.process_csv(csv)
        self.delete_img()
        return
    
    def save_img_unsafe(self, url:str, img_name:str)->bool:
        '''Downloads an image from the given URL.'''
        # headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"}
        check_time()
        time.sleep(1)
        response = requests.get(url)
        global last_request_time
        last_request_time = time.time()
        if response.ok:
            with open(img_name, "wb") as f:
                f.write(response.content)
        return response.ok
        
    def save_img(self, img_name:str, url:str)->bool:
        '''Calls a function that downloads an image from the URL, in case of an error enforces timeout (60 s) and tries again.'''
        try:
            response_ok = self.save_img_unsafe(url, img_name)
            return response_ok
        except Exception as e:
            print("Error occurred. The error is ", e)
            print("Let me try again...")
            time.sleep(60)
            response_ok = self.save_img_unsafe(url, img_name)
            return response_ok

class DIRManager(Manager):
    def __init__(self):
        super().__init__()
    

    def process_dir(self, dir_name:str):
        '''Processes a single directory, that contains images (PNG, JPEG, JPG), and writes the results to the output CSV file. 
        Ignores everything except for images.'''
        try:
            self.open_result_csv(os.path.basename(dir_name))
            imgs = os.listdir(dir_name)
            for img in imgs:
                img_path = os.path.join(dir_name, img)
                if (img_path.lower().endswith(('.png', '.jpg', '.jpeg'))):
                    print(img)
                    img_id = img
                    img_address = img_path

                    top_dist_classes = self.class_tool.get_classification_rank(img_address)
                    csv_entry = self.create_csv_entry(img_id, img_address, top_dist_classes)
                    self.writer.writerow(csv_entry)
                else:
                    print(f'Ignoring {img_path}. Not an image.')
                
            self.csv_to_write.close()
        except Exception as e:
            print("An error occurred...", e)
            self.csv_to_write.close()
        return


    def work_with_dir(self, dir_path:str):
        '''Expects a directory which contains directories which contain images. 
        Calls a function that processes a single directory for each directory. Ignores everything except for directories.'''
        dirs = os.listdir(dir_path)
        for d in dirs:
            d_path = os.path.join(dir_path, d)
            print(d_path)
            if os.path.isdir(d_path):
                print(d_path)
                self.process_dir(d_path)
            else:
                print(f'Ignoring {d_path}. Not a directory.')
        self.delete_img()
        return