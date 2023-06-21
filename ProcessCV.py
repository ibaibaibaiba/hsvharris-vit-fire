import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# TO DO - BENERIN RANGE HARRIS SAMA RANGE WARNA HSV NYA
# TES BETTER MAKE RGB APA HSV PAS TRAINING SAMA KLASIFIKASI

# COBA PAKE 2 RANGE YG BAGIAN CONTOUR NYA

# HSV CONVERSION

# define range of api in HSV
lower_fire1 = np.array([0,50,50])
upper_fire1 = np.array([10,255,255])
lower_fire1_5 = np.array([20,50,50])
upper_fire1_5 = np.array([40,255,255])
lower_fire2 = np.array([20,50,50])
upper_fire2 = np.array([40,255,255])
lower_fire2_5 = np.array([0,50,50])
upper_fire2_5 = np.array([10,255,255])

# define size object minimal
object_size = 32

def bgr2hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_img

def maskfire(mode,hsv_img): #minta hsv return hsv yg udh di mask range api sama mask binernya
    if mode == 1:
        mask = cv2.inRange(hsv_img, lower_fire1, upper_fire1) # mask biner
    elif mode == 2:
        mask = cv2.inRange(hsv_img, lower_fire1_5, upper_fire1_5) # mask biner
    elif mode == 3:
        mask = cv2.inRange(hsv_img, lower_fire2, upper_fire2) # mask biner
    elif mode == 4:
        mask = cv2.inRange(hsv_img, lower_fire2_5, upper_fire2_5) # mask biner
    masked_hsv = cv2.bitwise_and(hsv_img,hsv_img, mask= mask)
    return masked_hsv, mask

# HARRIS CORNER

# dapetin semua corner
def harrisCorner(masked_hsv_img):
    # Convert to grayscale
    mask_gray = cv2.cvtColor(masked_hsv_img, cv2.COLOR_BGR2GRAY)

    # Apply Harris corner detector
    dst = cv2.cornerHarris(mask_gray, 2, 3, 0.04)

    # Find corners
    corners = cv2.goodFeaturesToTrack(dst, 150, 0.01, 0) # PARAMETER HARRIS
    #if corners is None:
    #    corners = []

    return corners


# gambar corner di img plus itung angle, return img yg dibuletin
def gambarCorner(corners,masked_hsv_img,img_canvas):
    #mask_gray = cv2.cvtColor(masked_hsv_img, cv2.COLOR_BGR2GRAY)
    mask_gray = cv2.cvtColor(cv2.cvtColor(masked_hsv_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    topcorner = []
    # Loop through corners and calculate angle
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            sobelx = cv2.Sobel(mask_gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(mask_gray, cv2.CV_64F, 0, 1, ksize=5)
            angle = np.rad2deg(np.arctan2(sobely[int(y)][int(x)], sobelx[int(y)][int(x)]))
            if angle > 45 and angle < 135:
                if img_canvas is not None:
                    cv2.circle(img_canvas, (int(x), int(y)), 5, (0, 0, 255), 2)
                topcorner.append((int(x),int(y)))
            else:
                if img_canvas is not None:
                    cv2.circle(img_canvas, (int(x), int(y)), 5, (255, 0, 0), 1)
    #print(debugbanyakcorner)
    return img_canvas, topcorner

# DENOISE
def denoise(image,mode=1):
    if mode == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif mode == 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

# CONTOUR

def contourFire(mask):
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, hierarchy  = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #flattened_contours = np.concatenate(contours).reshape(-1, 2)
    #return flattened_contours
    #return np.concatenate(contours).flatten()
    return contours

def gambarContour(contours,img):
    if contours:
        cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    return img

def gambarContourBox(contours,image,mode=1):
    # Draw bounding boxes around the detected contours
    for contour in contours:
        # You can set a threshold for the contour area to filter out small false detections
        if cv2.contourArea(contour) > 20:
            if mode == 1:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            elif mode == 2:
                # yg ada kotak biasa, yg bawah kotak miring
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(image,[box],0,(0,255,0),2)
    return image

def mergeRegionBox(contours,corners,image,mode=1): # input array contour, array corner, gambar yg mau dicrop
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cnt in contours:
        # Check if the contour contains the predetermined point
        x, y, w, h = cv2.boundingRect(cnt)
        # for cnr in corners:
        #     dist = cv2.pointPolygonTest(cnt, cnr, False)
        #     if dist >= 0:
        #         #If the contour contains the point, draw it on the binary mask
        #         #if cv2.contourArea(cnt) > 20:
        #         cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
        if any(x <= px <= x+w and y <= py <= y+h for px, py in corners):
            #If the contour contains the point, draw it on the binary mask
            if mode == 1:
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            elif mode == 2:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(mask,[box],0,255,-1)
    
    final = cv2.bitwise_and(image,image,mask=mask)
    return final

def show(window_name, *images):
    combined = None
    for image in images:
        if combined is None:
            combined = image
        else:
            combined = np.hstack((combined, image))
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# MAIN sama output dll

def outputGambar(output_folder_path,file_name,img,folder,label):
    if label == 0:
        label = "fire"
    elif label == 1:
        label = "non_fire"
    output_file_path = os.path.join(os.path.join(os.path.join(output_folder_path,folder),label), file_name)
    cv2.imwrite(output_file_path, img)

def ProsesGambar(file_name,input_folder_path):
    # Check if the file is an image (JPEG or PNG)
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        print(file_name)

        # Load the image using cv2
        image_path = os.path.join(input_folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #image = cv2.resize(image,(256,256))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image here
        processed_image = image.copy()
        processed_image2 = image.copy()
        processed_image2_5 = image.copy()
        preliminary_image = image.copy()
        # ...

        # rgb to hsv
        processed_image = bgr2hsv(processed_image)
        processed_image2 = bgr2hsv(processed_image2)
        processed_image2_5 = bgr2hsv(processed_image2_5)
        show1 = processed_image.copy()
        # ...

        # mask hsv nya make limit api
        processed_image, fire_mask1 = maskfire(1,processed_image)
        processed_image2, fire_mask2 = maskfire(3,processed_image2)
        processed_image2_5, fire_mask2_5 = maskfire(4,processed_image2_5)
        show2 = processed_image.copy()
        # ...

        # cari corner
        corners = harrisCorner(processed_image)
        # if corner ada apa engga
        if len(corners) != 0:
            topcorners = []
        
            #print("banyak corner",len(corners))

            # gambar corner
            gambar_corner = processed_image.copy()
            gambar_corner, topcorners = gambarCorner(corners,processed_image,gambar_corner)
            preliminary_image, topcorners = gambarCorner(corners,processed_image,preliminary_image)
            # ...
            
            
            # denoise
            processed_image2 = denoise(processed_image2,1)
            processed_image2_5 = denoise(processed_image2_5,2)
            # ...
            
            # contour range 2
            # cek contour
            contours1 = contourFire(processed_image2)
            # gambar contour
            gambar_contour = processed_image2.copy()
            gambar_contour = gambarContour(contours1,gambar_contour)
            # ...
            # kotakin contour
            gambar_contour_box = processed_image2.copy()
            gambar_contour_box = gambarContourBox(contours1,gambar_contour_box)
            # ...
            

            # contour range 2_5
            # cek contour
            contours2 = contourFire(processed_image2_5)
            # gambar contour
            gambar_contour2 = processed_image2_5.copy()
            gambar_contour2 = gambarContour(contours2,gambar_contour2)
            # ...
            # kotakin contour
            gambar_contour_box2 = processed_image2_5.copy()
            gambar_contour_box2 = gambarContourBox(contours2,gambar_contour_box2,1)
            # ...

            # preliminary img
            contours = contours1+contours2
            preliminary_image = gambarContourBox(contours,preliminary_image,1)
            # ...

            #
            # ...

            #
            # ...
            

            # CROP BOX YG ADA TITIK MERAHNYA
            final_image = image.copy()
            final_image = mergeRegionBox(contours,topcorners,final_image,1)
            # ...

            # kotaku = image.copy()
            # kotaku = gambarContour(contours,kotaku)
            # show('kotaku',kotaku)


            #show(file_name,final_image,preliminary_image,gambar_corner,gambar_contour_box,gambar_contour_box2,show1)
            final_image_hsv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
            return final_image,preliminary_image,bgr2hsv(image),final_image_hsv,contours

        else :
            #print("banyak corner 0")
            # denoise
            processed_image2 = denoise(processed_image2)
            # ...
            
            # cek contour
            contours1 = contourFire(processed_image2)
            contours2 = contourFire(processed_image2_5)
            contours = contours1 + contours2
            # gambar contour
            final_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for cnt in contours:
                if cv2.contourArea(cnt) > 20:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(final_mask, (x, y), (x + w, y + h), 255, -1)
            #
            
            # ...
            #
            final_image = image.copy()
            final_image = cv2.bitwise_and(final_image,final_image,mask=final_mask)
            # ...
            
            #show('gada corner '+file_name,final_image,preliminary_image)
            final_image_hsv = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
            preliminary_image = image.copy()
            preliminary_image = gambarContourBox(contours,preliminary_image,1)

            return final_image,preliminary_image,bgr2hsv(image),final_image_hsv,contours
        

def getObject(img,cnt,corners): #input gambar sama satuan contour, output gambar 1 contour/kotak
    flag = False
    objek = None
    if len(corners) != 0:
        for cnr in corners:
            isInside = cv2.pointPolygonTest(cnt, cnr, False)
            if isInside >= 0:
                flag = True
                break
    if flag:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= object_size and h >=object_size:
            objek = img[y:y+h, x:x+w]
            return objek
        else:
            return None
    else:
        return None

def prosesObject(name,inputpath,outputpath,flagoutput):
    image = cv2.imread(os.path.join(inputpath, name), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #show('di main',image)

    img_range1 = img_range2 = img_range2_5 = image.copy()
    img_range1, _ = maskfire(1,img_range1)
    img_range2, _ = maskfire(3,img_range2)
    img_range2_5, _ = maskfire(4,img_range2_5)

    corners = harrisCorner(img_range1)

    img_range2 = denoise(img_range2,1)
    img_range2_5 = denoise(img_range2_5,2)

    contours1 = contourFire(img_range2)
    contours2 = contourFire(img_range2_5)
    contours = contours1 + contours2
    if len(corners) != 0:
        topcorners = []
        _, topcorners = gambarCorner(corners,img_range1,None)
        if len(contours) != 0:
            iterimage = 1
            for cnt in contours:
                if cv2.contourArea(cnt) > 20:
                    objek = getObject(image,cnt,topcorners) # keluarnya satu satu contour
                    if objek is not None and flagoutput:
                        objek = cv2.cvtColor(objek, cv2.COLOR_HSV2BGR)
                        nameoutput = f"{file_name} (object{iterimage}).jpg" 
                        outputimgpath = os.path.join(outputpath,nameoutput)
                        iterimage += 1
                        print(nameoutput)
                        cv2.imwrite(outputimgpath, objek)
                    elif objek is not None and not flagoutput:
                        show('output',objek)






if __name__ == "__main__":
    
    input_folder_path = ".\\Dataset\\final"
    output_folder_path = ".\\Dataset\\final"

    # input_testing = ".\\Dataset\\testing\\input" #testing
    # output_testing = ".\\Dataset\\final\\objects\\temporary" #testing

    #temporary object
    pure_objects_output_path = ".\\Dataset\\final\\objects\\temporary"
    pure_objects_output_path2 = ".\\Dataset\\final\\objects\\non_fire"

    pure_api = os.path.join(input_folder_path,"pure\\fire")
    #pure_api = .\Dataset\final\pure\fire
    pure_non = os.path.join(input_folder_path,"pure\\non_fire")

#     for file_name in os.listdir(pure_api):
#         finalimg, debugimg, imghsv, finalimghsv = ProsesGambar(file_name,pure_api)
#         #show('akhir',finalimg,debugimg,imghsv,finalimghsv)

#         outputGambar(output_folder_path,file_name,finalimg,"masked",0)
#         outputGambar(output_folder_path,file_name,finalimghsv,"maskedhsv",0)
#         outputGambar(output_folder_path,file_name,imghsv,"hsv",0)
#         outputGambar(output_folder_path,file_name,debugimg,"debug",0)
        
        
        
#    for file_name in os.listdir(pure_non):
#        finalimg, debugimg, imghsv, finalimghsv = ProsesGambar(file_name,pure_non)
#         #show('akhir',finalimg,debugimg,imghsv,finalimghsv)
        

#         outputGambar(output_folder_path,file_name,finalimg,"masked",1)
#         outputGambar(output_folder_path,file_name,finalimghsv,"maskedhsv",1)
#         outputGambar(output_folder_path,file_name,imghsv,"hsv",1)
#         outputGambar(output_folder_path,file_name,debugimg,"debug",1)
    
    # proses object
    for file_name in os.listdir(pure_api):
        prosesObject(file_name,pure_api,pure_objects_output_path,False)

    for file_name in os.listdir(pure_non):
        prosesObject(file_name,pure_non,pure_objects_output_path2,False)


    # for file_name in os.listdir(input_testing):
    #     #finalimg, debugimg, imghsv, finalimghsv, listcontours = ProsesGambar(file_name,input_testing)
    #     #cropContourOutput(output_testing,imghsv,listcontours)
    #     prosesObject(file_name,input_testing,output_testing,True)
    #     break

    #     # outputGambar(output_folder_path,file_name,finalimg,"masked",1)
    #     # outputGambar(output_folder_path,file_name,finalimghsv,"maskedhsv",1)
    #     # outputGambar(output_folder_path,file_name,imghsv,"hsv",1)
    #     # outputGambar(output_folder_path,file_name,debugimg,"debug",1)

    