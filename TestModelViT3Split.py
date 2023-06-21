import cv2
import numpy as np
import os
from ProcessCV import show, harrisCorner, maskfire, denoise, contourFire, gambarCorner

import tensorflow as tf

input_size = 256  # Update the input size according to your model

def predict(image):
    image = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    image = image / 255.0  # Normalize pixel values
    input_tensor = tf.convert_to_tensor(image)

    result = model(inputs=input_tensor, training=False)
    predicted_class = (result.numpy() > 0.5).astype("int32")
    certainty = np.round(result.numpy(), 3)

    return predicted_class, certainty

object_size = 32

def getObjectLocation(cnt,corners):
    flag = False
    location = []
    if len(corners) != 0:
        for cnr in corners:
            isInside = cv2.pointPolygonTest(cnt, cnr, False)
            if isInside >= 0:
                flag = True
                break
    if flag:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= object_size and h >=object_size:
            location = [(x,y,w,h)]
            return location
        else:
            return location
    return location

def prosesObjectList(imgname,imgpath):
    image = cv2.imread(os.path.join(imgpath, imgname), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    img_range1 = img_range1_5 = img_range2 = img_range2_5 = image.copy()
    img_range1, _ = maskfire(1,img_range1)
    img_range1_5, _ = maskfire(2,img_range1_5)
    img_range2, _ = maskfire(3,img_range2)
    img_range2_5, _ = maskfire(4,img_range2_5)

    corners1 = harrisCorner(img_range1)
    corners2 = harrisCorner(img_range1_5)
    if corners1 is not None and corners2 is not None:
        corners = np.concatenate((corners1, corners2))
    elif corners1 is not None and corners2 is None:
        corners = corners1
    elif corners2 is not None and corners1 is None:
        corners = corners2
    elif corners1 is None and corners2 is None:
        corners = []
    #corners = corners1 + corners2

    img_range2 = denoise(img_range2,1)
    img_range2_5 = denoise(img_range2_5,2)

    contours1 = contourFire(img_range2)
    contours2 = contourFire(img_range2_5)
    contours = contours1 + contours2
    #print("Contours 1:",len(contours1))
    #print("Contours 2:",len(contours2))
    #print("Contours:",len(contours))

    objectLocationList = []

    if len(corners) != 0:
        _, topcorners1 = gambarCorner(corners1,img_range1,None)
        _, topcorners2 = gambarCorner(corners2,img_range1_5,None)
        topcorners = topcorners1 + topcorners2
        if len(contours) != 0:
            for cnt in contours:
                if cv2.contourArea(cnt) > 20:
                    objeklocation = getObjectLocation(cnt,topcorners)
                    if len(objeklocation) != 0:
                        objectLocationList.append(objeklocation)
    return objectLocationList

def writeText(img,location,text):
    
    x, y, _, h = location[0]

    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    text = str(text)
    text_color = (255, 0, 0)

    text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
    text_x = x
    text_y = y + h - text_size[1]

    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)














#image_path = '.\\Dataset\\final\\xmlroi\\fire\\00000.jpg'
image_path = '.\\Dataset\\final\\xmlroi\\non_fire\\non_fire (2).jpg'

#test_folder = ".\\Dataset\\final\\pure\\fire"
test_folder = ".\\Dataset\\testing\\input"

image_folder1 = ".\\Dataset\\final\\pure\\fire"
image_folder2 = ".\\Dataset\\final\\pure\\non_fire"
output_folder1 = ".\\Dataset\\final\\prediksi di dataset\\fire"
output_folder2 = ".\\Dataset\\final\\prediksi di dataset\\non_fire"

testingFolder = ".\\Tes Model\\Input"
testingOutputFolder = ".\\Tes Model\\"

belumdiliatmodelfire = ".\\Dataset\\belumdiliatmodel\\fire"
belumdiliatmodelnon = ".\\Dataset\\belumdiliatmodel\\non_fire"
belumdiliatmodel = ".\\Dataset\\belumdiliatmodel\\0 hasil model"

def prediktinator(input_folder,output_folder,modeprediksi):
    os.makedirs(os.path.join(output_folder,"adaapi"), exist_ok=True)
    os.makedirs(os.path.join(output_folder,"gaadaapi"), exist_ok=True)
    for file_name in os.listdir(input_folder):
        listObject = prosesObjectList(file_name,input_folder)
        image = cv2.imread(os.path.join(input_folder,file_name))
        print(file_name)
        ada_api = False

        for obj in listObject:
            x = obj[0][0]
            y = obj[0][1]
            w = obj[0][2]
            h = obj[0][3]
            img_obj = image[y:y+h, x:x+w]

            
            prediction, certainty = predict(img_obj)
            #prediction = 0
            if modeprediksi == 0:
                if prediction == 0:
                    prediction = "Fire"
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    writeText(image,obj,prediction+str(certainty))
                    ada_api = True
                else:
                    prediction = "Non_Fire"
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    writeText(image,obj,prediction+str(certainty))
            elif modeprediksi == 1:
                prediction = np.argmax(prediction)
                if prediction == 0:
                    prediction = "Fire"
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    writeText(image,obj,prediction+str(certainty))
                    ada_api = True
                else:
                    prediction = "Non_Fire"
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    writeText(image,obj,prediction+str(certainty))

        #show("asd", image)
        if ada_api:
            cv2.imwrite(os.path.join(os.path.join(output_folder,"adaapi"),os.path.basename(input_folder)+file_name),image)
        else:
            cv2.imwrite(os.path.join(os.path.join(output_folder,"gaadaapi"),os.path.basename(input_folder)+file_name),image)
        ada_api = False


if __name__ == "__main__":
    # prediktinator(image_folder1,output_folder1)
    # prediktinator(image_folder2,output_folder2)

    model_name = "final_model5"
    model_path = os.path.join(".\\Saved Model",model_name) # JGN LUPA BUAT FOLDERNYA
    model = tf.saved_model.load(model_path)


    #prediktinator(testingFolder,os.path.join(testingOutputFolder,"final_model5")) # JGN LUPA GANTI NAMANYA
    os.makedirs(os.path.join(belumdiliatmodel,model_name), exist_ok=True) 

    prediktinator(belumdiliatmodelfire,os.path.join(belumdiliatmodel,model_name),0)
    prediktinator(belumdiliatmodelnon,os.path.join(belumdiliatmodel,model_name),0)