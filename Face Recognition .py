from PIL import Image
import os
import numpy as np
import cv2

cascPath = "C:\\Users\\OmerF\\anaconda3\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascPath)

font = cv2.FONT_HERSHEY_SIMPLEX

recognizer = cv2.face.LBPHFaceRecognizer_create()
names = []
def data_olustur():
    video_capture = cv2.VideoCapture(0)
    face_id=input('\n bir isim giriniz ==>  ')
    i=0
    while True:
        
        ret, frame = video_capture.read()
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(175,175),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    
      
        for (x, y, w, h) in faces:
            i=i+1
            filename = 'C:\\Users\\OmerF\\.spyder-py3\\fotolar\\'+ str(face_id) + '.' + str(i)+'.jpg'
            print("i =",i)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w] 
            cv2.imwrite(filename,gray[y:y+h, x:x+w])  
            
            
        cv2.putText(frame,'Yuz Taramasindan cikmak icin q basiniz..',(0, 30), font, 1,(255,0,0),2)
        cv2.imshow('Video', frame)
    
        if cv2.waitKey(1) & i==10 :
          break
    
    
    video_capture.release()
    cv2.destroyAllWindows()




def egitmen():
    path ='C:\\Users\\OmerF\\.spyder-py3\\fotolar\\'
    def getImagesAndLabels(path):
    
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []
        farkli=''
        say=0
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img,'uint8')
    
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            farkli=os.path.split(imagePath)[-1].split(".")[0]
            say +=1
            names[id]=farkli
            faces = faceCascade.detectMultiScale(img_numpy)
    
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
    
        return faceSamples,ids
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    
    
    recognizer.write('trainer.yml') 
    
    
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))




def dongu():
    recognizer.read('C:\\Users\\OmerF\\.spyder-py3\\trainer.yml')   
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    
    
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while True:
        ret, img =cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
    
        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(int(minW), int(minH)),)
        for (x, y, w, h) in faces:
    
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            print(id,confidence)
    
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "bilinmiyor"
                if id== "bilinmiyor" :
                    data_olustur()
                    egitmen()
                    dongu()
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            print(id)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
    
        cv2.imshow('camera', img)
        
        if cv2.waitKey(1) & len(faces) ==0 :
          break
      
    cam.release()
    cv2.destroyAllWindows()






while True:
    sayac=0
    a=('C:\\Users\\OmerF\\.spyder-py3\\trainer.yml')
    if a == 'C:\\Users\\OmerF\\.spyder-py3\\trainer.yml':
        path ='C:\\Users\\OmerF\\.spyder-py3\\fotolar\\'
        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
        for imagePath in imagePaths:
            sayac +=1
        if sayac == 0:
            data_olustur()
            egitmen()
            dongu()
    dongu()