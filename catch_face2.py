import cv2
import os

def Catch_PIC_FROM_VIDEO(catech_pic_num,path_name):

    capture = cv2.VideoCapture(0)

    classifier = cv2.CascadeClassifier(r'C:\Users\Asus\Desktop\app\bilibili\opencv\build\etc\haarcascades\haarcascade_frontalface_alt.xml')

    color = (0,255,0)

    num = 0

    while capture.isOpened():
        ok,frame = capture.read()
        #可有可无
        if not ok:
            break

        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faceRects = classifier.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
        if len(faceRects) > 0:
            # print('face detected')
            # color = (255, 255, 255)  # 白
            
            for faceRect in faceRects:
                x, y ,w,h= faceRect
                
                if not os.path.exists(path_name):
                    os.mkdir(path_name)
                img_name = '%s/%d.jpg' % (path_name,num)
                image = frame[y-10: y+h+10, x-10: x+w+10]
                cv2.imwrite(img_name,image)

                num += 1
                if num >(catech_pic_num):
                    break
                
                cv2.rectangle(frame, (x-10,y-10),(x+w+10,y+h+10),color,2)

                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame,'num:%d' % (num), (x+30, y+30),font, 1 ,(255,0,255),4)

        if num >(catech_pic_num):
            break
        
        cv2.imshow("hehe",frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    Catch_PIC_FROM_VIDEO(1000,r'C:\Users\Asus\Desktop\app\bilibili\data\baba')