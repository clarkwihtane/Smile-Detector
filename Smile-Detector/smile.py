import cv2


face_cascade_path = 'cascades/haarcascade_frontalface_default.xml'
eye_cascade_path = 'cascades/haarcascade_eye.xml'
smile_cascade_path= 'cascades/haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
smile_cascade=cv2.CascadeClassifier(smile_cascade_path)

def detection(grayscale, img):
    face = face_cascade.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
        eye = eye_cascade.detectMultiScale(ri_grayscale, 1.2, 18) 
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
        smile = smile_cascade.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
            cv2.putText(ri_color, 'guluyor', (x_smile, y_smile-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 130), 2, cv2.LINE_AA)
    return img 

vc = cv2.VideoCapture(0) 

while True:
    _, img = vc.read() 
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    final = detection(grayscale, img) 
    cv2.imshow('Video', final) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

vc.release() 
cv2.destroyAllWindows() 
# Relesing & destroying the windows


