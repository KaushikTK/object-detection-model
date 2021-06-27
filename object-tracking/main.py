import cv2

"""

use appropriate tracker depending on the requirement

'BOOSTING' : cv2.TrackerBoosting_create()
'MIL' : cv2.TrackerMIL_create()
'KCF' : cv2.TrackerKCF_create()
'TLD' : cv2.TrackerTLD_create()
'MEDIANFLOW' : cv2.TrackerMedianFlow_create()
'CSRT' : cv2.TrackerCSRT_create()
'MOSSE' : cv2.TrackerMOSSE_create()

"""

tracker = cv2.TrackerKCF_create()
initBB = None
camera = cv2.VideoCapture(0)


def track(img):
    global tracker
    
    (success, box) = tracker.update(img)

    if success:
        # get bounding box
        (x, y, w, h) = [int(v) for v in box]

        # draw bounding box
        cv2.rectangle(img, (x,y), (w+x,h+y), (0,0,255), 1)

    return img


while True:
    ret, img = camera.read()

    # do something with the image frame
    if initBB:
        track(img)

    cv2.imshow('tracking window', img)

    key = cv2.waitKey(10)  & 0xFF
    if key == ord('q'): break
    
    if key == ord('s'):
        initBB = cv2.selectROI('selection window', img, True, False)
        tracker.init(img, initBB)

        
        # destroy the selection window after the selection
        if(initBB): cv2.destroyWindow('selection window')

camera.release()
cv2.destroyAllWindows()
