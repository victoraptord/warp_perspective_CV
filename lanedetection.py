import cv2 as cv
import numpy as np

vidcap = cv.VideoCapture("highway45.mp4")
success, image = vidcap.read()

def nothing(x):
    pass

cv.namedWindow("Trackbars")

cv.createTrackbar("L - H", "Trackbars", 0,255, nothing)
cv.createTrackbar("L - S", "Trackbars", 0,255, nothing)
cv.createTrackbar("L - V", "Trackbars", 200,255, nothing)
cv.createTrackbar("U - H", "Trackbars", 255,255, nothing)
cv.createTrackbar("U - S", "Trackbars", 50,255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255,255, nothing)

while success :
    success, image = vidcap.read()
    frame = cv.resize(image, (640,480))

    ##choosing points for perspective transformation
    tl = (252, 337)
    bl = (122, 422)
    tr = (420,337)
    br = (550, 422)

    cv.circle(frame, tl, 5, (0, 0, 255), -1)
    cv.circle(frame, bl, 5, (0, 0, 255), -1)
    cv.circle(frame, tr, 5, (0, 0, 255), -1)
    cv.circle(frame, br, 5, (0, 0, 255), -1)

    # Connect circles with lines to their closest two points
    cv.line(frame, tl, bl, (0, 0, 255), 1)
    cv.line(frame, tl, tr, (0, 0, 255), 1)
    cv.line(frame, tr, br, (0, 0, 255), 1)
    cv.line(frame, bl, br, (0, 0, 255), 1)

    ## Applying perspective transformation
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0,0], [0,480], [640, 0], [640,480]])

    #Matrix to warp the image for birdseye window
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv.warpPerspective(frame, matrix, (640,480))

    ### Object Detection
    # Image Threshold
    hsv_transformed_frame = cv.cvtColor(transformed_frame, cv.COLOR_BGR2HSV)

    l_h = cv.getTrackbarPos("L - H", "Trackbars")
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv.inRange(hsv_transformed_frame, lower, upper)

    #Histogram
    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    #Sliding Window
    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y>0:
    #Left threshold
        img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                lx.append(left_base-50 + cx)
                left_base = left_base-50 + cx

    # Right threshold
        img = mask[y - 40:y, right_base - 50:right_base + 50]
        contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                lx.append(right_base - 50 + cx)
                left_base = right_base - 50 + cx

        cv.rectangle(msk, (left_base-50,y), (left_base+50, y-40), (255,255,255), 2)
        cv.rectangle(msk, (right_base-50, y), (right_base+50, y-40), (255, 255, 255), 2)
        y -= 40

    cv.imshow("Original", frame)
    cv.imshow("Bird's Eye View", transformed_frame)
    cv.imshow("Lane Detection - Image Thresholding", mask)
    cv.imshow("Lane Detection - Sliding Windows", msk)

    if cv.waitKey(50) == ord('q'):
        break