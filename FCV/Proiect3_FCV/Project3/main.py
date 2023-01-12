import cv2
import numpy as np
import glob
import os

images = glob.glob("./000*.jpg")


print(images)

for image in images:
    image_name = image[2:8]
    print(image_name)

    os.makedirs("./" + image_name, exist_ok=True)

    img = cv2.imread(image)
    kernel = np.ones((5,5),np.uint8)

    # Transform image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing a inverse binary thresholding with a thresh value of 120.
    # blackens img, not so visible white chars
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    # blackens img, visible white chars
    _, thresh_characters = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    # cv2.imshow('dilation', thresh)
    # cv2.waitKey(0)

    # Dilating letters
    if img.shape[0] > 1000:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,2))
    else:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    
    dilation = cv2.dilate(thresh, dilate_kernel, iterations = 7)
    # cv2.imshow('dilation', dilation)
    # cv2.waitKey(0)

    # Find contours
    contours,_ = cv2.findContours(dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = 0
    img_copy = img.copy()
    for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # discard those contours which have an area smaller than 10000 pixels.
            if h * w > 10000:
                cnt += 1

                # Getting the lines
                # find the coordinates of a bounding box
                rect = cv2.minAreaRect(c)
                (x1, y1), (height, width), angle = rect
                print('Width: ' + str(width), 'Height: ' + str(height))
                print(angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                src = box.astype("float32")
                print(box)

                # perform a perspective transform and dewarp the line. This way I can detect lines
                # of handwriting even if they are not straight
                dst = np.array([[0, 0],
                                [width - 1, 0],
                                [width - 1, height - 1],
                                [0, height - 1]], dtype = "float32")
                M = cv2.getPerspectiveTransform(src, dst)
                warped_line = cv2.warpPerspective(img, M, (int(width), int(height)))
                write_path = os.path.join("./"+image_name, image_name+ '_line_'+ str(cnt) + '.jpg')
                if angle < 1:
                    warped_line = cv2.rotate(warped_line, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    # cv2.imshow('dilation', warped_line)
                    # cv2.waitKey(0)
                # Saving the lines to file
                cv2.imwrite(write_path, warped_line)

                # --------------------- Processing to get characters ----------------------------
                warped_line_gray = cv2.cvtColor(warped_line, cv2.COLOR_BGR2GRAY)
                _, warped_line_thresh = cv2.threshold(warped_line_gray, 130, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

                # Eroding/Dilating letters
                # eroding step, in order to get hold of individual characters
                kernel = np.ones((2,1),np.uint8)
                eroded_line = cv2.erode(warped_line_thresh,kernel,iterations=1)
                eroded_image = cv2.erode(thresh_characters,kernel,iterations=1)
                # cv2.imshow('dilation', eroded_image)
                # cv2.waitKey(0)

                # dilation
                if img.shape[0] > 1000:
                   kernel = np.ones((3,3), np.uint8)
                else:
                   kernel = np.ones((1,1), np.uint8)

                dilated_line = cv2.dilate(eroded_line, kernel, iterations=1)
                dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

                ctrs, _ = cv2.findContours(dilated_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #Sorting contours
                sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

                for _, ctr in enumerate(sorted_ctrs):
                    x, y, w, h = cv2.boundingRect(ctr)
                    # if int(h) > int(height/5):
                    cv2.rectangle(warped_line, (x, y),(x + w, y + h),(255, 0, 255), 1)
                
                write_path = os.path.join("./"+image_name, image_name+ '_line_characters_'+ str(cnt) + '.jpg')
                cv2.imwrite(write_path,warped_line)

                # cv2.RETR_EXTERNAL instead of cv2.RETR_TREE. This way I avoid nding contours in contours.
                ctrs, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Sorting contours - to draw them on the image in the right order, from left to right.
                sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

                for _, ctr in enumerate(sorted_ctrs):
                    x, y, w, h = cv2.boundingRect(ctr)
                    # if int(h) > int(height/5):
                    cv2.rectangle(img_copy, (x, y),(x + w, y + h),(90, 0, 255), 1)
                
        

                # Drawing contours on line images and on the original image
                cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)

    cv2.imshow('img', img_copy)
    cv2.waitKey(0)
    cv2.imwrite('bb_' + image_name + '.jpg', img_copy)
