import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D

def extractareaofinterest(img):
    # this is a function to isolate the AOI from training images
    # we extract the QR code as the area of interest,
    # we shall use the QR code to determine the corners
    # and hence the orientation
    # This function returns the image with only the AOI(i.e.QR Code)
    # and a polygon which is the array containing the corners of
    # the AOI. External Noise like carpet texture background is removed

    # First we apply some filters and thresholding
    # to easily identify the AOIs

    # convert the image to greyscale for easy rendering
    gwashBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply blur to smoothen edges and threshold
    # to convert the image to binary bits
    blurred = cv2.GaussianBlur(gwashBW, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY)

    # square image kernel used for erosion
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    # this is for further removing small noises and holes in the image
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel)
    closing = cv2.erode(closing, kernel, iterations=4)

    # refines all edges in the binary image
    closing = cv2.dilate(closing, None, iterations=4)

    # find contours with simple approximation
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # arrays to store areas of contours
    areaarray = []

    for contour in contours:
        areaarray.append(cv2.contourArea(contour))

    # sort areas in descending and pick second largest
    areaarraysort = list(areaarray)
    areaarraysort.sort(reverse=True)
    areamaxindex = areaarraysort[1]
    areamaxindex = areaarray.index(areamaxindex)

    # mask used to extract the contour and store it
    mask = np.zeros(gwashBW.shape, dtype="uint8")*255
    cv2.drawContours(mask, [contours[areamaxindex]], -1, 255, -1)

    # extract the polygon
    arclength = cv2.arcLength(contours[areamaxindex], True)
    polygon = cv2.approxPolyDP(contours[areamaxindex], arclength * 0.02, 5)

    # the image is black-washed to make it easier to extract value from in the next step
    image = cv2.bitwise_and(gwashBW, gwashBW, mask=mask)*255
    return image, polygon

def extractqrsource(img):
    # this is a function to isolate the AOI from the source image
    # we extract the QR code as the area of interest,
    # we shall use the QR code to determine the corners
    # and hence the orientation.
    # This function returns the image with only the AOI(i.e.QR Code)
    # and a polygon which is the array containing the corners of
    # the AOI. External Noise like carpet texture background is removed

    # First we apply some filters and thresholding
    # to easily identify the AOIs

    # convert the image to greyscale for easy rendering
    gwashBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply blur to smoothen edges and threshold to convert the image to binary bits
    blurred = cv2.GaussianBlur(gwashBW, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blurred, 127, 255,cv2.THRESH_BINARY)

    # square image kernel used for erosion
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    # this is for further removing small noises and holes in the image
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE,kernel)
    closing = cv2.erode(closing, kernel, iterations=4)
    # refines all edges in the binary image
    closing = cv2.dilate(closing, None, iterations=4)

    # find contours with simple approximation
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # arrays to store areas of contours
    areaarray = []

    for contour in contours:
        areaarray.append(cv2.contourArea(contour))

    # sort areas in descending and pick second largest
    areaarraysort = list(areaarray)
    areaarraysort.sort(reverse=True)
    areamaxindex = areaarraysort[1]
    areamaxindex = areaarray.index(areamaxindex)

    # mask used to extract the contour and store it
    mask = np.zeros(gwashBW.shape, dtype="uint8")*255
    cv2.drawContours(mask, [contours[areamaxindex]], -1, 255, -1)

    #extract the polygon
    arclength = cv2.arcLength(contours[areamaxindex], True)
    polygon = cv2.approxPolyDP(contours[areamaxindex], arclength * 0.02, 5)

    # the image is black-washed to make it easier to extract value from in the next step
    image = cv2.bitwise_and(gwashBW, gwashBW, mask=mask)
    return image, polygon

def getqrpoints(img):
    # function to find the corners of the QR; The QR has 3 points which
    # are the squares we can extract and the fourth non square can be
    # approximated, the first step is to find the contours with simple
    # approximation. We use Canny edge detection to extract the edges of
    # the square easily
    # This function returns the centres of the 3 squares of the QR code
    # these are passed on and used to estiate the 4 edges of the QR code
    # in order
    out = cv2.Canny(img, 100, 200)
    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # to find the corners of a QR code, we know that the QR code's corners
    # are made of concentric squares which can be seen as 6 contours in a
    # parent child relation; i.e. There is a contour containing a contour
    # which contains another one so on so forth for a total of 6 times;
    # we shall use this knowledge along with the contour tree heirarchy
    # of storing to find out the contours
    # these contour ids shall be store as a,b,c. We shall also calculate
    # the moments of the contours to estimate the centers and store that too
    # in an array, another array is used to store the areas, and another to
    # store the polygons

    i = 0
    a = 0
    b = 0
    c = 0
    mark = 0
    mc = []
    mu = []
    areaarray = []
    polygonarray = []

    for contour in contours:
        arclength = cv2.arcLength(contour, True)
        areaarray.append(cv2.contourArea(contour))
        # we use a polygon since we are only interested in the shapes that
        # have four corners and not noise, we store the polygon in the array
        polygon = cv2.approxPolyDP(contour, arclength * 0.02, 5)
        polygonarray.append(polygon)
        if(len(polygon)==4):
            k = i
            count = 0
            while (hierarchy[0][k][2] != -1):
                k = hierarchy[0][k][2]
                count = count + 1
            if (hierarchy[0][k][2] != -1):
                count = count + 1
            if (count >= 5):
                if (mark == 0):
                    a = i
                elif (mark == 1):
                    b = i
                elif (mark == 2):
                    c = i
                mark = mark + 1
        # Calculate the moments of the points
        mu.append(cv2.moments(contour))
        # Calculate the centre of masses of the points a,b,c
        # *****This is a quick fix as some moments were 0, giving a divide by 0 error, not sure why******
        if(mu[i]["m10"]==0 and mu[i]["m00"]==0):
            mc.append(np.array((0.0,0.0)))
        else:
            mc.append(np.array((float(mu[i]["m10"] / mu[i]["m00"]), float(mu[i]["m01"] / mu[i]["m00"]))))
        i = i + 1

    # Calculate AB BC CA, line segments formed by A,B,C
    # This will help us determine the outlier, as bottom and right both
    # lie in the longest line and the one which doesn't is the outlier/top
    AB = np.linalg.norm(mc[a]-mc[b])
    BC = np.linalg.norm(mc[b]-mc[c])
    CA = np.linalg.norm(mc[c]-mc[a])

    if (AB > BC and AB > CA):
        outlier = c
        median1 = a
        median2 = b
    elif (CA > AB and CA > BC):
        outlier = b
        median1 = a
        median2 = c
    elif (BC > AB and BC > CA):
        outlier = a
        median1 = b
        median2 = c

    # Now we we need to determine which of median1/median2 is bottom and
    # right. We do this by first calculating the slope of the line joining them
    # and the value on the perpendicular connecting the third point to the
    # median1-median2 line(dist)
    # if slope and dist are both simultanoeusly positive or negative, then
    # median1 is bottom and median2 is right
    # if sign[slope] = -sign[dist], then median1 is right and median2 is bottom
    longline = np.polyfit([mc[median1][0],mc[median2][0]], [mc[median1][1],mc[median2][1]], 1)
    disttop = findperpendiculardist(mc[median1],mc[median2],mc[outlier])

    if(longline[0]>0):
        if(disttop<0):
            bottom = median2
            right = median1
        else:
            right = median2
            bottom = median1
    else:
        if(disttop<0):
            bottom = median1
            right = median2
        else:
            right = median1
            bottom = median2
    top = outlier
    return mc[top],mc[right],mc[bottom]

def findperpendiculardist(pointA,pointB,pointC):
    # Basic function to find the perpendicular line joining A and B
    # and substitute the value of C in the equation
    a = -(pointA[1]-pointB[1])/(pointA[0]-pointB[0])
    b = 1
    c = ((pointA[1]-pointB[1])/(pointA[0]-pointB[0])*pointB[0])-pointB[1]

    pdist = (a*pointC[0]+b*pointC[1]+c)/np.sqrt(a*a+b*b)
    return pdist

def processImage(sourcepoints, imgpoints, orgimg):
    # function wrapper to find the homographic transformation that took place
    # to convert source points to image points[source image to training image],
    # the source points and image points are given in the same order.
    # The source points are in 3d with the Z axis being 0 for all points

    # camera internals from the training image
    size = orgimg.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    axis = np.float32([[0, 0, 3], [0, 3, 0], [3, 0, 0]])

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))
    # use solvePnp to calculate the rotation and translation vectors
    rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(sourcepoints, imgpoints, camera_matrix, dist_coeffs)
    np_rodrigues = np.asarray(rotation_vector[:, :], np.float64)
    # rotM and cameraPosition are the values for world coordinates, was not clear
    # if this was required so leaving it in
    rotM = cv2.Rodrigues(np_rodrigues)[0]
    cameraPosition = -np.matrix(rotM).T * np.matrix(translation_vector)

    # print cameraPosition
    return rotation_vector,translation_vector

def draw(dummmy,qrimg,translationvector,filename):
    # function to draw the camera on a 3d map with respect to the image
    # this function is crude and needs work
    # prints the graph to the same folder as the images
    shape = dummmy.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0.0, 4000.01)
    ax.set_xlim(-shape[0], shape[0])
    ax.set_ylim(-shape[1], shape[1])

    qrshape = qrimg.shape
    qrimg = cv2.cvtColor(qrimg,cv2.COLOR_BGR2BGRA)

    X1 = np.arange(-qrshape[0]/2, qrshape[0]/2, 1)
    Y1 = np.arange(-qrshape[1]/2, qrshape[1]/2, 1)
    X1, Y1 = np.meshgrid(X1, Y1)
    ax.plot_surface(X1, Y1, 0, rstride=1, cstride=1, shade=False, facecolors=np.float32(qrimg/255))

    x = translationvector[0:1]
    y = translationvector[1:2]
    z = translationvector[2:3]

    print x,y,z

    ax.scatter(x, y, z, color='green')
    fig.savefig(filename + '_output.png', bbox_inches='tight')

def getpoints(top,right,bottom,polygon):
    # function to approximate the fourth point
    # tries to match points in top right bottom with the nearest in polygon
    # the nearest to each of them are returned as the new top, bottom
    # and right and the only remaining one is the fourth point
    min = 1000
    i = 0
    for points in polygon:
        val = np.linalg.norm(points - top)
        if(min>val):
            min = val
            topindex = i
        i = i+1
    min = 1000
    i = 0
    for points in polygon:
        val = np.linalg.norm(points - bottom)
        if (min > val):
            min = val
            bottomindex = i
        i = i + 1
    min = 1000
    i = 0
    for points in polygon:
        val = np.linalg.norm(points - right)
        if (min > val):
            min = val
            rightindex = i
        i = i + 1
    # since there are only 4 points 0,1,2,3 the only one remaining will be
    # 6 - sum(top+bottom+right)
    fourthindex = 6-rightindex-bottomindex-topindex
    return polygon[topindex],polygon[rightindex],polygon[bottomindex],polygon[fourthindex]


if __name__ == "__main__":
    # first load the pattern image
    img = cv2.imread('/mnt/excess/pythonworkspace/dronedeploy/images/pattern.jpg')
    # extract AOI and POI[Polygon of Interest]
    qrimg, qrpolygon = extractqrsource(img)

    # get the correct QR points with orientation
    topsource, rightsource, bottomsource = getqrpoints(qrimg)
    # get all 4 points with orientation
    topsource, rightsource, bottomsource, fourthsource = getpoints(topsource,rightsource,bottomsource,qrpolygon)
    # create source points array with extra Z dimension = 0, i.e. Flat surface in 3D
    sourcepoints = np.array([np.float32(np.append(topsource, 0)), np.float32(np.append(rightsource, 0)), np.float32(np.append(bottomsource, 0)), np.float32(np.append(fourthsource, 0))])
    #loop through the image directory and store the images in a directory
    rotationvectorarray = []
    translationvectorarray = []
    for filename in glob.glob('/mnt/excess/pythonworkspace/dronedeploy/images/*.JPG'):  # assuming gif
        image = cv2.imread(filename)
        # extract AOI and POI[Polygon of Interest]
        im, impolygon = extractareaofinterest(image)
        # get the correct QR points with orientation
        toparg, rightarg, bottomarg = getqrpoints(im)
        # get all 4 points with orientation
        toparg, rightarg, bottomarg, fourtharg = getpoints(toparg, rightarg, bottomarg,impolygon)
        # create image points array
        imagepoints = np.array(np.float32([toparg,rightarg,bottomarg,fourtharg]))
        # process the image and obtain rotation and translation vectors
        rotationvector, translationvector = processImage(sourcepoints,imagepoints,image)
        # print qrimg using draw function
        outputimg = draw(image,img,translationvector,filename)
        rotationvectorarray.append(rotationvector)
        translationvectorarray.append(translationvector)