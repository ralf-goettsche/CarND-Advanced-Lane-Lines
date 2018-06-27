import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2

from os import listdir
from os.path import isfile, join

from moviepy.editor import VideoFileClip

def get_cameracalib (dirpath):
    """
    Function for calibrating the camera based upon pictures under 'dirpath'
    Optionally, the results can be printed out_img
    
    Input: Path 'dirpath' to pictues for calibration
    Output: Camera matrix 'mtx' and camera distortion 'dst'
    """
    images = []
    imgpoints = []
    objpoints = []
    filenames = []
    # Definition of object points
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Get imagepoints of all suited pictures
    files = listdir(dirpath)
    for f in files:
        if isfile(join(dirpath, f)):
            file = join(dirpath, f)
            print('{}'.format(file), end='')
            img = mpimg.imread(file)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            # If chessboard corners could be generated in picture, store imagepoints,
            # objectpoints and filename (for optional printout)
            if ret == True:
                img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
                images.append(img)
                print(': integrated', end='')
                imgpoints.append(corners)
                objpoints.append(objp)
                filenames.append(file)
            print('.')
    
    # For calibrating the camera on one specific picture (1st part)
    if 0:
        img = mpimg.imread("camera_cal\calibration3.jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        objpoints, imgpoints = [], []
        objpoints.append(objp)
        imgpoints.append(corners)
    
    # Calculating camera matrix 'mtx', camera distortion 'dist', etc. (for both options(single or many pictures))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # For calibrating the camera on one specific picture, print out (2nd part)
    if 0:
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        undistimg = cv2.undistort(img, mtx, dist, None, mtx)
        mpimg.imsave(fname="output_images/calibration3_average.points.jpg", arr=undistimg)

    # Optional output of all pictures and their undistorted one
    if 1:
        f, axes = plt.subplots(nrows=int(len(images)/2) + len(images)%2, ncols=4, figsize=(15, 10), sharey=True, num=50)
        for (i, file) in enumerate(images):
            axes[int(i/2), i%2*2 + 0].imshow(images[i])
            axes[int(i/2), i%2*2 + 0].set_title('Dist. Image\n{}'.format(filenames[i]), fontsize=10)
            axes[int(i/2), i%2*2 + 0].set_axis_off()
            axes[int(i/2), i%2*2 + 1].imshow(cv2.undistort(images[i], mtx, dist, None, mtx))
            axes[int(i/2), i%2*2 + 1].set_title('Undist. Image\n{}'.format(filenames[i]), fontsize=10)
            axes[int(i/2), i%2*2 + 1].set_axis_off()
        if (len(images)%2):
            i += 1
            axes[ int(i/2), i%2*2 + 0].set_axis_off()
            axes[ int(i/2), i%2*2 + 1].set_axis_off()
            
        plt.show()
    
    # Handing back the camera matrix 'mtx' and the camera distortion 'dst'
    return (mtx, dist)

def RGBconvimg (img, method = 'gray'):
    """
    Function for color transformation of image 'img' according to 'method'
    
    Input: Image 'img', color transformation method 'method' 
    Output: The transformed image 'cvtimg'
    """
    cvtimg = {
        'hls':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS),
        'hls,h':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,0],
        'hls,l':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,1],
        'hls,s':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2],
        'hsv':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
        'hsv,h':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0],
        'hsv,s':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1],
        'hsv,v':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2],
        'bgr':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        'gray': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    }[method](img)
    
    return cvtimg

def calc_curvature(leftx, rightx, lefty, righty):
    """
    Function for calculating the curvature of a polynomial in meters
    (Taken from lecture with adoption of 'xm_per_pix')
    
    Input: X- and y-coordinates of the points building the left line ('leftx', 'lefty')
           and building the right line ('rightx', 'righty')
    Output: Curvature of the left line 'left_curverad' and right line 'right_curverad'
    """
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension

    y_eval_left = np.max(lefty)
    y_eval_right = np.max(righty)
    # Fit polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Curvature in meters
    return left_curverad, right_curverad
    
def calc_xoffset(left_fit, right_fit, shape):
    """
    Function for calculating the offset of the car to the center of the road in meters
    
    Input: Parameter of the adopted polynom for the left line ('left_fit') and right line ('right_fit'),
           Dimension of the picture 'shape'
    Output: Offset 'xoffset' of the car to the center of the road in meters
    """
    xm_per_pix = 3.7/600 # meters per pixel in x dimension

    xlbottom = left_fit[0]*shape[0]**2 + left_fit[1]*shape[0] + left_fit[2]
    xrbottom = right_fit[0]*shape[0]**2 + right_fit[1]*shape[0] + right_fit[2]
    
    xoffset = ((xlbottom + xrbottom) - shape[1])/2 * xm_per_pix
    return xoffset

def plotimg (img, method='rgb'):
    """
    Plotting one image with according color map
    
    Input: Image 'img', color map 'method'
    """
    plt.figure(200)
    if (len(img.shape) == 3):
        if (method == 'rgb'):
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=method)
    else:
        plt.imshow(img, cmap='gray')
    plt.show()

def plot2img (img, cvtimg, method='rgb'):
    """
    Plotting two images next to each other (mainly for comparison reason) in according color map
    
    Input: Images 'img' and 'cvtimg', color map 'method'
    """
    #number = plt.gcf().number + 1
    number = 40
    f, (axe1, axe2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), num=number)
    axe1.imshow(img)
    axe1.set_title('Orig', fontsize=15)
    if (len(cvtimg.shape) == 3):
        if (method == 'rgb'):
            axe2.imshow(cvtimg)
        else:
            axe2.imshow(cvtimg, cmap=method)
    else:
        axe2.imshow(cvtimg, cmap='gray')
    axe2.set_title('Converted Image', fontsize=15)
    plt.show()

def plot_lanelines(pltimg, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
    """
    Plotting one image with adopted polynomes for left and rigth line and highlighted lane lines
    
    Input: Warped image 'pltimg', parameters for the adopted polynomes for left and right line ('left_fit', 'right_fit'),
           indices of line points ('left_lane_inds', 'right_lane_inds') in nonzero arrays ('nonzerox', 'nonzeroy')
    """
    # Calculating the base in y-parameters
    ploty = np.linspace(0, pltimg.shape[0]-1, pltimg.shape[0] )
    # Calculating the points on the polynoms for left and right line in x-parameters
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Coloring the points of the lines
    pltimg[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    pltimg[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.figure(30)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imshow(pltimg)
    plt.xlim(0, pltimg.shape[1])
    plt.ylim(pltimg.shape[0], 0)
    plt.show()


def gen_img_w_marks(img, bin_warp, left_fit, right_fit, Minv):
    """
    Generating image with marked lane area
    (Mostly taken from lecture)
    
    Input: Image 'img', binary filtered and warped image 'bin_warp',
           parameters of the polynoms for left and right line ('left_fit', 'right_fit')
           and the unwarp matrix 'Minv'
    Output: Image with marked lane area
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Calculating the base in y-parameters
    ploty = np.linspace(0, bin_warp.shape[0]-1, bin_warp.shape[0] )
    # Calculating the points on the polynoms for left and right line in x-parameters
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the marked lane area onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    marks = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    img = cv2.addWeighted(img, 1, marks, 0.3, 0)
    
    return img

    
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255), method='gray'):
    """
    Edge detection in x or y direction with according color transformation in certain threshold region
    
    Input: RGB image 'img', the detection direction 'orient', filter size 'sobel_kernel',
           threshold range 'thresh', color transformation 'method'
    Output: Binary filtered image 'binary_output'
    """
    # Color transformation
    gray = RGBconvimg(img, method=method)
    # Edge detection according to orientation with chosen filter size 
    if orient == 'x':
        devimg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient== 'y':
        devimg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Transformation into image data
    devimg = np.absolute(devimg)
    devimg = np.uint8(255*devimg/np.max(devimg))
    
    # Filtering based upon threshold range
    binary_output = np.zeros_like(devimg)
    binary_output[(devimg >= thresh[0]) & (devimg <= thresh[1])] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), method='gray'):
    """
    Magnitude calculation of x- and y-edges with according color transformation in certain threshold region
    
    Input: RGB image 'img', filter size 'sobel_kernel', threshold range 'mag_thresh', color transformation 'method'
    Output: Magnitude binary filtered image 'binary_output'
    """
    # Color transformation
    gray = RGBconvimg(img, method=method)
    # Edge detection in x- and y-direction
    devximg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    devyimg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Magnitude of edge detection calculation and transformation into image data
    absimg = np.sqrt(devximg**2 + devyimg**2 )
    absimg = np.uint8(255*absimg/np.max(absimg))
    
    # Filtering based upon threshold range
    binary_output = np.zeros_like(absimg)
    binary_output[(absimg >= mag_thresh[0]) & (absimg <= mag_thresh[1])] = 1
    
    return binary_output

def dir_threshold(img, sobel_kernel=3, dir_thresh=(0, np.pi/2), method='gray'):
    """
    Edge direction calculation with according color transformation in certain threshold region
    
    Input: RGB image 'img', filter size 'sobel_kernel', threshold range 'dir_thresh', color transformation 'method'
    Output: Edge direction filtered image 'binary_output'
    """
    # Color transformation
    gray = RGBconvimg(img, method=method)
    # Edge detection in x- and y-direction
    devximg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    devyimg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculation of direction of the edges
    absximg = np.absolute(devximg)
    absyimg = np.absolute(devyimg)
    dirdevimg = np.arctan2(absyimg, absximg)
    
    # Filtering based upon direction threshold range
    binary_output = np.zeros_like(dirdevimg)
    binary_output[(dirdevimg >= dir_thresh[0]) & (dirdevimg <= dir_thresh[1])] = 1
    
    return binary_output
    
def cut_region_of_interest(img, filterframe):
    """
    Cutting off regions in an image which are not of interest
    
    Input: Image 'img', frame 'filterframe' surrounding region of interest
    Output: Image 'masked_image'
    """
    mask = np.zeros_like(img)   
    
    # Filter value
    ignore_mask_color = 1.0
        
    # Filling pixels inside the polygon defined by 'filterframe' with the filter value    
    cv2.fillPoly(mask, filterframe, ignore_mask_color)
    
    # Returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image


def playground_grad_color (image):
    """
    Function to play around with different filters, filter sizes, thresholds, color conversions
    and their combinations to determine the best for the according image
    Input: Image 'image'
    """
    # Selection of color conversion (all channel: rgb to hsv, hls, bgr and their single dimensions plus gray)
    method = 'hls,s'
    #method = 'hsv,v'
    # Definition of different filters
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100), method=method)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100), method=method)
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30,100), method=method)
    dir_binary = dir_threshold(image, sobel_kernel=15, dir_thresh=(0.7, 1.3), method=method)
    dir_binary1 = dir_threshold(image, sobel_kernel=3, dir_thresh=(0.7, 1.0), method=method)
    dir_binary2 = dir_threshold(image, sobel_kernel=3, dir_thresh=(1.0, 1.3), method=method)
    dir_binary_sum = np.zeros_like(dir_binary1)
    dir_binary_sum[((dir_binary1 == 1) | (dir_binary2 == 1))] = 1

    # Combining different filters 
    combined  = np.zeros_like(dir_binary)
    combined1 = np.zeros_like(dir_binary)
    combined2 = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((grady == 1) & (dir_binary == 1))] = 1
    #combined1[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined1[((gradx == 1) & (grady == 1) & (dir_binary == 1))] = 1
    #combined2[((gradx == 1) & (grady == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined2[((gradx == 1) & (grady == 1) & (dir_binary_sum == 1)) | ((mag_binary == 1) & (dir_binary_sum == 1))] = 1
    
    # Optional print out (original, 1+2 combinations, gradx, grady, magnitude, directed)
    if 1:
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(4, 2, figsize=(24, 9), num=20)
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined Image', fontsize=20)
        ax3.imshow(gradx, cmap='gray')
        ax3.set_title('Gradx', fontsize=20)
        ax4.imshow(grady, cmap='gray')
        ax4.set_title('Grady', fontsize=20)
        ax5.imshow(mag_binary, cmap='gray')
        ax5.set_title('Magnitude Gradient Image', fontsize=20)
        ax6.imshow(dir_binary_sum, cmap='gray')
        ax6.set_title('Directed Gradient Sum Image', fontsize=20)
        ax7.imshow(combined1, cmap='gray')
        ax7.set_title('Combined1', fontsize=20)
        ax8.imshow(combined2, cmap='gray')
        ax8.set_title('Combined2', fontsize=20)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        plt.show()

def my_grad_color_filter (image):
    """
    Filter and color transformation showing most promising results
    
    Input: RGB Image 'image'
    Output: Binary filtered image 'combined'
    """
    ### First filter definition based upon the l-layer of hls transformation
    method = 'hls,l'
    # Filtering l-image with similar parameters of lecture
    gradx_hls_l = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100), method=method)
    grady_hls_l = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100), method=method)
    mag_binary_hls_l = mag_thresh(image, sobel_kernel=3, mag_thresh=(30,100), method=method)
    dir_binary_hls_l = dir_threshold(image, sobel_kernel=15, dir_thresh=(0.7, 1.3), method=method)
    # Combining filters like lecture as they showed best result
    combined_hls_l  = np.zeros_like(dir_binary_hls_l)
    combined_hls_l[((gradx_hls_l == 1) & (grady_hls_l == 1)) | ((mag_binary_hls_l == 1) & (dir_binary_hls_l == 1))] = 1
    ### Second filter definition based upon the s-layer of hls transformation
    method = 'hls,s'
    # Filtering s-image, parameter of gradients have been lowered for better results 
    gradx_hls_s = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(5, 100), method=method)
    grady_hls_s = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(5, 100), method=method)
    mag_binary_hls_s = mag_thresh(image, sobel_kernel=3, mag_thresh=(30,100), method=method)
    dir_binary_hls_s = dir_threshold(image, sobel_kernel=15, dir_thresh=(0.7, 1.3), method=method)
    # Combining filters like lecture as they showed best result
    combined_hls_s  = np.zeros_like(dir_binary_hls_s)
    combined_hls_s[((gradx_hls_s == 1) & (grady_hls_s == 1)) | ((mag_binary_hls_s == 1) & (dir_binary_hls_s == 1))] = 1

    # Combining both filters
    combined  = np.zeros_like(combined_hls_s)
    combined[((combined_hls_l == 1) | (combined_hls_s == 1))] = 1

    # Optional print out for better analysis; 
    # !!! To be switched off during movie generation!!!
    if 0:
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2, figsize=(24, 9), num=10)
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined Image', fontsize=20)
        ax3.imshow(gradx_hls_l, cmap='gray')
        ax3.set_title('Gradx, hls_l', fontsize=20)
        ax4.imshow(grady_hls_l, cmap='gray')
        ax4.set_title('Grady, hls_l', fontsize=20)
        ax5.imshow(gradx_hls_s, cmap='gray')
        ax5.set_title('Gradx, hls_s', fontsize=20)
        ax6.imshow(grady_hls_s, cmap='gray')
        ax6.set_title('Grady, hls_s', fontsize=20)
        ax7.imshow(mag_binary_hls_l, cmap='gray')
        ax7.set_title('Magnitude Gradient Image, hls_l', fontsize=20)
        ax8.imshow(dir_binary_hls_l, cmap='gray')
        ax8.set_title('Directed Gradient Image, hls_l', fontsize=20)
        ax9.imshow(mag_binary_hls_s, cmap='gray')
        ax9.set_title('Magnitude Gradient Image, hls_s', fontsize=20)
        ax10.imshow(dir_binary_hls_s, cmap='gray')
        ax10.set_title('Directed Gradient Image, hls_s', fontsize=20)

        plt.show()
    
    return combined

def corners_unwarp(img):
    """
    Warping image by just considering region of interest
    
    Input: Image 'img'
    Output: Warped image 'warped', transformation matrix 'M' and its inverse 'Minv'
    """
    imgx = img.shape[1]
    imgy = img.shape[0]
    # Definition of trapezoid to cut off sides of the road and the top above vanishing point
    frame_vertices = np.array([[(0, imgy), (555,450), (735,450), (imgx,imgy)]], dtype=np.int32)
    # Cutting of areas
    img = cut_region_of_interest(img, frame_vertices)
    ## Definition of rectangle inside of image to be transformed
    #src = np.float32([[0,720], [575,455], [705,455], [1280,720]])
    #src = np.float32([[-640,720], [576,440], [704,440], [1920,720]])
    #src = np.float32([[-320,720], [592,440], [688,440], [1600,720]])
    src = np.float32([[-320,720], [558,450], [722,450], [1600,720]])
    ## Definition of target rectangle (full image area)
    #dst = np.float32([[0,720], [0,0], [1280,0], [1280,720]])
    dst = np.float32([[0,720], [0,0], [1280,0], [1280,720]])
    
    # Calculation of tranformation matrices
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Image transformation
    warped = cv2.warpPerspective(img, M, (imgx, imgy), flags=cv2.INTER_LINEAR)
    
    return warped, M, Minv

def lineidx_detection_per_window (bin_warped_img, nonzerox, nonzeroy, nwindows=9):
    """
    Line detection based upon window methodology
    (Taken from lectures and modified with regards to window imaging)
    
    Input: Binary filtered and warped image 'bin_warped_img', x-indices (nonzerox) and y-indices (nonzeroy)
           of non-zero points in binary filtered and warped image, number of windows 'nwindows' used for line detection
    Output: Location of points of the left line ('left_lane_inds') and right line ('right_lane_inds'),
            image with the windows 'win_img' for later optional print out
    """
    # Determining the histogram and hence the area of the lines in the image incl. optional print out
    histogram = np.sum(bin_warped_img[bin_warped_img.shape[0]//2:,:], axis=0)
    if 0:
        plt.plot(histogram)
        plt.show()
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(bin_warped_img.shape[0]/nwindows)
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right line pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Image for inserting the windows later on
    win_img = np.zeros_like(np.uint8(np.dstack((bin_warped_img,bin_warped_img,bin_warped_img))))

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = bin_warped_img.shape[0] - (window+1)*window_height
        win_y_high = bin_warped_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(win_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        [0,255,0], 2) 
        cv2.rectangle(win_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        [0,255,0], 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    return left_lane_inds, right_lane_inds, win_img

def lineidx_detection_per_fit(bin_warped_img, nonzerox, nonzeroy, left_fit, right_fit, margin=100):
    """
    Line detection based upon fitting methodolgy (using existing fitting and adopting it)
    (Taken from lectures and modified with regards to search area imaging)
    
    Input: Binary filtered and warped image 'bin_warped_img', x-indices (nonzerox) and y-indices (nonzeroy)
           of non-zero points in binary filtered and warped image, parameters of the polynoms
           representing the left ('left_fit') and right line ('right_fit'), 'margin' of the search window
    Output: Location of points of the left line ('left_lane_inds') and right line ('right_lane_inds'),
            image with the search areas 'win_img' for later optional print out
    """
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, bin_warped_img.shape[0]-1, bin_warped_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    win_img = np.uint8(np.dstack((bin_warped_img, bin_warped_img, bin_warped_img))*255)
    cv2.fillPoly(win_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(win_img, np.int_([right_line_pts]), (0,255, 0))

    return left_lane_inds, right_lane_inds, win_img

def sanity_check(left_fit, right_fit, left_curverad, right_curverad, shape):
    """
    Function for the sanity check of determined lane detection
    Here, the check only consists of checking the distance of the two estimated lanes
    
    Input: Parameter of the polynoms representing the left ('left_fit') and right line ('right_fit'),
           the calculated curvature of left ('left_curverad') and right line ('right_curverad'),
           image dimensions 'shape'
    Output: Boolean value 'sanity'
    """
    
    # Defining the expected range for the polynom values to be in for sanity
    min_delta = 520 # equals approx. 3.20m
    max_delta = 760 # equals approx. 4.70m
    
    # Calculation of polynom points
    ploty = np.linspace(0, shape[0]-1, shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Calculation of distances of the points
    delta_fitx = right_fitx - left_fitx
    max_delta_fitx = np.max(delta_fitx)
    min_delta_fitx = np.min(delta_fitx)
    # Determining the sanity value for distance
    if ((min_delta_fitx >= min_delta) & (max_delta_fitx <= max_delta)):
        dist_sanity = 1
    else:
        dist_sanity = 0
        
    # Determining overall sanity (might be extended later on)
    sanity = dist_sanity
    
    return sanity
    
def process_single_image():
    """
    Function for processing one picture of choice, analyzing it and adopt parameters
    """
    
    # Uncomment for getting all the plots immediately
    #plt.ion()
    
    # Play around with camera calibration and print out result for storage below
    #mtx, dist = get_cameracalib("camera_cal")
    #print('mtx = {}'.format(mtx))
    #print('dist = {}'.format(dist))

    ### Example calibrations derived from one picture (calibration3) and 
    ### of all pictures in folder 'test_images'
    # calibration3
    #mtx = [[  1.16949654e+03   0.00000000e+00   6.68285391e+02]
    #       [  0.00000000e+00   1.14044391e+03   3.59592347e+02]
    #       [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
    #dist = [[-0.25782023 -0.0988196   0.01240745  0.00407057  0.52894628]]

    # average
    #mtx = [[  1.15396093e+03   0.00000000e+00   6.69705359e+02]
    #       [  0.00000000e+00   1.14802495e+03   3.85656232e+02]
    #       [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
    #dist = [[ -2.41017968e-01  -5.30720497e-02  -1.15810318e-03  -1.28318543e-04 2.67124302e-02]]
    
    # The chosen values for camera matrix and camera distortion
    mtx = np.array([[  1.15396093e+03,   0.00000000e+00,   6.69705359e+02],
           [  0.00000000e+00,   1.14802495e+03,   3.85656232e+02],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    dist = np.array([[ -2.41017968e-01,  -5.30720497e-02,  -1.15810318e-03,  -1.28318543e-04, 2.67124302e-02]])

    # Definition of images, use one at a time
    image = mpimg.imread('./test_images/straight_lines1.jpg')
    #image = mpimg.imread('./test_images/straight_lines2.jpg')
    #image = mpimg.imread('./test_images/test1.jpg')
    #image = mpimg.imread('./test_images/test2.jpg')
    #image = mpimg.imread('./test_images/test3.jpg')
    #image = mpimg.imread('./test_images/test4.jpg')
    #image = mpimg.imread('./test_images/test5.jpg')
    #image = mpimg.imread('./test_images/test6.jpg')
    ### Extra images taken out of the video describing different situation at crossing the first bridge
    #image = mpimg.imread('./Snapshots/vlcsnap-01.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-02.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-03.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-04.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-05.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-06.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-07.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-08.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-09.jpg')
    #image = mpimg.imread('./Snapshots/vlcsnap-10.jpg')
    
    # Undistorting image
    image = cv2.undistort(image, mtx, dist, None, mtx)
    plotimg(image)

    # Choose this one if you want to play around with edge filtereing, color transformation
    if 0:
        playground_grad_color(image)
    
    # Use the chosen edge detection and color transformation for filtering the image
    filtered_image = my_grad_color_filter(image)
    plotimg(filtered_image)

    
    # Generate the warped image
    birdimage, perspective_M, Minv = corners_unwarp(filtered_image)
    plot2img(image, birdimage)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = birdimage.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Transforming one dimensional, filtered and warped image into 3 channel image
    out_img = np.uint8(np.dstack((birdimage, birdimage, birdimage))*255)

    ### Simulating first time lane detection based upon window methodology
    left_lane_inds, right_lane_inds, win_img = lineidx_detection_per_window(birdimage, nonzerox, nonzeroy)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Combine image with windows and 3-dim bird image, print it out
    plt_img = cv2.addWeighted(out_img, 1, win_img, 1.0, 0)
    plot_lanelines(plt_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)


    ### Simulating second time lane detection based upon search area method
    nonzero = birdimage.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # lane detection based upon search area method (requiring fitted polynoms)    
    left_lane_inds, right_lane_inds, win_img = lineidx_detection_per_fit(birdimage, nonzerox, nonzeroy, left_fit, right_fit)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Combine image with search area and 3-dim bird image, print it out
    plt_img = cv2.addWeighted(out_img, 1, win_img, 0.3, 0)
    plot_lanelines(plt_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

    # Calculate curvature and offset from center of road for sanity check and print out into final picture
    left_curverad, right_curverad = calc_curvature(leftx, rightx, lefty, righty)
    xoffset = calc_xoffset(left_fit, right_fit, image.shape)
    sanity = sanity_check(left_fit, right_fit, left_curverad, right_curverad, image.shape)
    
    # Unwarp bird view image, integrate the polynom lines as marker into the image
    image = gen_img_w_marks(image, birdimage, left_fit, right_fit, Minv)

    # Add curvature, offset, and sanity information into final image
    str =("left curve:   {:7.2f}m".format(left_curverad))
    cv2.putText(image, str, (100,40), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
    str =("right curve: {:7.2f}m".format(right_curverad))
    cv2.putText(image, str, (100,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
    str =("Center offset:  {:5.2f}m".format(xoffset))
    cv2.putText(image, str, (100,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
    str =("Sanity: {}".format(sanity))
    cv2.putText(image, str, (100,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)

    # Plot the final image
    plotimg(image)
    # Uncomment if you want to get all pictures immediately but have a final look at them
    #input('Press Enter to quit')


    
def process_images(image):
    """
    Function for processing consecutive images of a movie
    
    Input: RGB Image 'image'
    Output: Image with masked lane area and curvature, offset information ('marked_image')
    """
    
    # Information if the first image has been analyzed => use search area method
    global first_pic_done
    # Storage of parameter for polynom of lines
    global left_fit_glob
    global right_fit_glob
    
    # Chosen camera matrix and distortion values
    mtx = np.array([[  1.15396093e+03,   0.00000000e+00,   6.69705359e+02],
           [  0.00000000e+00,   1.14802495e+03,   3.85656232e+02],
           [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
    dist = np.array([[ -2.41017968e-01,  -5.30720497e-02,  -1.15810318e-03,  -1.28318543e-04, 2.67124302e-02]])

    # Undistorting image
    image = cv2.undistort(image, mtx, dist, None, mtx)

    # Color transformation and filtering according to best method evaluated
    filtered_image = my_grad_color_filter(image)

    # Warping image
    birdimage, perspective_M, Minv = corners_unwarp(filtered_image)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = birdimage.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Transforming one-dimensional birds view image into 3-dim image
    out_img = np.uint8(np.dstack((birdimage, birdimage, birdimage))*255)
    
    # Choosing either window method (first picture) or search area (all other pictures) for lane line detection
    if first_pic_done:
        left_fit = left_fit_glob
        right_fit = right_fit_glob
    
        left_lane_inds, right_lane_inds, win_img = lineidx_detection_per_fit(birdimage, nonzerox, nonzeroy, left_fit, right_fit)
    else:
        left_lane_inds, right_lane_inds, win_img = lineidx_detection_per_window(birdimage, nonzerox, nonzeroy)
        first_pic_done = 1

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Calculate curvature and offset from center of road for sanity check and print out into final picture
    left_curverad, right_curverad = calc_curvature(leftx, rightx, lefty, righty)
    xoffset = calc_xoffset(left_fit, right_fit, image.shape)
    sanity = sanity_check(left_fit, right_fit, left_curverad, right_curverad, image.shape)
    
    ### Placeholder for sanity check
    #if not(first_pic_done) & sanity:
    left_fit_glob = left_fit
    right_fit_glob = right_fit
    
    
    # Unwarp bird view image, integrate the polynom lines as marker into the image
    marked_image = gen_img_w_marks(image, birdimage, left_fit_glob, right_fit_glob, Minv)

    # Add curvature and offset information into final image
    str =("left curve:   {:7.2f}m".format(left_curverad))
    cv2.putText(marked_image, str, (100,40), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
    str =("right curve: {:7.2f}m".format(right_curverad))
    cv2.putText(marked_image, str, (100,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
    str =("Center offset:  {:5.2f}m".format(xoffset))
    cv2.putText(marked_image, str, (100,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
    #str =("Sanity: {}".format(sanity))
    #cv2.putText(marked_image, str, (100,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)

    return marked_image

### MAIN
# Either analyze one picture or generate a movie
if 0:
    process_single_image()
else:
    first_pic_done = 0
    left_fit_glob, right_fit_glob = [], []
    moviepath = "./project_video.mp4"
    movieoutpath = "./project_video_masked.mp4"
    #moviepath = "./challenge_video.mp4"
    #movieoutpath = "./challenge_video_masked.mp4"
    #moviepath = "./harder_challenge_video.mp4"
    #movieoutpath = "./harder_challenge_video_masked.mp4"
    movie = VideoFileClip(moviepath)
    masked_movie = movie.fl_image(process_images)
    masked_movie.write_videofile(movieoutpath, audio=False)

