import cv2
from matplotlib import pyplot as plt



#orimage= cv2.imread("histogram\d1-c1-003.jpg", 1)
#img = cv2.resize(orimage, (500,500)) #resize if needed

# read an image
img = cv2.imread("histogram\d1-c1-003.jpg", 1)
plt.imshow(img)


#Convert image to LAB space (L:luminosity, A and B for color info)
lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


#Split to L, A and B channels
l, a, b = cv2.split(lab_img)

#Plot the histogram 
#plt.hist(l.flat, bins=100, range=(0,255))


#Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_img = clahe.apply(l)


#Combine the CLAHE enhanced L-channel back with A and B channels
updated_lab_img2 = cv2.merge((clahe_img,a,b))

#Convert LAB image back to color (RGB)
CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
plt.imshow(CLAHE_img)


cv2.imshow("Original image", img)
cv2.imshow('CLAHE Image', CLAHE_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

