#import torchvision.transforms.functional as f
import torchvision.transforms.functional as TF
import random
import torch
def my_segmentation_transforms(image,angle):
    #if random.random() > 0.5:
        #angle = random.randint(-30, 30)
    image = TF.rotate(image, angle)
        #segmentation = TF.rotate(segmentation, angle)
    # more transforms ...
    return image#,angle

image=torch.Tensor(([[0,0,1],[0,1,0],[1,0,0]])).reshape(1,1,3,3)
mask=(image.clone().detach().bool()).float()


#print(image)
"""
import math
x =torch.Tensor([[3,1.5],[1.5,3],[0,3]]) #3,2
#phi = torch.tensor(math.pi / 2)
#s = torch.sin(phi)
#c = torch.cos(phi)
#rot = torch.Tensor([[c, s],[-s, c]])

print("rot",rot)
original_point=torch.Tensor([1.5,1.5])

print("((x-original_point[None,:])",(x-original_point[None,:]))
x_rot = (rot@((x-original_point[None,:]).transpose(0,1))).transpose(0,1)+original_point[None,:]
#rotation_matrix=
print("x_rot",x_rot)
#print(my_segmentation_transforms(input,45))

import cv2
  
# Reading the image
#image = cv2.imread('image.jpeg')
  
# Extracting height and width from 
# image shape
height, width = image.shape[:2]
  
# get the center coordinates of the
# image to create the 2D rotation
# matrix
center = (width/2, height/2)
  
# using cv2.getRotationMatrix2D() 
# to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=90, scale=1)
  
# rotate the image using cv2.warpAffine 
# 90 degree anticlockwise
print(rotate_matrix)
rotated_image = cv2.warpAffine(
    src=image.numpy(), M=rotate_matrix, dsize=(width, height))
  
#cv2.imshow("rotated image:", rotated_image)
#cv2.imwrite('rotated_image.jpg', rotated_image)
"""
import kornia.geometry.transform as t
import kornia
input_center=torch.Tensor([1,1]).reshape(1,2)
angle=torch.Tensor([-90])
scale=torch.Tensor([1,1]).reshape(1,2)
a=t.get_rotation_matrix2d(input_center,angle,scale)

#a=t.get_affine_matrix2d()
b=kornia.geometry.transform.warp_affine(image,a,dsize=(3,3))

rotated_mask=kornia.geometry.transform.warp_affine(mask,a,dsize=(3,3))

print("b",b)
print("rotated_mask",rotated_mask.bool())
coordinate=torch.Tensor([[0,2],[1,1],[2,0]]) #3,2
B,_=coordinate.shape
extra_coordinate=torch.cat([coordinate,torch.ones((B,1))],dim=-1)
print("before extra_coordinate",extra_coordinate)
extra_coordinate=extra_coordinate.transpose(0,1)
print("after extra_coordinate",extra_coordinate)
print("extra_coordinate",extra_coordinate.shape)
print("rotation_matrix.shape",a.shape)
#print("image",b) #
print("rotation_matrix@extra_coordinate",(a.squeeze(0)@extra_coordinate).transpose(0,1))