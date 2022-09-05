import kornia.geometry.transform as t
import kornia
import numpy as np
import kornia.geometry.transform as t
import torch

#np.ones
import util.misc as m


def transformed(a_nested_Tensor,device):
     
    image = a_nested_Tensor.tensors.clone().detach()
    b,c,h,w=image.shape
    image_dtype=image.dtype
    mask_dtype=a_nested_Tensor.mask.dtype
    mask = a_nested_Tensor.mask.float()-1   # [0,1]-> [-1,0]

    angles=np.random.choice([-180,-90,-15,-10,5,10,15,90,180])

    angle=torch.tensor(angles).clone().detach().type(image_dtype).to(device)

    input_center=torch.tensor([h/2,w/2]).repeat((b,1)).type(image_dtype).to(device)
    angle=angle.repeat((b)).type(image_dtype).to(device)
    
    #random_trans_x=torch.randint(low=-30,high=30,size=(2,))
    translation=torch.randint(low=-30,high=30,size=(2,)).reshape(1,2).repeat((b,1)).type(image_dtype).to(device)
    #print("translation.shape",translation.shape)
    #print("image.shape",image.shape)
    scale=torch.Tensor([1,1]).repeat((b,1)).type(image_dtype).to(device)
    rotation_matrix=t.get_rotation_matrix2d(input_center,angle,scale)
    d_affine_matrix=t.get_affine_matrix2d(translation,input_center,scale,angle,sx=None,sy=None)[:,:2,:]
    #print("d_affine_matrix.shape",d_affine_matrix)
    #print("rotation_matrix.shape",rotation_matrix.shape)
    
    #print("d_affine_matrix.shape",d_affine_matrix.shape)
    #rotated_image=kornia.geometry.transform.warp_affine(image,rotation_matrix,dsize=(h,w))
    rotated_image=kornia.geometry.transform.warp_affine(image,d_affine_matrix,dsize=(h,w))
    mask=mask.unsqueeze(1)
    #rotated_mask=kornia.geometry.transform.warp_affine(mask,rotation_matrix,dsize=(h,w))
    rotated_mask=kornia.geometry.transform.warp_affine(mask,d_affine_matrix,dsize=(h,w))

    rotated_mask=rotated_mask.squeeze(1)

    real_mask= torch.isclose(rotated_mask,torch.zeros(rotated_mask.shape).to(device),atol=1e-1).type(mask_dtype).to(device)
    
    return m.NestedTensor(rotated_image,real_mask),d_affine_matrix,torch.tensor([h,w]).clone().detach().type(image_dtype).to(device)




"""

def transformed(a_nested_Tensor,device):
     
    image = a_nested_Tensor.tensors.clone().detach()
    #image=torch.tensor([[100,200,11,0,0],[500,600,2,0,0],[900,1000,3,0,0],[1300,1400,4,0,0],[1600,1700,5,0,0]]).float().unsqueeze(0).unsqueeze(0)
    #a_mask=torch.tensor([[False,False,False,True,True],[False,False,False,True,True],[False,False,False,True,True],[False,False,False,True,True],[False,False,False,True,True]]).unsqueeze(0)
    #mask_dtype=a_mask.dtype
    #mask=a_mask.float()-1
    
    #print("image",image)
    #print("mask",mask)
    b,c,h,w=image.shape
    image_dtype=image.dtype
    mask_dtype=a_nested_Tensor.mask.dtype
    #
    mask = a_nested_Tensor.mask.float()-1   # [0,1]-> [-1,0]

    angles=np.random.choice([-90,-60,-45,-30,0,90,60,45,30])
    #angle=np.random.choice([360])
    angle=torch.tensor(angles).clone().detach().type(image_dtype).to(device)
    #print("image.shape",image.shape)
    #print("mask",mask.shape)
    input_center=torch.tensor([h/2,w/2]).repeat((b,1)).type(image_dtype).to(device)
    angles=torch.tensor(angle).clone().detach().repeat((b)).type(image_dtype).to(device)
    scale=torch.Tensor([1,1]).repeat((b,1)).type(image_dtype).to(device)
    rotation_matrix=t.get_rotation_matrix2d(input_center,angles,scale)
    rotated_image=kornia.geometry.transform.warp_affine(image,rotation_matrix,dsize=(h,w))
    #print("rotated_image",rotated_image)
    mask=mask.unsqueeze(1)
    rotated_mask=kornia.geometry.transform.warp_affine(mask,rotation_matrix,dsize=(h,w))
    rotated_mask=rotated_mask.squeeze(1)
    #print("rotated_mask",rotated_mask)
    real_mask= torch.isclose(rotated_mask,torch.zeros(rotated_mask.shape).to(device),atol=1e-1).type(mask_dtype).to(device)
    #print("real_mask",real_mask)
    #print("a_nested_Tensor.mask.sum",a_nested_Tensor.mask.sum())
    #print("real_mask.sum()",real_mask.sum())
    #print("real_mask[0,:,:]!",real_mask)
    #print("a_nested_Tensor.mask[0,:,:]",a_nested_Tensor.mask[0,:,:])
    #print("a_mask[0,:,:]",a_mask)
    #print("real_mask.sum()",real_mask.sum())
    #print("a_nested_Tensor.mask.sum()",a_nested_Tensor.mask.sum())
    #print("a_mask.sum()",a_mask.sum())
    #print("real_mask[0,:,:]!=a_nested_Tensor.mask[0,:,:].sum()",(real_mask.sum())==(a_nested_Tensor.mask.sum()))
    #print("real_mask[0,:,:]==mask[0,:,:].sum()",(real_mask.sum())==(a_mask.sum()))
    #real_mask>=a_mask
    return rotated_image,real_mask,rotation_matrix

"""
#transformed([],'cpu')