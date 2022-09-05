# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable

#import MultiScaleDeformableAttention as MSDA


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()

def return_sampling_grid_l__core_pytorch(value, value_spatial_shapes, sampling_locations,attention_weights,return_shift=True):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    #print("value",value.shape)
    #unscaled_sample_locations=[] #[B,300,head,level,num_sample_points,2]
    #print("value_spatial_shapes",value_spatial_shapes)
    #sampling_locations.shape torch.Size([8, 300, 8, level, 4, 2])
    #print("sampling_locations[0,0,0,0,0,0]",sampling_locations[0,0,0,0,0,0])
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    #print("len(value_list)",len(value_list))
    #print("value_list[0].shape",value_list[0].shape)
    #print("value_list[1].shape",value_list[1].shape)
    #print("value_list[2].shape",value_list[2].shape)
    #print("value_list[3].shape",value_list[3].shape)
    sampling_grids = 2 * sampling_locations - 1  #make [0,1]->[-1,1]
    #print("sampling_grids.shape",sampling_grids[0,0,0,0,0,0])
    
    sampling_value_list = []
    #sclaed_sample_location=sampling_locations[:, :, :, lid_]
    #prnt
    #print("value_spatial_shapes",value_spatial_shapes)
    if return_shift:
        sampling_grid_l_list=[]
        scale_factors=[]
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        ###customer
        sclaed_sample_location=sampling_locations[:, :, :, lid_].transpose(1, 2).flatten(0, 1) #[B*heads,300,num_samples,2]
        #encoder:       sampling_grid_l_.shape torch.Size([16, 10723, 4, 2])
        #sclaed_sample_location.shape torch.Size([16, 10723, 4, 2])
        #decoder: sclaed_sample_location.shape torch.Size([16, 300, 4, 2])
        if return_shift:
            factor=torch.unsqueeze(value_spatial_shapes[lid_],0)
            scale_factors.append(factor)
        #print("before",sclaed_sample_location[0,0,0,:])
        #print("factor",factor)
        #unsclaed_sample_location=sclaed_sample_location*factor[None,None,:,]
        #print("after",unsclaed_sample_location)
        #torch.Size([16, 10723, 4, 2])
        
        #[B*num_heads,300,4,2]
        #h_s=H_ * np.ones(sampling_locations)
        #w_s=W_ * np.ones(sampling_locations)
        #unscaled_sample_locations.append(sclaed_sample_location*value_spatial_shapes)
 
        #print("lid_",lid_)
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).contiguous().reshape(N_*M_, D_, H_, W_)
        #value_l_.shape torch.Size([64, 32, 37, 38])
        #print("value_l_.shape",value_l_.shape)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        #print("sampling_grids[:, :, :, lid_].shape",sampling_grids[:, :, :, lid_].shape) #torch.Size([8, 300, 8, 4, 2])
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1) 
        if return_shift:
            sampling_grid_l_list.append(sampling_grid_l_)
        #print('sampling_grid_l_.shape',sampling_grid_l_.shape) 
        #encoder: sampling_grid_l_.shsape torch.Size([16, 10723, 4, 2])
        #decoder: sclaed_sample_location.shape torch.Size([16, 300, 4, 2])

        #encoder
        #torch.Size([8*8, 300, 4, 2]) #(B*head,300,4,2)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_) #[64,32,300,4]
        
        #print("sampling_value_l_",sampling_value_l_.shape)
        #sampling_value_l_ torch.Size([16, 32, 10723, 4]) 4个points
    #print("torch.stack(sampling_value_list, dim=-2).shape",torch.stack(sampling_value_list, dim=-2).shape)
    #torch.Size([B*num_heads, head_dim, 10723, num_levels, points]) 
    #4个level 
    #print("(torch.stack(sampling_value_list, dim=-2).flatten(-2).shape",(torch.stack(sampling_value_list, dim=-2).flatten(-2).shape))
    #[16, 32, 10723, 16])
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    #print("attention_weights.shape",attention_weights.shape)
    #attention_weights.shape torch.Size([2, 10723, 8, 4, 4])
    attention_weights = attention_weights.transpose(1, 2).contiguous().reshape(N_*M_, 1, Lq_, L_*P_)
    #print("attention_weights.shape",attention_weights.shape)
    #attention_weights.shape torch.Size([16, 1, 10723, 16])
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    if return_shift:
        return output.transpose(1, 2).contiguous(),torch.stack(sampling_grid_l_list,dim=0),torch.stack(scale_factors,dim=0)
    else:
        return output.transpose(1, 2).contiguous()