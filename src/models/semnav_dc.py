"""Semantic-guided depth completion for navigation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.segmentation import deeplabv3_resnet50


class SemNavDC(nn.Module):
    """
    Semantic-guided depth completion with navigation awareness.
    
    Architecture:
    - UNet backbone (same as baseline)
    - Frozen DeepLabV3 semantic segmentation branch
    - Semantic gating at decoder scales
    - Optional free-space prediction head
    """
    
    def __init__(self, encoder_name='resnet34', pretrained=True, 
                 predict_freespace=True):
        super().__init__()
        
        self.predict_freespace = predict_freespace
        
        # UNet encoder (5 channels)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2, 3, 4]
        )
        
        # Adapt first conv to 5 channels
        original_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(
            5, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3] = original_conv.weight
            self.encoder.conv1.weight[:, 3:] = 0.0
        
        # Semantic segmentation branch (frozen)
        self.semantic = deeplabv3_resnet50(pretrained=True)
        for param in self.semantic.parameters():
            param.requires_grad = False
        self.semantic.eval()
        
        # Decoder
        enc_channels = [64, 64, 128, 256, 512]
        dec_channels = [256, 128, 64, 32, 16]
        
        self.decoder = nn.ModuleList()
        self.semantic_gates = nn.ModuleList()
        
        for i in range(len(dec_channels)):
            in_ch = enc_channels[-(i+1)] + (dec_channels[i-1] if i > 0 else enc_channels[-1])
            out_ch = dec_channels[i]
            
            self.decoder.append(DecoderBlock(in_ch, out_ch))
            
            # Semantic gating: 1x1 conv on semantic logits -> sigmoid
            self.semantic_gates.append(nn.Sequential(
                nn.Conv2d(21, out_ch, 1),  # COCO has 21 classes
                nn.Sigmoid()
            ))
        
        # Depth output head
        self.depth_head = nn.Sequential(
            nn.Conv2d(dec_channels[-1], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.ReLU()
        )
        
        # Free-space prediction head (optional)
        if self.predict_freespace:
            self.freespace_head = nn.Sequential(
                nn.Conv2d(dec_channels[-1] + 1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()  # Binary mask
            )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 5, H, W) input tensor
            
        Returns:
            dict with 'depth': (B, 1, H, W) and optionally 'freespace': (B, 1, H, W)
        """
        B, _, H, W = x.shape
        
        # Extract RGB for semantic branch (denormalize if needed, or use as-is)
        rgb = x[:, :3]  # First 3 channels
        
        # Semantic logits (frozen)
        with torch.no_grad():
            semantic_out = self.semantic(rgb)['out']  # (B, 21, H//8, W//8)
        
        # Encode depth features
        enc_features = self.encoder(x)
        
        # Decode with semantic gating
        d = enc_features[-1]
        for i, (decoder_block, gate) in enumerate(zip(self.decoder, self.semantic_gates)):
            enc_feat = enc_features[-(i+2)]
            d = decoder_block(d, enc_feat)
            
            # Apply semantic gate
            sem_up = F.interpolate(semantic_out, size=d.shape[-2:], 
                                  mode='bilinear', align_corners=True)
            gate_weights = gate(sem_up)
            d = d * gate_weights
        
        # Depth prediction
        depth = self.depth_head(d)
        
        # Upsample to input size if needed
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)
        
        output = {'depth': depth}
        
        # Free-space prediction
        if self.predict_freespace:
            fs_input = torch.cat([d, depth], dim=1)
            freespace = self.freespace_head(fs_input)
            if freespace.shape[-2:] != (H, W):
                freespace = F.interpolate(freespace, size=(H, W), 
                                        mode='bilinear', align_corners=True)
            output['freespace'] = freespace
        
        return output


class DecoderBlock(nn.Module):
    """Decoder block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
