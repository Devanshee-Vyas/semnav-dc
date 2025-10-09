"""UNet baseline for depth completion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class UNetBaseline(nn.Module):
    """
    UNet with ResNet-34 encoder for depth completion.
    Input: 5 channels (RGB=3 + Sparse=1 + Mask=1)
    Output: 1 channel (dense depth in meters)
    """
    
    def __init__(self, encoder_name='resnet34', pretrained=True):
        super().__init__()
        
        # ResNet-34 encoder
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
        
        # Initialize new channels with existing weights (RGB) and zeros (sparse+mask)
        with torch.no_grad():
            self.encoder.conv1.weight[:, :3] = original_conv.weight
            self.encoder.conv1.weight[:, 3:] = 0.0
        
        # Get actual encoder channels dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 5, 224, 224)
            enc_features = self.encoder(dummy_input)
            enc_channels = [f.shape[1] for f in enc_features]
        
        # Decoder feature channels (need len(enc_channels)-1 decoder blocks)
        dec_channels = [256, 128, 64, 32]
        
        # Decoder blocks
        self.decoder = nn.ModuleList()
        for i in range(len(dec_channels)):
            # in_ch = upsampled_prev + skip_connection
            # For i=0: enc_channels[-1] upsampled + enc_channels[-2] skip
            # For i>0: dec_channels[i-1] upsampled + enc_channels[-(i+2)] skip
            if i == 0:
                in_ch = enc_channels[-1] + enc_channels[-2]
            else:
                in_ch = dec_channels[i-1] + enc_channels[-(i+2)]
            out_ch = dec_channels[i]
            self.decoder.append(DecoderBlock(in_ch, out_ch))
        
        # Final output conv
        self.out_conv = nn.Sequential(
            nn.Conv2d(dec_channels[-1], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.ReLU()  # Depth is positive
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 5, H, W) input tensor
            
        Returns:
            (B, 1, H, W) predicted depth
        """
        # Encode
        enc_features = self.encoder(x)
        
        # Decode
        d = enc_features[-1]
        for i, decoder_block in enumerate(self.decoder):
            enc_feat = enc_features[-(i+2)]
            d = decoder_block(d, enc_feat)
        
        # Output
        out = self.out_conv(d)
        
        # Upsample to match input size if needed
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        return out


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
        # Match sizes if needed
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
