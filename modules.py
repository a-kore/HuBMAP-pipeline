import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv2dEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 depth=1024,
                 num_res_layers=3
                 ):
        super(Conv2dEncoder, self).__init__()

        pass


class Conv2dDecoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 depth=1024,
                 num_res_layers=3
                 ):
        super(Conv2dDecoder, self).__init__()

        pass


class Conv1dEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 depth=1024,
                 num_res_layers=3
                 ):
        super(Conv1dEncoder, self).__init__()

        pass


class Conv1dDecoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 depth=1024,
                 num_res_layers=3
                 ):
        super(Conv1dDecoder, self).__init__()

        pass


class MultiModalEncoder(nn.Module):
    def __init__(self,
                 depth=1024,
                 video_channels=3,
                 audio_channels=2,
                 num_actions=43,
                 video_ksize=(18, 32),
                 video_stride=None,
                 audio_ksize=15,
                 audio_stride=None,
                 ):
        super(MultiModalEncoder, self).__init__()

        if video_stride:
            self.video_encoder = nn.Conv2d(video_channels, depth, video_ksize, video_stride)
        else:
            self.video_encoder = nn.Conv2d(video_channels, depth, video_ksize, video_ksize)

        if audio_stride:
            self.audio_encoder = nn.Conv1d(audio_channels, depth, audio_ksize, audio_stride)
        else:
            self.audio_encoder = nn.Conv1d(audio_channels, depth, audio_ksize, audio_ksize)

        self.actions_encoder = nn.Linear(num_actions, depth)

    def forward(self, multimodal_dict):
        video_encoding = self.video_encoder(multimodal_dict['video'])
        vid_dim = (video_encoding.shape[2], video_encoding.shape[3])
        video_encoding = video_encoding.flatten(2, 3).permute(0, 2, 1)
        audio_encoding = self.audio_encoder(multimodal_dict['audio']).permute(0, 2, 1)
        actions_encoding = self.actions_encoder(multimodal_dict['actions']).unsqueeze(1)
        encoding_dims = (video_encoding.shape[1],
                         audio_encoding.shape[1],
                         actions_encoding.shape[1])
        multimodal_concat = torch.cat([video_encoding,
                                       audio_encoding,
                                       actions_encoding], dim=1)
        return multimodal_concat, encoding_dims, vid_dim


class MultiModalDecoder(nn.Module):
    def __init__(self,
                 depth=1024,
                 video_channels=3,
                 audio_channels=2,
                 num_actions=43,
                 video_ksize=(18, 32),
                 video_stride=None,
                 audio_ksize=15,
                 audio_stride=None,
                 ):
        super(MultiModalDecoder, self).__init__()

        if video_stride:
            self.video_decoder = nn.ConvTranspose2d(depth, video_channels,
                                                    video_ksize, video_stride)
        else:
            self.video_decoder = nn.ConvTranspose2d(depth, video_channels,
                                                    video_ksize, video_ksize)

        if audio_stride:
            self.audio_decoder = nn.ConvTranspose1d(depth, audio_channels,
                                                    audio_ksize, audio_stride)
        else:
            self.audio_decoder = nn.ConvTranspose1d(depth, audio_channels,
                                                    audio_ksize, audio_ksize)

        self.actions_decoder = nn.Linear(depth, num_actions)

    def forward(self, z, multimodel_dict, encoding_dims, vid_dims):
        # recieves embedding of shape ()
        video_features = z[:, 0:encoding_dims[0], :]
        video_features = video_features.reshape(-1, vid_dims[0],
                                                vid_dims[1], z.shape[-1]).permute(0, 3, 1, 2)
        audio_features = z[:, encoding_dims[0]:encoding_dims[0] + encoding_dims[1], :].permute(0, 2, 1)
        actions_features = z[:, -1, :]
        video = self.video_decoder(video_features)
        video = video + multimodel_dict['video']
        audio = self.audio_decoder(audio_features) + multimodel_dict['audio']
        actions = self.actions_decoder(actions_features) + multimodel_dict['actions']
        return {'video': video,
                'audio': audio,
                'actions': actions}


class AttMMAE(nn.Module):
    def __init__(self,
                 depth=1024,
                 video_channels=3,
                 audio_channels=2,
                 num_actions=43,
                 video_ksize=(18, 32),
                 video_stride=None,
                 audio_ksize=15,
                 audio_stride=None,
                 num_layers=3,
                 nhead=1
                 ):
        super(AttMMAE, self).__init__()

        self.encoder = MultiModalEncoder(depth=depth, video_channels=video_channels,
                                         audio_channels=audio_channels, num_actions=num_actions,
                                         video_ksize=video_ksize, video_stride=video_stride,
                                         audio_ksize=audio_ksize, audio_stride=audio_stride)

        self.decoder = MultiModalDecoder(depth=depth, video_channels=video_channels,
                                         audio_channels=audio_channels, num_actions=num_actions,
                                         video_ksize=video_ksize, video_stride=video_stride,
                                         audio_ksize=audio_ksize, audio_stride=audio_stride)

        self.self_attn = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=depth, nhead=nhead),
                                               num_layers=num_layers)

    def forward(self, multimodal_dict):
        z, enc_dim, vid_dim = self.encoder(multimodal_dict)
        z = F.relu(self.self_attn(F.relu(z)))
        x_hat = self.decoder(z, multimodal_dict, enc_dim, vid_dim)
        return x_hat

