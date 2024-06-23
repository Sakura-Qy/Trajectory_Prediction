import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer, CarAttention, CarProbAttention
from models.embed import DataEmbedding, PositionalEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # LSTM
        self.lstm = nn.GRU(enc_in, 1024, 2, dropout=dropout, batch_first=True)
        self.hidden_out = nn.Linear(1024, enc_in)

        # each car attention layer
        attr_index = [[3, 14], [23, 30], [30, 37], [37, 44], [44, 51], [51, 58], [58, 65]]
        self.car_attention = CarAttention(attr_index, factor, dropout, 18, 6)

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # h0 = torch.zeros(2, x_enc.size(1), 2)
        # c0 = torch.zeros(2, x_enc.size(1), 2)
        # out, _ = self.lstm(x_enc, (h0, c0))

        # out, _ = self.lstm(x_enc)
        # out = self.hidden_out(out)
        # x_enc = x_enc + out

        x_enc1 = self.car_attention(x_enc)

        enc_out = self.enc_embedding(x_enc + x_enc1, x_mark_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)     # 维度不一样，
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]

class CarAttentionLstm(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CarAttentionLstm, self).__init__()
        self.pred_len = out_len
        attr_index = [[3, 14], [23, 30], [30, 37], [37, 44], [44, 51], [51, 58], [58, 65]]
        self.car_attention = CarAttention(attr_index, factor, dropout, 18, 6)

        self.lstm = nn.LSTM(enc_in, 1024, 3, dropout=dropout, batch_first=True)
        self.hidden_out = nn.Linear(1024, 4)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc1 = self.car_attention(x_enc)
        r_out, _ = self.lstm(x_enc+x_enc1)
        out = self.hidden_out(r_out)

        return out.view(-1, self.pred_len, 2)

class CarAttentionGru(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CarAttentionGru, self).__init__()
        self.pred_len = out_len
        attr_index = [[3, 14], [23, 30], [30, 37], [37, 44], [44, 51], [51, 58], [58, 65]]
        self.car_attention = CarAttention(attr_index, factor, dropout, 18, 6)

        self.gru = nn.GRU(enc_in, 1024, 3, dropout=dropout, batch_first=True)
        self.hidden_out = nn.Linear(1024, 4)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc1 = self.car_attention(x_enc)
        r_out, _ = self.gru(x_enc+x_enc1)
        out = self.hidden_out(r_out)

        return out.view(-1, self.pred_len, 2)

class CarProbAttentionGru(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CarProbAttentionGru, self).__init__()
        self.pred_len = out_len
        attr_index = [[3, 14], [23, 30], [30, 37], [37, 44], [44, 51], [51, 58], [58, 65]]
        self.car_attention = CarProbAttention(attr_index, factor, dropout, 18, 6)

        self.lstm = nn.GRU(enc_in, 1024, 3, dropout=dropout, batch_first=True)
        self.hidden_out = nn.Linear(1024, 4)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc1 = self.car_attention(x_enc)
        r_out, _ = self.lstm(x_enc + x_enc1)
        out = self.hidden_out(r_out)

        return out.view(-1, self.pred_len, 2)     # [B, L, D]

class CarEmbedProbAttentionGru(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CarEmbedProbAttentionGru, self).__init__()
        self.pred_len = out_len

        self.enc_embedding = PositionalEmbedding(enc_in)
        attr_index = [[3, 14], [23, 30], [30, 37], [37, 44], [44, 51], [51, 58], [58, 65]]
        self.car_attention = CarProbAttention(attr_index, factor, dropout, 18, 6)

        self.gru = nn.GRU(enc_in, 1024, 3, dropout=dropout, batch_first=True)
        self.hidden_out = nn.Linear(1024, 4)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc = x_enc + self.enc_embedding(x_enc)
        x_enc1 = self.car_attention(x_enc)
        r_out, _ = self.gru(x_enc + x_enc1)
        out = self.hidden_out(r_out)

        return out.view(-1, self.pred_len, 2)     # [B, L, D]

class CarPAGpdecoder(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(CarPAGpdecoder, self).__init__()
        self.pred_len = out_len
        self.enc_embedding = PositionalEmbedding(enc_in)
        attr_index = [[3, 14], [23, 30], [30, 37], [37, 44], [44, 51], [51, 58], [58, 65]]
        self.car_attention = CarProbAttention(attr_index, factor, dropout, 18, 6)
        self.hidden_out = nn.Linear(1024, d_model)

        # self.lstm = nn.LSTM(enc_in, 1024, 3, dropout=dropout, batch_first=True)

        self.batch_norm = nn.BatchNorm1d(label_len + out_len)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(ProbAttention(True, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # encoder
        x_enc = x_enc + self.enc_embedding(x_enc)
        x_enc1 = self.car_attention(x_enc)
        # r_out, _ = self.lstm(x_enc + x_enc1)

        enc_out = self.hidden_out(x_enc + x_enc1)

        # decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

