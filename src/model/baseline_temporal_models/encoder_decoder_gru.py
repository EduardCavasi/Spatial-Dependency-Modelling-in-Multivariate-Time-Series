import torch
from torch import nn
from typing import Optional

from src.model.baseline_temporal_models.gru import GRUConfig
from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig

@dataclass
class EncoderDecoderGRUConfig(BasicTSModelConfig):
    input_size: int = 1 # this is the nr of features per input example at one timestamp
    input_len: int = 12 # this is only for compatibility with the runner (not used)
    output_len: int = 12
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    use_tod: bool = False
    use_dow: bool = False
    teacher_forcing: bool = False

class Encoder(nn.Module):
    def __init__(self, config: GRUConfig):
        super().__init__()
        assert config.input_size == 1 or config.input_size == 207, "either single sensor per example or all per example"

        self.use_dow = config.use_dow
        self.use_tod = config.use_tod
        self.nr_features = config.input_size + (2 if self.use_tod else 0) + (2 if self.use_dow else 0)
        self.input_size = config.input_size  
        self.gru = nn.GRU(
            input_size=self.nr_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert (self.use_tod or self.use_dow) == (inputs_timestamps is not None)
        B, L, S = inputs.shape
        if self.input_size == 1:
            # inputs = (batch_size, input_len, S) -> we want (batch_size * S, input_len, 1)
            x = inputs.permute(0, 2, 1).reshape(B * S, L, 1)

            t = inputs_timestamps.permute(0, 2, 1) if inputs_timestamps is not None else None # t = (batch_size, 2, input_len)

            features = [x]
            if self.use_tod:
                features.append(torch.sin(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, L, 1))
                features.append(torch.cos(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, L, 1))
            if self.use_dow:
                features.append(torch.sin(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, L, 1))
                features.append(torch.cos(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, L, 1))
        else:
            x = inputs
            features = [x]
            t = inputs_timestamps # t = (batch_size, input_len, 2)
            if self.use_tod:
                features.append(torch.sin(2 * torch.pi * t[:, :, 0:1]))
                features.append(torch.cos(2 * torch.pi * t[:, :, 0:1]))
            if self.use_dow:
                features.append(torch.sin(2 * torch.pi * t[:, :, 1:2]))
                features.append(torch.cos(2 * torch.pi * t[:, :, 1:2]))

        x = torch.cat(features, dim=-1)

        _, h = self.gru(x)
        # h will be the last activations and will have shape (nr_layers, batch_size, hidden_size)
        return h

class EncoderDecoderGRU(nn.Module):
    def __init__(self, config: EncoderDecoderGRUConfig):
        super().__init__()
        assert config.input_size == 1 or config.input_size == 207, "either single sensor per example or all per example"

        self.use_dow = config.use_dow
        self.use_tod = config.use_tod
        self.nr_features = config.input_size + (2 if self.use_tod else 0) + (2 if self.use_dow else 0)
        self.input_size = config.input_size

        self.output_len = config.output_len
        self.teacher_forcing = config.teacher_forcing

        self.encoder = Encoder(config)

        self.decoder = nn.GRU(
            input_size=self.nr_features,
            hidden_size=config.hidden_size, 
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(config.hidden_size, config.input_size)

    def _forward_teacher_forcing(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, targets_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, S = inputs.shape
        activations = self.encoder(inputs, inputs_timestamps)

        if self.input_size == 1:
            last_speed_input = inputs[:, -1, :].reshape(B * S, 1, 1)
            targets = targets.permute(0, 2, 1).reshape(B * S, self.output_len, 1)
            targets = torch.cat([last_speed_input, targets[:, :-1, :]], dim=1)
            features = [targets]
            t = targets_timestamps.permute(0, 2, 1) if targets_timestamps is not None else None
            if self.use_tod:
                tod_sin = torch.sin(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
                tod_cos = torch.cos(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
                features.append(tod_sin)
                features.append(tod_cos)
            if self.use_dow:
                dow_sin = torch.sin(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
                dow_cos = torch.cos(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
                features.append(dow_sin)
                features.append(dow_cos)
        else:
            last_speed_input = inputs[:, -1, :].reshape(B, 1, S)
            targets = torch.cat([last_speed_input, targets[:, :-1, :]], dim=-2)
            features = [targets]
            t = targets_timestamps
            if self.use_tod:
                features.append(torch.sin(2 * torch.pi * t[:, :, 0:1]))
                features.append(torch.cos(2 * torch.pi * t[:, :, 0:1]))
            if self.use_dow:
                features.append(torch.sin(2 * torch.pi * t[:, :, 1:2]))
                features.append(torch.cos(2 * torch.pi * t[:, :, 1:2]))
        
        decoder_input = torch.cat(features, dim=-1)
        out, _ = self.decoder(decoder_input, activations)

        y = self.fc(out)
        if self.input_size == 1:
            return y.reshape(B, S, -1).permute(0, 2, 1) # y = (batch_size, output_len, 207)
        else:
            return y
    
    def _forward_autoregressive(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None, targets_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, S = inputs.shape
        activations = self.encoder(inputs, inputs_timestamps)

        predictions = []

        if self.input_size == 1:
            # calculate output timestamps
            t = targets_timestamps.permute(0, 2, 1) if targets_timestamps is not None else None
            if self.use_tod:
                tod_sin = torch.sin(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
                tod_cos = torch.cos(2 * torch.pi * t[:, 0:1, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
            if self.use_dow:
                dow_sin = torch.sin(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
                dow_cos = torch.cos(2 * torch.pi * t[:, 1:2, :]).expand(-1, S, -1).reshape(B * S, self.output_len, 1)
        
            speed_input = inputs[:, -1, :].reshape(B * S, 1, 1)
         
        else:
            t = targets_timestamps
            if self.use_tod:
                tod_sin = torch.sin(2 * torch.pi * t[:, :, 0:1])
                tod_cos = torch.cos(2 * torch.pi * t[:, :, 0:1])

            if self.use_dow:
                dow_sin = torch.sin(2 * torch.pi * t[:, :, 1:2])
                dow_cos = torch.cos(2 * torch.pi * t[:, :, 1:2])

            speed_input = inputs[:, -1, :].reshape(B, 1, S)

        for i in range(self.output_len):
            features = [speed_input]
            if self.use_tod:
                features.append(tod_sin[:, i:i+1, :])
                features.append(tod_cos[:, i:i+1, :])

            if self.use_dow:
                features.append(dow_sin[:, i:i+1, :])
                features.append(dow_cos[:, i:i+1, :])

            decoder_input = torch.cat(features, dim=-1) # one -> (B * S, 1, nr_features) / all -> (B, S, nr_features)
        
            out, h = self.decoder(decoder_input, activations)
            prediction = self.fc(out) # out = one -> (B * S, 1, hidden_size) / all -> (B, 1, hidden_size)
            predictions.append(prediction)

            speed_input = prediction
            activations = h

        predictions = torch.cat(predictions, dim=-2)
        if self.input_size == 1:
            return predictions.reshape(B, S, self.output_len).permute(0, 2, 1)
        else:
            return predictions
    
    def forward(self, inputs: torch.Tensor, inputs_timestamps: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, targets_timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.teacher_forcing:
            if self.training:
                # we will use teacher forcing
                return self._forward_teacher_forcing(inputs, inputs_timestamps, targets, targets_timestamps)
            else:
                # autoregressive inference
                return self._forward_autoregressive(inputs, inputs_timestamps, targets_timestamps)
        else:
            return self._forward_autoregressive(inputs, inputs_timestamps, targets_timestamps)