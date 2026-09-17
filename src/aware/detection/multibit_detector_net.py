import warnings
import torch
import torch.nn as nn
from aware.interfaces.detection import BaseDetectorNet
from aware.detection.modules import Conv1dBlock, MelFilterBankLayer, GlobalStandardize, BRH


class AWAREDetectorNet(BaseDetectorNet):

    """
    Neural network for watermark detection in audio spectrograms
    Uses mel filter bank followed by conv blocks with global average pooling

    `sample_rate` and `n_fft` define the reference (base) configuration. Other sample rates are
    supported through `forward(stft_magnitude, sample_rate)`: a mel filter bank is built for that
    rate with the same `fmin`/`fmax`, so mel channel i always covers the same Hz range and all
    convolutional weights are shared across rates.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 n_mels: int = 128,
                 fmin: float = 0.0,
                 fmax: float = None,
                 initial_pool_size: int = 2,
                 initial_pool_stride: int = 2,
                 num_blocks: int = 3,
                 n_filters: list[int] = [512, 1024, 1024],
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 norm_layer: str = 'instance',
                 activation: str = 'leaky_relu',
                 output_length: int = 20,
                 final_activation: str = 'tanh'):
        
        super(AWAREDetectorNet, self).__init__()
        
        assert len(n_filters) == num_blocks, "Number of filters must match number of blocks"
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        # Fixed mel range (defaults to the reference Nyquist) so every sample rate maps to the same mel channels
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else sample_rate / 2
        self.num_blocks = num_blocks
        self.initial_pool_size = initial_pool_size
        self.output_length = output_length

        self.global_norm1 = GlobalStandardize()

        # Mel filter bank as first layer
        self.mel_layer = MelFilterBankLayer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        # Mel filter banks for other sample rates, keyed by (sample_rate, n_fft).
        # Plain dict on purpose: they are derived buffers, not part of the model state.
        self._rate_mel_layers = {}

        self.instance_norm = nn.InstanceNorm1d(n_mels)
        self.global_norm2 = GlobalStandardize()

        self.initial_pool = nn.AvgPool1d(kernel_size=initial_pool_size, stride=initial_pool_stride)
        
        self.conv_blocks = nn.ModuleList()
        
        # Calculate channel dimensions for each block
        channels = [n_mels] + [n_filters[i] for i in range(num_blocks)] + [2*output_length]
        
        for i in range(num_blocks + 1):
            block = Conv1dBlock(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                norm_layer=norm_layer,
                in_channels=channels[i],
                out_channels=channels[i+1],
                activation=activation
            )
            self.conv_blocks.append(block)
        

        self.final_activation = self._get_activation(final_activation)
        # Bitwise Readout Head(BRH)
        self.BRH = BRH(self.final_activation)

        #Fixing seed for reproducibility
        torch.manual_seed(328656719)
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str):
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.2)
        elif activation.lower() == 'gelu':
            return nn.GELU()
        elif activation.lower() == 'swish':
            return nn.SiLU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Invalid activation: {activation}")
            
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.InstanceNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
         
    def _get_mel_layer(self, sample_rate: int, n_freq: int, device: torch.device) -> MelFilterBankLayer:
        """Return the mel filter bank for a sample rate and STFT size (n_freq = n_fft // 2 + 1)"""
        n_fft = 2 * (n_freq - 1)

        if sample_rate is None or sample_rate == self.sample_rate:
            if n_fft != self.n_fft:
                raise ValueError(f"STFT has {n_freq} frequency bins, expected {self.n_fft // 2 + 1} "
                                 f"for {self.sample_rate}Hz. Pass the sample rate of the audio.")
            return self.mel_layer

        key = (int(sample_rate), n_fft)
        mel_layer = self._rate_mel_layers.get(key)
        if mel_layer is None:
            with warnings.catch_warnings():
                # Below the reference rate, mel filters above Nyquist are empty by design
                warnings.filterwarnings("ignore", message="Empty filters detected")
                mel_layer = MelFilterBankLayer(sample_rate=int(sample_rate), n_fft=n_fft, n_mels=self.n_mels,
                                               fmin=self.fmin, fmax=self.fmax)
            self._rate_mel_layers[key] = mel_layer

        if mel_layer.mel_filter_bank.device != device:
            mel_layer.to(device)

        return mel_layer

    def forward(self, stft_magnitude, sample_rate: int = None):
        """
        Forward pass through the network

        Args:
            stft_magnitude: torch.Tensor of shape (batch_size, freq_bins, time_frames)
            sample_rate: sample rate of the audio the STFT was computed from (default: reference rate)

        Returns:
            torch.Tensor of shape (batch_size, num_classes)
        """
        mel_layer = self._get_mel_layer(sample_rate, stft_magnitude.shape[1], stft_magnitude.device)

        x = stft_magnitude
        # Standardize to zero mean, unit variance
        x = self.global_norm1(x)

        #convert to mel spectrogram
        x = mel_layer(stft_magnitude)
        
        x = self.instance_norm(x)
        # Standardize to zero mean, unit variance
        x = self.global_norm2(x)

        # Do initial pool
        x = self.initial_pool(x)
        
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Apply bitwise readout head
        x = self.BRH(x)

        return x


    def get_model_info(self):
        """Get information about model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'n_mels': self.n_mels,
            'mel_fmin': self.fmin,
            'mel_fmax': self.fmax,
            'cached_sample_rates': sorted({sr for sr, _ in self._rate_mel_layers}),
            'num_blocks': self.num_blocks,
            'output_length': self.output_length,
            'final_activation': self.final_activation,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
