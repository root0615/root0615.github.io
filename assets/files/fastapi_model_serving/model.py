import torch
import torch.nn as nn

class CNNAutoEncoder(nn.Module):

    def __init__(self, num_feature: int, target_len: int, latent_dim: int):
        # num_feature = 센서의 개수
        # latent_dim = 압축된 벡터의 크기 설정
        super().__init__()

        self.num_feature = num_feature
        self.target_len = target_len
        self.latent_dim = latent_dim

        # ── Encoder ──────────────────────────────────────────
        # (B, 8, 1498) → (B, 16, 749) → (B, 32, 374) → (B, 64, 187) → (B, 128, 93)
        self.encoder_conv = nn.Sequential(
            # Conv1d는 시계열에서 패턴을 찾는 필터
            # kernel_size=3 : 한번에 3개의 시간포인트를 보면서 패턴 탐색
            # stride=2 : 2칸씩 건너 뛰면서 이동 -> 길이가 절만으로 줄어듦
            # padding=1 : 데이터 양 끝에 0을 1칸 추가 -> 경계처리
            # 채널(필터 수)은 점점 늘어나면서(8 -> 16 -> 32 -> 64 ->128) 처음엔 단순한 패턴을, 깊어질수록 복잡한 패턴을 학습
            nn.Conv1d(num_feature, 16, kernel_size=3, stride=2, padding=1),  # 1498 → 749
            # BatchNorm1d : 데이터 정규화를 레이어 중간에 해주는 역할
            # 값의 범위를 적당히 유지시켜주고, 학습속도가 빨라지고 안정적이게 된다.
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 749 → 375
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 375 → 188
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # 188 → 94
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Conv 이후 실제 길이 계산
        self._enc_len = self._get_enc_len(target_len)
        self._flat_dim = 128 * self._enc_len

        # Flatten 후 latent vector로 압축
        self.encoder_fc = nn.Sequential(
            nn.Linear(self._flat_dim, latent_dim),
            nn.ReLU(),
        )

        # ── Decoder ──────────────────────────────────────────
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self._flat_dim),
            nn.ReLU(),
        )

        # (B, 128, enc_len) → 원래 길이 복원
        self.decoder_conv = nn.Sequential(
            # ConvTranspose1d는 Conv1d를 거꾸로 하는 연산. 길이를 늘리는 역할(복원)
            # output_padding=1 : stride=2로 늘릴때 홀수/짝수 문제로 길이가 1차이날 수 있어서 그걸 보정해주기 위한 옵션
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.ConvTranspose1d(16, num_feature, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def _get_enc_len(self, length: int) -> int:
        # Conv1d stride=2를 4번 적용후 실제 길이 계산
        for _ in range(4):
            # stride=2, padding=1, kernel=3
            length = (length + 2 * 1 - 3) // 2 + 1
        return length

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)  →  latent: (B, latent_dim)
        DataLoader에서 나오는 shape이 (B, T, C)이므로 Conv1d용으로 transpose
        """
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T): 1번 축과 2번축을 변경
        x = self.encoder_conv(x)  # (B, 128, enc_len)
        x = x.flatten(1)  # (B, flat_dim)
        x = self.encoder_fc(x)  # (B, latent_dim)
        return x

    def decode(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.decoder_fc(z)  # (B, flat_dim)
        x = x.view(x.size(0), 128, self._enc_len)  # (B, 128, enc_len)
        x = self.decoder_conv(x)  # (B, C, T')
        # ConvTranspose1d로 복원된 길이가 target_len과 다를 수 있어서 맞춰줌
        x = x[:, :, :target_len]  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        return x

    def forward(self, x: torch.Tensor) -> tuple:
        z = self.encode(x)
        recon = self.decode(z, x.size(1))
        return recon, z