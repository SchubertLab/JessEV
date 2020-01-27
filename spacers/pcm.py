class DoennesKohlbacherPcm:
    # https://onlinelibrary.wiley.com/doi/full/10.1110/ps.051352405
    AMINOS = 'ARNDCQEGHILKMFPSTWYV'
    AMINO_IDX = {a: i for i, a in enumerate(AMINOS)}
    PCM_MATRIX = [
        [-0.07, -0.18, -0.01, 0.22, 0.30, -0.18],
        [-0.31, -0.31, 0.38, 0.29, 0.20, -0.02],
        [-0.24, -0.12, 0.24, -0.71, 0.16, 0.57],
        [-0.34, -0.23, -0.81, 0.41, -0.34, 0.28],
        [0.16, -5.49, 0.16, -5.49, -5.49, -5.49],
        [-0.17, -0.08, 0.43, 0.07, -0.08, 0.14],
        [-0.03, -0.03, 0.04, 0.16, -0.27, -0.18],
        [0.32, -0.09, 0.22, -0.29, 0.22, 0.08],
        [0.12, 0.12, -0.16, -0.01, -0.16, 0.44],
        [0.05, -0.04, -0.73, -0.27, -1.64, 0.05],
        [-0.20, 0.13, -0.42, 0.38, -0.27, 0.22],
        [-0.23, 0.42, 0.04, -0.85, -0.02, -1.40],
        [0.22, 0.22, 0.22, 0.37, 0.63, -0.47],
        [0.13, 0.01, 0.23, 0.33, -0.96, 0.41],
        [0.22, -0.02, -1.29, -1.29, 0.18, -0.34],
        [0.04, -0.16, 0.20, -0.71, 0.34, -0.40],
        [-0.12, -0.23, 0.28, -0.48, 0.14, 0.14],
        [0.31, 0.31, -1.29, 1.00, 0.31, 0.64],
        [0.27, 0.05, 0.16, 0.80, 0.53, 0.16],
        [-0.14, 0.18, 0.13, 0.18, -0.96, -0.20],
    ]

    def encode_sequences(self, sequences):
        return [
            [self.get_index(a) for a in seq]
            for seq in sequences
        ]

    def get_pcm_matrix(self):
        return self.PCM_MATRIX

    def get_at_pcm(self, amino_idx, pos_idx):
        return self.PCM_MATRIX[amino_idx][pos_idx]

    def get_amino(self, idx):
        return self.AMINOS[idx]

    def get_index(self, amino):
        return self.AMINO_IDX[amino]

    def cleavage_per_position(self, sequence):
        return [
            sum(
                self.PCM_MATRIX[self.AMINO_IDX[sequence[j + i - 4]]][i]
                for i in range(6)
                if 0 <= j + i - 4 < len(sequence)
            )
            for j in range(len(sequence))
        ]
