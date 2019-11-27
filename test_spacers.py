from strobe_spacers import StrobeSpacer
from pcm import PCM_MATRIX, AMINOS, AMINO_IDX, get_pcm_matrix


def position_cleavage(seq):
    return [
        sum(
            PCM_MATRIX[AMINO_IDX[seq[j + i - 4]]][i]
            for i in range(6)
            if 0 <= j + i - 4 < len(seq)
        )
        for j in range(len(seq))
    ]


class DummyModel:
    def __init__(self, spacer_length=None, x=None, y=None, epitopes=None):
        self.x = x or {}
        self.y = y or {}
        self.PssmMatrix = {
            (i, j): row[j]
            for i, row in enumerate(get_pcm_matrix())
            for j in range(-4, 2)
        }

        if epitopes:
            self.Epitopes = list(range(len(epitopes)))
            self.EpitopeLength = len(epitopes[0])
            self.EpitopeSequences = {
                (i, j): AMINO_IDX[a]
                for i, epi in enumerate(epitopes)
                for j, a in enumerate(epi)
            }
        else:
            self.Epitopes = []
            self.EpitopeSequences = {}
            self.EpitopeLength = None

        self.SpacerLength = spacer_length or 0
        self.Aminoacids = list(range(len(AMINOS)))

        if self.x:
            self.VaccineLength = max(pos for _, pos in self.x.keys())


def get_x_for_epitopes(indices, num_epitopes):
    return {
        (i, j): int(indices[i] == j)
        for i in range(len(indices))
        for j in range(num_epitopes)
    }


def get_y_for_spacers(spacers):
    return {
        (i, j, k): int(spacers[i][j] == AMINOS[k])
        for i in range(len(spacers))
        for j in range(len(spacers[0]))
        for k in range(len(AMINOS))
    }


def junction_cleavages(epi1, spacer, epi2):
    cleavages = position_cleavage(epi1 + spacer + epi2)

    model = DummyModel(
        spacer_length=len(spacer),
        epitopes = [epi1, epi2],
        x=get_x_for_epitopes([0, 1], 2),
        y=get_y_for_spacers([spacer]),
    )

    pre = StrobeSpacer.compute_pre_junction_cleavage(model, 0)
    assert abs(pre - cleavages[9]) < 0.01

    post = StrobeSpacer.compute_post_junction_cleavage(model, 0)
    assert abs(post - cleavages[9 + len(spacer)]) < 0.01



def cleavage_within_epitope(epi1, spacer1, epi2, spacer2, epi3):
    cleavages = position_cleavage(epi1 + spacer1 + epi2 + spacer2 + epi3)

    model = DummyModel(
        spacer_length=len(spacer1),
        epitopes=[epi1, epi2, epi3],
        x=get_x_for_epitopes([0, 1, 2], 3),
        y=get_y_for_spacers([spacer1, spacer2]),
    )

    for epi in range(3):
        for pos in range(9):
            print('---')
            print(epi, pos)
            cleavage = StrobeSpacer.compute_cleavage_within_epitope(model, epi, pos)
            expected = cleavages[epi * (9 + len(spacer1)) + pos]
            assert abs(cleavage - expected) < 0.01


def test_junction_cleavages_long_spacers():
    junction_cleavages('LPVGAANFR', 'AAYAAY', 'AMRIGAEVY')


def test_junction_cleavages_mid_spacers():
    junction_cleavages('LPVGAANFR', 'AAYAAY', 'AMRIGAEVY')


def test_junction_cleavages_short_spacers():
    junction_cleavages('LPVGAANFR', 'Y', 'AMRIGAEVY')


def test_cleavage_within_epitope_lond_spacers():
    cleavage_within_epitope('LPVGAANFR', 'AAYAAY', 'AMRIGAEVY', 'NFWNFW', 'QANGWGVMV')


def test_cleavage_within_epitope_mid_spacers():
    cleavage_within_epitope('LPVGAANFR', 'AAY', 'AMRIGAEVY', 'NFW', 'QANGWGVMV')


def test_cleavage_within_epitope_short_spacers():
    cleavage_within_epitope('LPVGAANFR', 'Y', 'AMRIGAEVY', 'W', 'QANGWGVMV')