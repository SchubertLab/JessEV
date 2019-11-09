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


def test_junction_cleavages():
    epi1, spacer, epi2 = 'LPVGAANFR', 'AAY', 'AMRIGAEVY'
    cleavages = position_cleavage(epi1 + spacer + epi2)

    model = DummyModel(
        spacer_length=3,
        epitopes = [epi1, epi2],
        x=get_x_for_epitopes([0, 1], 2),
        y=get_y_for_spacers([spacer]),
    )

    pre = StrobeSpacer.compute_pre_junction_cleavage(model, 0)
    assert abs(pre - cleavages[9]) < 1e-12

    post = StrobeSpacer.compute_post_junction_cleavage(model, 0)
    assert abs(post - cleavages[12]) < 1e-12


def test_max_epitope_cleavage():
    epi1, spacer1, epi2, spacer2, epi3 = 'LPVGAANFR', 'AAY', 'AMRIGAEVY', 'NFW', 'QANGWGVMV'
    cleavages = position_cleavage(epi1 + spacer1 + epi2 + spacer2 + epi3)

    model = DummyModel(
        spacer_length=3,
        epitopes = [epi1, epi2],
        x=get_x_for_epitopes([0, 1], 2),
        y=get_y_for_spacers([spacer1, spacer2]),
    )

    first_actual = StrobeSpacer.compute_max_epitope_cleavage(model, 0) 
    first_expected = max(cleavages[i] for i in range(9))
    assert abs(first_actual - first_expected) < 1e-12

    middle_actual = StrobeSpacer.compute_max_epitope_cleavage(model, 1)
    middle_expected = max(cleavages[i] for i in range(13, 21))
    assert abs(middle_actual - middle_expected) < 1e-12

    last_actual = StrobeSpacer.compute_max_epitope_cleavage(model, 2)
    last_expected = max(cleavages[i] for i in range(26, 33))


# def test_spacers_oneaa():
#     solver = StrobeSpacer(
#         epitope_immunogen=[1, 1],
#         epitope_cleavage_before=[10, 20],
#         epitope_cleavage_after=[20, 10],
#         spacer_length=2,
#         num_epitopes=2,
#         pcm_matrix=[
#             [0, 1, 3, 5, 7, 11, 13],
#         ]
#     ).build_model()

#     result = solver.solve()

#     assert result.epitopes == [1, 0]
#     assert result.spacers == [[0, 0]]
#     assert result.immunogen == 2
#     assert result.cleavage == -44


# def test_spacers_twoaa():
#     solver = StrobeSpacer(
#         epitope_immunogen=[1, 1],
#         epitope_cleavage_before=[10, 20],
#         epitope_cleavage_after=[20, 10],
#         spacer_length=2,
#         num_epitopes=2,
#         pcm_matrix=[
#             [0, 1, 3, 5, 7, 11, 13],
#             [0, -1, -3, -5, -7, -11, -13],
#         ]
#     ).build_model()

#     result = solver.solve()

#     assert result.epitopes == [1, 0]
#     assert result.spacers == [[1, 1]]
#     assert result.immunogen == 2
#     assert result.cleavage == 4