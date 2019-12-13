from strobe_spacers import StrobeSpacer, MinimumCleavageGap
from pcm import DoennesKohlbacherPcm


def test_all():
    pcm = DoennesKohlbacherPcm()
    epitopes = ['MGNKWSKSK', 'MGNKWSKSI', 'ARHHKAREL', 'SSNTEATNA', 'NNCLLHPMS']
    immunogens = [0.082, 0.100, 0.115, 0.047, 0.016]

    gap = 0.6
    problem = StrobeSpacer(
        epitopes, immunogens,
        spacer_length=4,
        vaccine_length=2, 
        vaccine_constraints=MinimumCleavageGap(gap),
        pcm=pcm,
    )

    solution = problem.solve()

    correct_sequence = 'ARHHKARELCMMGNKWSKSI'
    correct_cleavages = pcm.cleavage_per_position(solution.sequence)

    assert solution.epitopes == [2, 1]
    assert solution.spacers == ['CM']
    assert solution.sequence == correct_sequence
    assert abs(solution.immunogen - 0.215) < 1e-6
    assert abs(sum(solution.cleavage) - sum(correct_cleavages)) < 1e-6
    assert all(solution.cleavage[13] >= solution.cleavage[i] + gap for i in range(14, 22))


if __name__ == '__main__':
    test_all()