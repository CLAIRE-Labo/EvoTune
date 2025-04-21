from manim import *

from code_evolution.utils import get_evenly_spaced_function_from_programbank, shorten_function_code


class CodeEvolution(Scene):
    def construct(self):
        # Title
        title = Text(
            "FlatPack Problem — Evolution of Best Function in an Island",
            font_size=28
        ).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Load from programbank:
        code_versions, code_scores, round_nums = get_evenly_spaced_function_from_programbank(
            ["out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_100.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_500.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_1000.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_1500.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_1600.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2000.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2300.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2600.pkl",
             "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2700.pkl",
             ], score_threshold=-20
        )
        code_versions = [shorten_function_code(code_version, 45, 0) for code_version in code_versions]
        code_scores = [round(-score / 100, 2) for score in code_scores]

        code_config = {
            "tab_width": 4,
            "language": "python",
            "add_line_numbers": False,
            "formatter_style": "monokai",
            "background": "window",
            "background_config": {"stroke_color": BLACK}
        }

        # First code block
        prev_code = Code(code_string=code_versions[0], **code_config).scale(0.25)

        max_score, min_score = code_scores[0], code_scores[-1]

        norm_score = (code_scores[0] - min_score) / (max_score - min_score)
        prev_score_text = Text(
            f"Optimality Gap: {code_scores[0]:.3f}",
            font_size=24,
            font="Courier New",
            color=interpolate_color(GREEN, RED, norm_score),
        ).next_to(prev_code, DOWN)

        prev_round_text = Text(
            f"Round: {round_nums[0]}",
            font_size=20,
            font="Courier New",
            color=WHITE,
        ).next_to(prev_score_text, DOWN)

        # Appear
        self.play(FadeIn(prev_code), Write(prev_score_text), Write(prev_round_text))
        self.wait(4)

        # Animation loop through versions
        for next_version, score, round_num in zip(code_versions[1:], code_scores[1:], round_nums[1:]):
            next_code = Code(code_string=next_version, **code_config).scale(0.25)

            norm_score = (score - min_score) / (max_score - min_score)
            color = interpolate_color(GREEN, RED, norm_score)

            next_score_text = Text(
                f"Optimality Gap: {score:.3f}",
                font_size=24,
                color=color,
                font="Courier New"
            ).next_to(next_code, DOWN)

            next_round_text = Text(
                f"Round: {round_num}",
                font_size=20,
                font="Courier New",
                color=WHITE,
            ).next_to(next_score_text, DOWN)

            self.play(
                Transform(prev_code, next_code),
                Transform(prev_score_text, next_score_text),
                Transform(prev_round_text, next_round_text),
            )
            self.wait(4)
