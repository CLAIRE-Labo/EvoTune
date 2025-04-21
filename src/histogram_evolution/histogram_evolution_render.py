import math
import os

from manim import *
import numpy as np

from histogram_evolution.utils import load_all_programsbanks_scores


class HistogramEvolution(Scene):
    def construct(self):
        # Example data setup (replace with your real data):

        directory = "out/tsp/evotune"
        assert os.path.exists(directory)

        files = os.listdir(directory)
        files = [os.path.join(directory, file) for file in files]

        score_data_per_round_evo, round_nums = load_all_programsbanks_scores(files, score_threshold=-400,
                                                                             return_as_islands=False
                                                                             )
        directory = "out/tsp/funsearch"
        assert os.path.exists(directory)

        files = os.listdir(directory)
        files = [os.path.join(directory, file) for file in files]

        score_data_per_round_fun, _ = load_all_programsbanks_scores(files, score_threshold=-400,
                                                                    return_as_islands=False
                                                                    )

        score_data_per_round_evo = [[[-s / 100 for s in scores] for scores in all_scores] for all_scores in
                                    score_data_per_round_evo]
        score_data_per_round_fun = [[[-s / 100 for s in scores] for scores in all_scores] for all_scores in
                                    score_data_per_round_fun]

        # score_data_per_round = score_data_per_round_evo
        score_data_per_round = [
            [score_data_per_round_evo[round_idx][0], score_data_per_round_fun[round_idx][0]]
            for round_idx in range(len(score_data_per_round_evo))
        ]

        # Define histogram bins
        x_min, x_max, x_step = 2, 4, 0.2
        num_bins = 100

        bins = np.linspace(0.0, x_max, num_bins)

        num_islands = len(score_data_per_round[0])
        colors = [interpolate_color(BLUE, ORANGE, i / (num_islands - 1)) for i in
                  range(num_islands)]  # [RED, YELLOW, ORANGE, GREEN, BLUE, PURPLE]
        print(f'Found {num_islands} Histograms')

        # Precompute max height from all score datasets
        all_hist_values = [np.histogram(scores, bins=bins)[0] for all_scores in score_data_per_round for scores in
                           all_scores]
        global_max_height = max([h.max() for h in all_hist_values])
        print(f'Global max: {global_max_height}')

        global_y_max = int(int((global_max_height + 100) / 100) * 100)

        x_length, y_length = 10, 5

        scene_height_per_count = y_length / global_y_max

        # Set up axes
        axes = Axes(
            x_range=[x_min, x_max, x_step],
            y_range=[0, global_y_max, int(round(global_y_max / 5))],
            tips=False,
            axis_config={"include_numbers": True},
            x_length=x_length, y_length=y_length,
        ).to_edge(UP * 1.5)

        # Create axis titles
        x_label = Text("Optimality Gap", font_size=28)
        y_label = Text("Function Count", font_size=28)

        # Position them relative to axes
        x_label.next_to(axes.x_axis, DOWN, buff=0.4)
        y_label.next_to(axes.y_axis, buff=0.4).rotate(PI / 2).move_to(
            axes.c2p(0, global_y_max / 2) + LEFT * 1.5)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Text to show round number and total count
        round_text = Text(f"Round: {round_nums[0]}", font_size=24, font="Courier New").next_to(x_label, DOWN)

        # Initial histogram
        all_bars = VGroup()
        for hist_idx, histogram_score_data_per_round in enumerate(score_data_per_round[0]):
            hist_values, _ = np.histogram(histogram_score_data_per_round, bins=bins)
            bar_width = (bins[1] - bins[0]) * (
                    x_length / (x_max - x_min))  # Some spacing TODO: Figure out how to tweak this analytically
            bars = VGroup()
            for i, height in enumerate(hist_values):
                bar = Rectangle(
                    width=bar_width,
                    height=height * scene_height_per_count,  # scale height
                    fill_color=colors[hist_idx],
                    fill_opacity=0.7,
                    stroke_width=0
                )
                bar.next_to(
                    axes.c2p((bins[i] + bins[i + 1]) / 2, 0),
                    direction=UP,
                    buff=0
                )
                bars.add(bar)
            all_bars.add(bars)

        # Create legend in the top right corner
        use_islands = False

        legend_items = VGroup()
        for i, color in enumerate(colors):
            color_box = Square(side_length=0.3, fill_color=color, fill_opacity=1, stroke_width=0)
            if use_islands:
                label = Text(f"Island {i + 1}", font_size=20, font="Courier New")
            else:
                assert len(colors) == 2
                if i == 0:
                    label = Text(f"EvoTune", font_size=20, font="Courier New")
                else:
                    label = Text(f"FunSearch", font_size=20, font="Courier New")
            legend_item = VGroup(color_box, label).arrange(RIGHT, buff=0.3)
            legend_items.add(legend_item)

        # Arrange all legend items in a vertical stack and place in the top-right
        legend = legend_items.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.to_corner(UR, buff=0.5)

        self.play(Write(round_text), FadeIn(all_bars), Write(legend))

        for round_num, scores in zip(round_nums[1:], score_data_per_round[1:]):
            next_all_bars = VGroup()
            for hist_idx, histogram_score_data_per_round in enumerate(scores):

                next_hist_values, _ = np.histogram(histogram_score_data_per_round, bins=bins)

                # Recompute bars with updated axis scaling
                next_bars = VGroup()
                for i, height in enumerate(next_hist_values):
                    bar = Rectangle(
                        width=bar_width,
                        height=height * scene_height_per_count,  # dynamic scaling for height
                        fill_color=colors[hist_idx],
                        fill_opacity=0.7,
                        stroke_width=0
                    )
                    bar.next_to(
                        axes.c2p((bins[i] + bins[i + 1]) / 2, 0),
                        direction=UP,
                        buff=0
                    )
                    next_bars.add(bar)
                next_all_bars.add(next_bars)

            new_round_text = Text(f"Round: {round_num}", font_size=24, font="Courier New").next_to(x_label, DOWN)

            self.play(
                Transform(all_bars, next_all_bars),
                Transform(round_text, new_round_text),
                run_time=0.5
            )
            # self.wait(0.0)

        self.wait(5)
