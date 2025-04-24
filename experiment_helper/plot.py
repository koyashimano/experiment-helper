import os
from collections.abc import Callable
from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .types import (
    PlotCurve,
    PlotData,
    PlotDataWithDataFrame,
    PolyfitData,
)

DEFAULT_COLORS = [
    "blue",
    "red",
    "green",
    "purple",
    "orange",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "navy",
    "maroon",
    "mint",
    "azure",
    "beige",
    "coral",
]


class Plot:
    def __init__(
        self,
        *,
        data: list[PlotData],
        curves: list[PlotCurve] | None = None,
        x_axis: str,
        y_axis: str,
        fig_file_path: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        semilog: bool = False,
        transform_y: Callable[[float], float] | None = None,
        polar: bool = False,
        connect_first_and_last: bool | None = None,
    ):
        if connect_first_and_last is None:
            connect_first_and_last = polar
        self.data: list[PlotDataWithDataFrame] = []
        for d in data:
            assert d["file_path"].endswith(".csv")
            df = pd.read_csv(d["file_path"])
            if r := d.get("range"):
                df = df[(df[x_axis] >= r[0]) & (df[x_axis] <= r[1])]
            if transform_y:
                df[y_axis] = df[y_axis].apply(transform_y)
            if connect_first_and_last:
                df = pd.concat([df, df.iloc[[0]]])
            self.data.append(PlotDataWithDataFrame(df=df, **d))

        self.curves = curves or []

        self.x_axis = x_axis
        self.y_axis = y_axis
        self.xlim = xlim
        self.ylim = ylim
        self.polar = polar

        self.polyfit_data: dict[str, PolyfitData] = {}

        # 図のファイルパスを設定
        self.fig_file_path = (
            fig_file_path if fig_file_path else data[0]["file_path"].replace(".csv", ".png")
        )

        if self.polar:
            plt.figure(figsize=(10, 10))
        else:
            plt.figure(figsize=(10, 6))
            plt.xlabel(x_label if x_label else x_axis)
            plt.ylabel(y_label if y_label else y_axis)

        self.semilog = semilog

    def plot(self) -> None:
        self._set_lim()

        index = 0
        for data in self.data:
            self._plot_data(data, index)
            index += 1

        for curve in self.curves:
            self._plot_curve(curve, index)
            index += 1

        plt.grid(True)
        if any(d.get("label") is not None for d in self.data):
            plt.legend()

        if self.semilog:
            plt.yscale("log")

        os.makedirs(os.path.dirname(self.fig_file_path), exist_ok=True)
        plt.savefig(self.fig_file_path, dpi=300)
        plt.close()

    def _plot_data(self, data: PlotDataWithDataFrame, index: int) -> None:
        df = data["df"]
        if polyfit_ranges := data.get("polyfit_ranges"):
            self._polyfit(df, polyfit_ranges)

        if self.polar:
            plt.polar(
                np.deg2rad(df[self.x_axis]),
                df[self.y_axis],
                "o-",
                markersize=5,
                label=data.get("label"),
                color=data.get("color", DEFAULT_COLORS[index % len(DEFAULT_COLORS)]),
            )
        else:
            plt.plot(
                df[self.x_axis],
                df[self.y_axis],
                "o",
                markersize=5,
                label=data.get("label"),
                color=data.get("color", DEFAULT_COLORS[index % len(DEFAULT_COLORS)]),
            )

    def _plot_curve(self, data: PlotCurve, index: int) -> None:
        if self.polar:
            angle = np.linspace(0, 2 * np.pi, 100)
            vfunc = np.vectorize(data["func"])
            radius = vfunc(angle)
            plt.polar(
                angle,
                radius,
                "-",
                markersize=5,
                label=data.get("label"),
                color=data.get("color", DEFAULT_COLORS[index % len(DEFAULT_COLORS)]),
            )
        else:
            xlim = plt.xlim()
            x = np.linspace(xlim[0], xlim[1], 100)
            vfunc = np.vectorize(data["func"])
            y = vfunc(x)
            plt.plot(
                x,
                y,
                "-",
                markersize=5,
                label=data.get("label"),
                color=data.get("color", DEFAULT_COLORS[index % len(DEFAULT_COLORS)]),
            )

    def _polyfit(self, df: pd.DataFrame, polyfit_ranges: list[tuple[float, float]] | str) -> None:
        polyfit_df = df.copy()
        if self.semilog:
            polyfit_df[self.y_axis] = np.log10(polyfit_df[self.y_axis])
        if polyfit_ranges == "all":
            slope, intercept = np.polyfit(polyfit_df[self.x_axis], polyfit_df[self.y_axis], 1)
            self._plot_line(slope, intercept)
            self.polyfit_data[str(polyfit_ranges)] = {
                "slope": float(slope),
                "intercept": float(intercept),
            }
        else:
            for r in polyfit_ranges:
                df_line = polyfit_df[
                    (polyfit_df[self.x_axis] >= r[0]) & (polyfit_df[self.x_axis] <= r[1])
                ]
                slope, intercept = np.polyfit(df_line[self.x_axis], df_line[self.y_axis], 1)
                self._plot_line(slope, intercept)
                self.polyfit_data[str(r)] = {
                    "slope": float(slope),
                    "intercept": float(intercept),
                }

    def _plot_line(self, slope: float, intercept: float, color: str = "black") -> None:
        x_min, x_max = plt.xlim()
        if self.semilog:
            plt.plot(
                [x_min, x_max],
                [10 ** (slope * x_min + intercept), 10 ** (slope * x_max + intercept)],
                color=color,
            )
        else:
            plt.plot(
                [x_min, x_max],
                [slope * x_min + intercept, slope * x_max + intercept],
                color=color,
            )

    def _set_lim(self) -> None:
        if self.polar:
            return

        all_data = pd.concat([d["df"] for d in self.data])

        if self.xlim:
            plt.xlim(self.xlim)
        else:
            x_min = all_data[self.x_axis].min()
            x_max = all_data[self.x_axis].max()
            x_margin = (x_max - x_min) * 0.03
            xlim = (min(x_min - x_margin, 0), x_max + x_margin)
            plt.xlim(xlim)

        if self.ylim:
            plt.ylim(self.ylim)
        else:
            y_min = all_data[self.y_axis].min()
            y_max = all_data[self.y_axis].max()
            y_margin = (y_max - y_min) * 0.05
            ylim = (min(y_min - y_margin, 0), y_max + y_margin)
            plt.ylim(ylim)

    def print_polyfit_data(self) -> None:
        pprint(self.polyfit_data)
