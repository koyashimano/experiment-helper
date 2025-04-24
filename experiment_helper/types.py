from collections.abc import Callable
from typing import NotRequired, TypedDict

import pandas as pd


class PlotBase(TypedDict):
    label: NotRequired[str]
    color: NotRequired[str]


class PlotData(PlotBase):
    file_path: str
    range: NotRequired[tuple[float, float]]
    polyfit_ranges: NotRequired[list[tuple[float, float]] | str]


class PlotCurve(PlotBase):
    func: Callable[[float], float]


class PlotDataWithDataFrame(PlotData):
    df: pd.DataFrame


class PolyfitData(TypedDict):
    slope: float
    intercept: float
