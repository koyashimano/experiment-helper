from typing import NotRequired, TypedDict

import pandas as pd


class PlotData(TypedDict):
    file_path: str
    label: NotRequired[str]
    range: NotRequired[tuple[float, float]]
    polyfit_ranges: NotRequired[list[tuple[float, float]] | str]
    color: NotRequired[str]


class PlotDataWithDataFrame(PlotData):
    df: pd.DataFrame


class PolyfitData(TypedDict):
    slope: float
    intercept: float
