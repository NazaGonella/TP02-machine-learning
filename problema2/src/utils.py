import numpy as np

def get_area_under_curve(x : np.ndarray[float], y : np.ndarray[float]) -> float:
    sorted_x : np.ndarray[float] = np.sort(x)
    sorted_y : np.ndarray[float] = np.sort(y)
    return np.trapz(y=sorted_y, x=sorted_x)