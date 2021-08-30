import re
import pandas as pd


def split_query(x):
    manufacturer = ''
    device_model = ''
    text = ''
    digit_pattern = r'\s\d{3,}\w{0,2}'
    if re.findall(digit_pattern, x):
        groups = re.split(digit_pattern, x)
        manufacturer = groups[0].strip()
        text = groups[1].strip()
        device_model = re.findall(digit_pattern, x)[0].strip()
    else:
        special_cases = ['burner', 'flow meter', 'hydrocarbon analyzer', 'liquid analyzer',
                         'pressure meter', 'temperature meter', 'viscosity meter', 'eho bettis',
                         'eho', 'level meter']
        for case in special_cases:
            result = re.match(case, x)
            if result is not None:
                manufacturer = case
                text = x[result.end() + 1:].strip()
                break

    return pd.Series([manufacturer, device_model, text])