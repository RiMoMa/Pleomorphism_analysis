from lifelines import CoxPHFitter
import pandas as pd

def survival_analysis(data_path):
    """Ejecuta una regresión de Cox para analizar sobrevida."""
    df = pd.read_csv(os.path.join(data_path, "metadata/survival_data.csv"))
    cph = CoxPHFitter()
    cph.fit(df, duration_col='time', event_col='event')
    cph.print_summary()
    cph.plot()

