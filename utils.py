from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import mlflow


def data_review(df):
    info = {
      'Number of Rows': df.shape[0],
      'Number of Columns': df.shape[1],
      'Missing Values': df.isnull().sum().sum(),
      'Duplicate Values': df.duplicated().sum(),
    }
    return info


def plot_series(series, array_like, label='', title=''):
    data_range = range(len(series))
    plt.plot(data_range, array_like, label=label)
    plt.title(title)
    plt.legend()
    plt.grid(True)


def calculate_mae_with_mlflow(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    return mae