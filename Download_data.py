import kagglehub
import os

def setup_and_download():
    # Mount Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception:
        pass

    # Login to Kaggle
    kagglehub.login()

    # Download competition data
    kaggle_competition_dl_f_2025_path = kagglehub.competition_download('kaggle-competition-dl-f-2025')

    print('Data source import complete.')
    print(kaggle_competition_dl_f_2025_path)

    # List files
    for dirname, _, filenames in os.walk('/root/.cache/kagglehub/competitions'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            
    return 

if __name__ == "__main__":
    setup_and_download()
