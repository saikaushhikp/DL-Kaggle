# DL-MultiSpectral-Image-Segmentation
 Contains implementation of the Kaggle competition conducted as a part of the DL coursework here at IIT Hyderabad

commands to run the code(ofcouse you should have access to the dataset via Kaggle login credentials):
1. Clone the repository
   ```bash
   git clone https://github.com/saikaushhikp/DL-MultiSpectral-Image-Segmentation-Kaggle-Task.git
   ```
2. Navigate to the repository folder
   ```bash
   cd DL-MultiSpectral-Image-Segmentation-Kaggle-Task
   ```
3. Run the training scripts (can be completely run via the [`Note-book.ipynb`](Note-book.ipynb) as well)
   ```bash
   python3 Download_data.py
   python3 Train.py
   # to make predictions on test data
   python3 Inference.py
   ```
