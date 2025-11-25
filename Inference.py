import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MFNet
from Dataset_and_utils import MultiSpectralDataset

def run_inference(test_loader, model, device):
        

    use_model_for_test = model #inference_model
    use_model_for_test.eval()

    predictions = []
    use_threshold = 0.099 #best_thr
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Predicting "):
            imgs = imgs.to(device)
            logits, _, _ = use_model_for_test(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
            preds = (probs > use_threshold).astype(np.uint8)
            for mask in preds:
                predictions.append(mask.squeeze().astype(np.uint8).flatten())


    #################
    # submission.csv
    #################
    
    # as per rules of the comeptition
    submission = pd.DataFrame({
        "id": np.arange(len(predictions)),
        "pixels": [",".join(map(str, p)) for p in predictions]
    })
    SUBMISSION_PATH = "submission.csv"
    submission.to_csv(SUBMISSION_PATH, index=False)
    print("Saved", SUBMISSION_PATH)

    print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_cached())# if deprecated, use torch.cuda.memory_reserved()
    try:
        print(torch.cuda.memory_cached())# if deprecated, use torch.cuda.max_memory_reserved()
    except Exception as e:
        print(torch.cuda.memory_reserved())
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    return

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BEST_MODEL_PATH = "/content/best_model.pth"

    X_test  = np.load('/root/.cache/kagglehub/competitions/kaggle-competition-dl-f-2025/X_test_256.npy', mmap_mode='r')
    print("Test shape:", X_test.shape)
    indices = np.arange(len(X_test))
    test_ds = MultiSpectralDataset(X_test, Y=None,indices = indices,compute_stats=True, augment=False)
    del X_test
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    model = MFNet(in_ch=16, n_class=1, use_se=True, deep_supervision=True).to(device)

    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    run_inference(test_loader, model, device)