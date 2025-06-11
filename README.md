# TNS_SAM
Thyroid nodule segment SAM model
# Thyroid Nodule Segmentation with MedSAM on TN3K Dataset

This repository provides the implementation of **MedSAM**, a medical image segmentation model, applied to the **TN3K public dataset** for thyroid nodule segmentation.

We fine-tuned MedSAM for precise segmentation of thyroid nodules in ultrasound images. Both quantitative and qualitative evaluations show that MedSAM achieves high performance across standard segmentation metrics.

---

## 🧪 Quantitative Evaluation (on TN3K Test Set)

| Metric        | Value  |
|---------------|--------|
| Dice (DSC)    | 0.8861 |
| IoU           | 0.8075 |
| Precision     | 0.9440 |
| Recall        | 0.8505 |
| Specificity   | 0.9926 |
| Accuracy      | 0.9805 |

---

## 🖼️ Qualitative Results

The following figure shows visual examples of thyroid nodule segmentation results on the TN3K test set.

### 📷 Sample Outputs

![Sample Results](/output.jpg)

> *Left: Input Ultrasound Image | Middle: Ground Truth Mask | Right: MedSAM Prediction*

To view more examples, see the [`tn3k_infer_results/`](results/) directory.

---

## 📂 Dataset

We used the **TN3K dataset** for thyroid ultrasound segmentation tasks.  
If you need access to the original dataset, please contact me at **424umar@gmail.com**.


## 🛠️ How to Reproduce

1. Clone this repo  
2. Prepare the TN3K dataset in the expected format  
3. Run the training script using our MedSAM configuration  
4. Visual and numerical results will be saved under the `results/` directory

---

## 📫 Contact

For questions or collaborations, please get in touch with 424umar@gmail.com & umarfarooq@hanyang.ac.kr or open an issue in this repository.
