import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import scipy.io.wavfile
from tqdm import tqdm
import yaml
import argparse
import librosa  # For MFCC extraction

SAMPLE_RATE = 16000  # Expected audio duration: 1 second

def load_model(model_path):
    """Load a TensorFlow Lite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Expected input shape:", input_details[0]['shape'])  # Example: [1, 16, 96]
    return interpreter, input_details, output_details

def load_samples_from_config(config_path):
    """Load positive and negative samples as (filepath, label) from YAML config."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pos_dir = config.get("positive_dir")
    neg_dir = config.get("negative_dir")

    if not (pos_dir and neg_dir):
        raise ValueError("positive_dir and negative_dir must be defined in the config file.")

    pos_samples = [(os.path.join(pos_dir, f), 1) for f in os.listdir(pos_dir) if f.endswith(".wav")]
    neg_samples = [(os.path.join(neg_dir, f), 0) for f in os.listdir(neg_dir) if f.endswith(".wav")]

    return pos_samples + neg_samples

def preprocess_audio(audio, sr=16000):
    """Convert raw waveform to MFCC features with shape (16, 96)."""
    if len(audio) > sr:
        audio = audio[:sr]
    elif len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))

    audio = audio.astype(np.float32) / 32768.0

    # Compute 96 MFCC coefficients
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=96, n_fft=512, hop_length=512)
    mfcc = mfcc.T  # Shape: (time, n_mfcc)

    # Ensure fixed time steps: 16
    if mfcc.shape[0] < 16:
        mfcc = np.pad(mfcc, ((0, 16 - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:16, :]

    return mfcc  # Final shape: (16, 96)

def predict(interpreter, input_details, output_details, features):
    """Run inference on MFCC features."""
    input_data = np.expand_dims(features, axis=0)  # Shape: (1, 16, 96)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def evaluate_model(samples, interpreter, input_details, output_details):
    """Evaluate TFLite model on given samples."""
    y_true, y_pred = [], []

    for filepath, label in tqdm(samples, desc="Evaluating samples"):
        try:
            sr, audio = scipy.io.wavfile.read(filepath)
            features = preprocess_audio(audio, sr=sr)
            prediction = predict(interpreter, input_details, output_details, features)
            y_true.append(label)
            y_pred.append(prediction)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred_binary)
    tn = fp = fn = tp = 0

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        print(f"⚠️ Warning: Incomplete confusion matrix. Shape: {cm.shape}")
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn = cm[0][0]
            else:
                tp = cm[0][0]
        elif cm.shape == (1, 2):
            tn, fp = cm[0]
        elif cm.shape == (2, 1):
            tn = cm[0][0]
            fn = cm[1][0]

    metrics = {
        "tn": tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "true_positive_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
    }

    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        metrics.update({
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": auc(fpr, tpr)
        })
    else:
        print("⚠️ ROC AUC can't be calculated with only one class present in y_true.")
        metrics.update({
            "fpr": [0, 1],
            "tpr": [0, 1],
            "roc_auc": 0.0
        })

    return metrics

def plot_results(metrics):
    """Display evaluation metrics and ROC curve."""
    print("\nEvaluation Metrics:")
    print(f"True Positive: {metrics['tp']:.4f}")
    print(f"True Negative: {metrics['tn']:.4f}")
    print(f"False Positive: {metrics['fp']:.4f}")
    print(f"False Negative: {metrics['fn']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"True Positive Rate: {metrics['true_positive_rate']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")

    plt.figure()
    plt.plot(metrics['fpr'], metrics['tpr'], color='darkorange', lw=2,
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .tflite model")
    parser.add_argument("--config", type=str, required=True, help="YAML config path with sample directories")
    args = parser.parse_args()

    print("Loading model...")
    interpreter, input_details, output_details = load_model(args.model)

    print("Loading samples from config...")
    samples = load_samples_from_config(args.config)

    print("Evaluating...")
    metrics = evaluate_model(samples, interpreter, input_details, output_details)

    plot_results(metrics)