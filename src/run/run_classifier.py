import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Paths
data_dir = "/archive/home/mdelsant/PEARL/resources/data/output/feature_dataset"
output_dir = "/archive/home/mdelsant/PEARL/resources/data/output/classifier_results"
os.makedirs(output_dir, exist_ok=True)

train_file = os.path.join(data_dir, "features_train.csv")
dev_file = os.path.join(data_dir, "features_dev.csv")
test_file = os.path.join(data_dir, "features_test.csv")

# Feature selection configuration
# Options: 'AD', 'CN', 'DELTA', 'AD_CN', 'AD_DELTA', 'CN_DELTA', 'AD_CN_DELTA'
FEATURE_SELECTION = 'DELTA'


def select_features(df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    """
    Seleziona le feature in base al tipo specificato.
    
    Args:
        df: DataFrame con tutte le feature
        feature_type: Tipo di feature da selezionare
                     'AD', 'CN', 'DELTA', oppure combinazioni con '_'
    
    Returns:
        DataFrame con solo le feature selezionate
    """
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'disease', 'patient_id', 'label']]
    
    selected_cols = []
    types = feature_type.split('_')
    
    for col in feature_cols:
        col_upper = col.upper()
        for ftype in types:
            if ftype in col_upper:
                selected_cols.append(col)
                break
    
    return df[selected_cols] if selected_cols else df[feature_cols]


class ClassifierEvaluator:
    """Gestisce training, validation e evaluation di classificatori."""
    
    # Mapping label test set
    TEST_LABEL_MAPPING = {
        'Non Healthy': 'AD',
        'Control': 'CN'
    }
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Carica train, dev e test data con selezione feature."""
        print(f"\n{'='*60}")
        print(f"Caricamento dati per {self.name}")
        print(f"Selezione feature: {FEATURE_SELECTION}")
        print(f"{'='*60}")
        
        # Load train
        train_df = pd.read_csv(train_file)
        print(f"Train: {train_df.shape[0]} campioni")
        
        # Load dev
        dev_df = pd.read_csv(dev_file)
        print(f"Dev: {dev_df.shape[0]} campioni")
        
        # Load test
        test_df = pd.read_csv(test_file)
        print(f"Test: {test_df.shape[0]} campioni")
        
        # Combine train + dev per training (come comune practice)
        combined_df = pd.concat([train_df, dev_df], ignore_index=True)
        print(f"Train+Dev combinati: {combined_df.shape[0]} campioni")
        
        # Extract features con selezione
        X_combined = select_features(combined_df, FEATURE_SELECTION).values
        y_combined = combined_df['disease'].values
        
        X_test = select_features(test_df, FEATURE_SELECTION).values
        y_test = test_df['label'].values
        
        print(f"Feature selezionate: {X_combined.shape[1]}")
        
        # Map test labels
        y_test_mapped = np.array([self.TEST_LABEL_MAPPING.get(label, label) for label in y_test])
        
        # Encode labels
        y_combined_encoded = self.label_encoder.fit_transform(y_combined)
        y_test_encoded = np.array([
            self.label_encoder.transform([label])[0] if label in self.label_encoder.classes_
            else 0  # fallback
            for label in y_test_mapped
        ])
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class distribution (train+dev): {np.bincount(y_combined_encoded)}")
        print(f"Class distribution (test): {np.bincount(y_test_encoded)}")
        
        return X_combined, y_combined_encoded, X_test, y_test_encoded, test_df['patient_id'].values
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Allena il modello."""
        print(f"\nTraining {self.name}...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        print(f"✓ {self.name} trained")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str = "Test") -> Dict:
        """Valuta il modello."""
        X_scaled = self.scaler.transform(X)
        
        # Predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        prec_per_class = precision_score(y, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y, y_pred, average=None, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
        except:
            roc_auc = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        results = {
            'accuracy': acc,
            'precision_weighted': prec,
            'recall_weighted': rec,
            'f1_weighted': f1,
            'roc_auc': roc_auc,
            'precision_per_class': prec_per_class,
            'recall_per_class': rec_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'y_true': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }
        
        self.results[set_name] = results
        return results
    
    def print_report(self):
        """Stampa report dettagliato."""
        print(f"\n{'='*60}")
        print(f"REPORT: {self.name}")
        print(f"{'='*60}")
        
        for set_name, metrics in self.results.items():
            print(f"\n{set_name} Set:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision (weighted): {metrics['precision_weighted']:.4f}")
            print(f"  Recall (weighted):    {metrics['recall_weighted']:.4f}")
            print(f"  F1 (weighted):        {metrics['f1_weighted']:.4f}")
            if not np.isnan(metrics['roc_auc']):
                print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            # Per-class
            print(f"\n  Per-class metrics:")
            for i, cls in enumerate(self.label_encoder.classes_):
                print(f"    Class '{cls}':")
                print(f"      Precision: {metrics['precision_per_class'][i]:.4f}")
                print(f"      Recall:    {metrics['recall_per_class'][i]:.4f}")
                print(f"      F1:        {metrics['f1_per_class'][i]:.4f}")
            
            # Confusion matrix
            print(f"\n  Confusion Matrix:")
            print(f"    {metrics['confusion_matrix']}")
    
    def save_report_csv(self):
        """Salva report in CSV."""
        report_file = os.path.join(output_dir, f"report_{self.name}.csv")
        
        rows = []
        for set_name, metrics in self.results.items():
            row = {
                'Classifier': self.name,
                'Dataset': set_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision_weighted': f"{metrics['precision_weighted']:.4f}",
                'Recall_weighted': f"{metrics['recall_weighted']:.4f}",
                'F1_weighted': f"{metrics['f1_weighted']:.4f}",
                'ROC_AUC': f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else 'N/A',
            }
            
            # Add per-class metrics
            for i, cls in enumerate(self.label_encoder.classes_):
                row[f'Precision_{cls}'] = f"{metrics['precision_per_class'][i]:.4f}"
                row[f'Recall_{cls}'] = f"{metrics['recall_per_class'][i]:.4f}"
                row[f'F1_{cls}'] = f"{metrics['f1_per_class'][i]:.4f}"
            
            rows.append(row)
        
        df_report = pd.DataFrame(rows)
        df_report.to_csv(report_file, index=False)
        print(f"✓ Report salvato: {report_file}")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix."""
        fig, axes = plt.subplots(1, len(self.results), figsize=(12, 4))
        
        if len(self.results) == 1:
            axes = [axes]
        
        for ax, (set_name, metrics) in zip(axes, self.results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            ax.set_title(f"{set_name} - Confusion Matrix")
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"confusion_matrix_{self.name}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Plot salvato: {plot_file}")


def main():
    print("="*60)
    print("CLASSIFICATORE MULTI-MODELLO")
    print("="*60)
    
    # Crea valutatori per diversi modelli
    models_config = [
        ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)),
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=150, max_depth=7, random_state=42)),
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
        ('SVM_RBF', SVC(kernel='rbf', probability=True, random_state=42)),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ]
    
    # Inizializza lista di evaluators
    evaluators_list = []
    
    # Load data una volta sola (dal primo modello)
    first_evaluator = ClassifierEvaluator('Temp')
    X_combined, y_combined, X_test, y_test, patient_ids = first_evaluator.load_data()
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )
    
    print(f"\nTrain set: {X_train.shape[0]} campioni")
    print(f"Validation set: {X_val.shape[0]} campioni")
    
    # Train e valuta ogni modello
    for model_name, model in models_config:
        evaluator = ClassifierEvaluator(model_name)
        evaluator.model = model
        
        # Copy encoder dal primo evaluator
        evaluator.label_encoder = first_evaluator.label_encoder
        
        # Train
        evaluator.train(X_train, y_train)
        
        # Evaluate on validation e test
        evaluator.evaluate(X_val, y_val, set_name='Validation')
        evaluator.evaluate(X_test, y_test, set_name='Test')
        
        # Report
        evaluator.print_report()
        evaluator.save_report_csv()
        evaluator.plot_confusion_matrix()
        
        evaluators_list.append(evaluator)
    
    # Comparative report
    print(f"\n{'='*60}")
    print("COMPARATIVE REPORT - TEST SET")
    print(f"{'='*60}")
    
    comparative_data = []
    for evaluator in evaluators_list:
        if 'Test' in evaluator.results:
            metrics = evaluator.results['Test']
            comparative_data.append({
                'Classifier': evaluator.name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision_weighted']:.4f}",
                'Recall': f"{metrics['recall_weighted']:.4f}",
                'F1': f"{metrics['f1_weighted']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else 'N/A',
            })
    
    comp_df = pd.DataFrame(comparative_data)
    print(comp_df.to_string(index=False))
    
    comp_file = os.path.join(output_dir, "comparative_report.csv")
    comp_df.to_csv(comp_file, index=False)
    print(f"\n✓ Comparative report salvato: {comp_file}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_keys = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    
    for idx, (ax, metric_name, metric_key) in enumerate(zip(axes.flat, metrics_names, metrics_keys)):
        values = []
        names = []
        for evaluator in evaluators_list:
            if 'Test' in evaluator.results:
                values.append(evaluator.results['Test'][metric_key])
                names.append(evaluator.name)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
        ax.bar(names, values, color=colors)
        ax.set_ylabel(metric_name)
        ax.set_ylim([0, 1])
        ax.set_title(f'{metric_name} - Test Set')
        ax.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Rotate labels
        ax.set_xticklabels(names, rotation=45, ha='right')
    
    plt.tight_layout()
    comp_plot = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(comp_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot salvato: {comp_plot}")
    
    # Salva predizioni del miglior modello
    best_idx = np.argmax([evaluators_list[i].results['Test']['f1_weighted'] for i in range(len(evaluators_list))])
    best_evaluator = evaluators_list[best_idx]
    
    print(f"\n{'='*60}")
    print(f"Miglior modello: {best_evaluator.name}")
    print(f"{'='*60}")
    
    # Salva predizioni test
    test_predictions = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': best_evaluator.label_encoder.inverse_transform(best_evaluator.results['Test']['y_true']),
        'predicted_label': best_evaluator.label_encoder.inverse_transform(best_evaluator.results['Test']['y_pred']),
        'confidence': best_evaluator.results['Test']['y_pred_proba'],
    })
    
    pred_file = os.path.join(output_dir, f"predictions_{best_evaluator.name}.csv")
    test_predictions.to_csv(pred_file, index=False)
    print(f"✓ Predizioni test salvate: {pred_file}")
    
    print(f"\n{'='*60}")
    print(f"Tutti i risultati salvati in: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
