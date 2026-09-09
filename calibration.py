from os import mkdir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import joblib
import os
import argparse


class PlattCalibration:

    def __init__(self):
        self.calibrator = LogisticRegression(solver='lbfgs')

    def fit(self, pred_probs, true_labels):
        X = pred_probs.reshape(-1, 1)
        self.calibrator.fit(X, true_labels)
        return self

    def predict_proba(self, pred_probs):
        X = pred_probs.reshape(-1, 1)

        return self.calibrator.predict_proba(X)[:, 1]

    def save(self, filepath='platt_model.pkl'):

        joblib.dump(self, filepath)


    @classmethod
    def load(cls, filepath='platt_model.pkl'):

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model does not exist: {filepath}")

        model = joblib.load(filepath)
        print(f"Platt model is loaded from {filepath}")
        return model



class IsotonicCalibration:

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds='clip')

    def fit(self, pred_probs, true_labels):
        self.calibrator.fit(pred_probs, true_labels)
        return self

    def predict_proba(self, pred_probs):

        return self.calibrator.predict(pred_probs)

    def save(self, filepath='isotonic_model.pkl'):

        joblib.dump(self, filepath)
        print(f"Model has been saved to: {filepath}")

    @classmethod
    def load(cls, filepath='isotonic_model.pkl'):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model does not exist: {filepath}")

        model = joblib.load(filepath)
        print(f"Isotonic model is loaded from {filepath}")
        return model



def load_data(file_path,  pred_column,  class_number=None,expert_columns=None,model_type='platt'):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model does not exist: {file_path}")

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Only accept csv or excel")

    if pred_column not in df.columns:
        if pred_column=='class_1_prob' and 'pred' in df.columns:
            pred_probs = df['pred'].values
        else:
            raise ValueError(f"Missing columns: {pred_column}")
    else:
        pred_probs = df[pred_column].values


    if class_number is not None:
        if f'soft_label_{class_number}' not in df.columns and expert_columns is not None:
            class_votes = (df[expert_columns] == class_number).sum(axis=1)
            total_valid_votes = df[expert_columns].notna().sum(axis=1)

            df[f'soft_label_{class_number}'] = np.where(total_valid_votes > 0,
                                               class_votes / total_valid_votes,
                                               0)
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Only accept csv or excel")

        true_labels = df[f'soft_label_{class_number}'].values

    elif expert_columns is not None:
        class_votes = (df[expert_columns] == 1).sum(axis=1)
        total_valid_votes = df[expert_columns].notna().sum(axis=1)

        df[f'soft_label'] = np.where(total_valid_votes > 0,
                                                    class_votes / total_valid_votes,
                                                    0)

        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif file_path.endswith('.xlsx'):
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Only accept csv or excel")

        true_labels = df['soft_label'].values
    else:
        true_labels = df['soft_score'].values


    if model_type=='platt':
        unique_values = np.unique(true_labels)
        if len(unique_values) > 2 or not np.all(np.isin(unique_values, [0, 1])):
            print("Warning: Continuous or non-binary labels detected. Binarizing labels (threshold: 0.5)..")
            true_labels = (true_labels >=0.5).astype(int)

    print(f"Loaded {len(pred_probs)} samples")
    return pred_probs, true_labels



def evaluate_calibration(y_true, original_probs, platt_probs, isotonic_probs, fig_path, class_name='', save_fig=True):

    unique_values = np.unique(y_true)
    if len(unique_values) > 2 or not np.all(np.isin(unique_values, [0, 1])):
        print("Warning: Continuous or non-binary labels detected. Binarizing labels (threshold: 0.5)..")
        y_true = (y_true >= 0.5).astype(int)

    brier_orig = brier_score_loss(y_true, original_probs)
    print(f"\n== Calibration evaluation results ==")
    print(f"Brier score of the original predictions: {brier_orig:.3f}")

    plt.figure(figsize=(10, 8))

    prob_true_orig, prob_pred_orig = calibration_curve(y_true, original_probs, n_bins=10)
    plt.plot(prob_pred_orig, prob_true_orig, linewidth=2, color='steelblue',marker='o',
             label=f'Original {brier_orig:.3f}')

    if platt_probs is not None:
        brier_platt = brier_score_loss(y_true, platt_probs)
        print(f"Brier score after Platt scaling: {brier_platt:.3f}")

        prob_true_platt, prob_pred_platt = calibration_curve(y_true, platt_probs, n_bins=10)
        plt.plot(prob_pred_platt, prob_true_platt, linewidth=2, marker='o', color='red',
                 label=f'Platt {brier_platt:.3f}')

    if isotonic_probs is not None:
        brier_isotonic = brier_score_loss(y_true, isotonic_probs)
        print(f"Brier score after isotonic regression: {brier_isotonic:.3f}")

        prob_true_isotonic, prob_pred_isotonic = calibration_curve(y_true, isotonic_probs, n_bins=10)
        plt.plot(prob_pred_isotonic, prob_true_isotonic, linewidth=2,marker='o', color='black',
                 label=f'Isotonic {brier_isotonic:.3f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='')

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.grid(False)

    plt.xlabel('Predicted Probability', fontsize=16)
    plt.ylabel('True Probability', fontsize=16)

    plt.legend(
        loc='lower right',
        fontsize=24,
        frameon=False,
        title=class_name,
        title_fontsize=26, handlelength=1, handletextpad=1,bbox_to_anchor=(1.1, 0))

    if save_fig:
        plt.savefig(fig_path)
        print(f"Calibration curve has been saved to: {fig_path}")




def train_calibration_models(file_path, pred_column, calibration_task_dir,class_number=None,class_name=None, expert_columns = None, test_size=0.1, random_state=42):

    print("\n== Training calibration model ==")

    pred_probs, true_labels = load_data(file_path=file_path,
                                        pred_column=pred_column,
                                        class_number=class_number,
                                        expert_columns=expert_columns,
                                        model_type='platt')

    X_train, X_test, y_train, y_test = train_test_split(
        pred_probs, true_labels, test_size=test_size, random_state=random_state
    )
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")

    print("\nTraining Platt scaling model...")
    platt_model = PlattCalibration()
    platt_model.fit(X_train, y_train)

    print("\nTesting Platt scaling model...")
    platt_probs = platt_model.predict_proba(X_test)


    pred_probs, true_labels = load_data(file_path=file_path,
                                        pred_column=pred_column,
                                        class_number=class_number,
                                        expert_columns=expert_columns,
                                        model_type='isotonic')

    X_train, X_test, y_train, y_test_iso= train_test_split(
        pred_probs, true_labels, test_size=test_size, random_state=random_state
    )

    print("\nTraining isotonic regression model...")
    isotonic_model = IsotonicCalibration()
    isotonic_model.fit(X_train, y_train)

    print("\nTesting isotonic regression model...")
    isotonic_probs = isotonic_model.predict_proba(X_test)

    fig_path=os.path.join(calibration_task_dir, 'results', f'calibration_{args.class_number}.png')
    evaluate_calibration(y_true= y_test,
                         original_probs= X_test,
                         platt_probs=platt_probs,
                         isotonic_probs=isotonic_probs,
                         fig_path=fig_path,
                         save_fig=False)



    os.makedirs(os.path.join(calibration_task_dir,'models'), exist_ok=True)

    if class_name is not None:
        model_path = os.path.join(calibration_task_dir, 'models',  f'{class_name}_platt_model.pkl')
        platt_model.save(model_path)
        model_path = os.path.join(calibration_task_dir, 'models',  f'{class_name}_isotonic_model.pkl')
        isotonic_model.save(model_path)

    elif class_number is not None:
        model_path=os.path.join(calibration_task_dir,'models',f'{class_number}_platt_model.pkl')
        platt_model.save(model_path)
        model_path = os.path.join(calibration_task_dir, 'models', f'{class_number}_isotonic_model.pkl')
        isotonic_model.save(model_path)
    else:
        model_path = os.path.join(calibration_task_dir, 'models', 'platt_model.pkl')
        platt_model.save(model_path)
        model_path = os.path.join(calibration_task_dir, 'models', 'isotonic_model.pkl')
        isotonic_model.save(model_path)

    print("\n== Training Done ==")
    return platt_model, isotonic_model


def apply_calibration(new_data_path,
                      pred_column,
                      class_name,
                      output_path,
                      platt_model_path,
                      isotonic_model_path,
                      soft_score_column=None,
                      ):

    print("\n== Applying the calibration model ==")

    output_dir=os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if new_data_path.endswith('.csv'):
        df_new = pd.read_csv(new_data_path)
    elif new_data_path.endswith('.xlsx'):
        df_new = pd.read_excel(new_data_path)
    else:
        raise ValueError(f"Only accept csv or excel")


    if pred_column not in df_new.columns:
        raise ValueError("Missing column: pred")

    pred_probs = df_new[pred_column].values
    print(f"Loaded {len(pred_probs)} samples")


    platt_model = PlattCalibration.load(platt_model_path)
    isotonic_model = IsotonicCalibration.load(isotonic_model_path)

    print("Applying model...")
    platt_probs = platt_model.predict_proba(pred_probs)
    isotonic_probs = isotonic_model.predict_proba(pred_probs)

    df_new['platt_calibrated'] = platt_probs
    df_new['isotonic_calibrated'] = isotonic_probs

    df_new.to_csv(output_path, index=False)
    print(f"Calibrated results are saved to: {output_path}")

    if soft_score_column is not None and soft_score_column in df_new.columns:
        true_labels = df_new[soft_score_column].values
        fig_path=output_path.split('.')[0]+'.png'
        evaluate_calibration(true_labels, pred_probs, platt_probs, isotonic_probs,class_name=class_name,
                             fig_path=fig_path)

    print("\n== Applying Done ==")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Probability Calibration Tools - Platt Scaling and Isotonic Regression')
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    train_parser = subparsers.add_parser('train', help='Train calibration model')
    train_parser.add_argument('--file_path',
                              help='Path to CSV file containing prediction probabilities and true labels')
    train_parser.add_argument('--pred_column', help='Column containing original prediction results')

    train_parser.add_argument('--class_number', help='Class index for calibration', type=int, default=None)
    train_parser.add_argument('--class_name', help='Class name', default=None)
    train_parser.add_argument('--calibration_task_dir', help='Working directory for calibration task')
    train_parser.add_argument(
        "--expert_columns",
        nargs="+",
        help="List of expert names",
        default=[],
    )
    train_parser.add_argument('--test_size', type=float, default=0.1, help='Test set proportion (default: 0.1)')
    train_parser.add_argument('--random_state', type=int, default=42, help='Random seed (default: 42)')

    apply_parser = subparsers.add_parser('apply', help='Apply calibration model to new data')
    apply_parser.add_argument('--new_data_path', help='Path to CSV file containing new prediction probabilities')
    apply_parser.add_argument('--output_path', help='Path to CSV file for calibrated results output')
    apply_parser.add_argument('--class_name', help='Class name')
    apply_parser.add_argument('--platt_model_path', default='platt_model.pkl', help='Path to Platt scaling model file')
    apply_parser.add_argument('--isotonic_model_path', default='isotonic_model.pkl',
                              help='Path to Isotonic regression calibration model file')
    apply_parser.add_argument('--pred_column', default='', help='Prediction results')
    apply_parser.add_argument('--soft_score_column', default='', help='Labels')


    args = parser.parse_args()


    if args.command == 'train':
        if args.expert_columns:
            expert_columns = args.expert_columns
        else:
            expert_columns = None

        train_calibration_models(file_path=args.file_path,
                                 pred_column=args.pred_column,
                                 calibration_task_dir=args.calibration_task_dir,
                                 class_number=args.class_number,
                                 class_name=args.class_name,
                                 expert_columns=expert_columns,
                                 test_size= args.test_size,
                                 random_state=args.random_state)


    elif args.command == 'apply':
        apply_calibration(new_data_path=args.new_data_path,
                          pred_column=args.pred_column,
                          output_path=args.output_path,
                          class_name=args.class_name,
                          platt_model_path=args.platt_model_path,
                          isotonic_model_path=args.isotonic_model_path,
                          soft_score_column=args.soft_score_column)

    else:
        parser.print_help()

