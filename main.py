import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import seaborn as sns
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling1D, Conv1D, Input
from sklearn.utils.class_weight import compute_class_weight
import time
from data_loader import get_dataset
from tensorflow.keras.optimizers import Adam

def trainer(args, X_train, y_train, optimizer, class_weights_dict):
    model = create_model(args, X_train.shape[1], optimizer)
    model_name = args.model
    if(model_name=='cnn'):
        batch_size = 64
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    else:
        batch_size = 128
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights_dict)

    output_weight_path = f"./weight/{model_name}_{round(time.time()*1000)}.keras"
    model.save(output_weight_path)
    print("Save model. Finish training...!!")
    return output_weight_path

def evaluation(args, weight_path, X_test, y_test):
    model = load_model(weight_path)
    if args.model == 'cnn':
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_proba = model.predict(X_test).flatten()

    # if args.model == "cnn":        

    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    
    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUROC: {auroc:.3f}")
    print(f"AUPRC: {auprc:.3f}")
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['Susceptible (S)', 'Resistant (R)',])
    print(report)
    print("Finish evaluation!")
    name = os.path.splitext(os.path.basename(weight_path))[0]
    print("Start visualization!")
    draw_roc(args,name, model, X_test, y_test)
    draw_prc(args,name, model, X_test, y_test)
    print("Finish")

def draw_roc(args, name, model, X_test, y_test):
    antimicrobial_name = args.medicine
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    font_size = 14
    plt.figure(figsize=(10, 8), dpi=100)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=font_size)
    plt.ylabel('True Positive Rate', fontsize=font_size)
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {antimicrobial_name}', fontsize=font_size)
    plt.legend(loc="lower right", fontsize=font_size)
    # plt.show()
    plt.savefig(f'./visualization/{name}_ROC.png')

def draw_prc(args, name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = y_pred.flatten()
    # Calculate the Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # Calculate the AUPRC score (average precision score)
    auprc = average_precision_score(y_test, y_pred_proba)
    # Plot the Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', label=f'PR Curve (AUPRC = {auprc:.3f})')
    # Add a diagonal line representing the baseline (where Precision = Recall)
    plt.plot([0, 1], [1, 0], linestyle='--', color='grey', label='Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'./visualization/{name}_PRC.png')

def create_model(args, input_shape, optimizer):
    if(args.model=='dnn'):
        model = Sequential([
            Dense(200, activation='relu', input_shape=(input_shape,)),
            Dropout(0.5),
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(50, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Sigmoid for binary classification
        ])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model
    else:
        model = Sequential([
            Input(shape= (input_shape,1)),
            Conv1D(48, 3, activation='relu'),
            MaxPooling1D(pool_size=2, strides=1, padding="valid"),
            Dropout(0.5),
            Conv1D(96, 3, activation='relu'),
            MaxPooling1D(pool_size=2, strides=1, padding="valid"),
            Flatten(),
            Dropout(0.5),
            Dense(200, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

def main():
    parser = argparse.ArgumentParser(description='Classification...')
    parser.add_argument("--medicine", type=str, default='Amoxicillin-Clavulanic acid',help="medicine you want to classify")
    parser.add_argument("--target_dataset", type=str, default="./data", help="Folder where data located")
    parser.add_argument("--model", type=str, default="cnn", help="Deeplearning model architecture")
    parser.add_argument("--lr", type=int, default=0.0001, help="Learning rate")
    args = parser.parse_args()
    #load data
    X_train, X_test, y_train, y_test = get_dataset(args)

    #optimizer
    optimizer = Adam(learning_rate=args.lr)

    #calculate weight
    classes = np.unique(y_train)  # Unique classes in the dataset
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)

    #convert to a dictionary
    class_weights_dict = dict(zip(classes, class_weights))

    weight = trainer(args,  X_train, y_train, optimizer, class_weights_dict)
    #evaluation
    evaluation(args, weight, X_test, y_test)

if __name__=="__main__":
    main()