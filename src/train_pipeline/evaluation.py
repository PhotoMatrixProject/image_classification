import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
# import tensorflow as tf


def display_confusion_matrix(test_labels:np.ndarray, predictions:np.ndarray, class_names:list[str]):
    '''Saves and displays confusion matrix.'''
    cm = confusion_matrix(test_labels, predictions, labels = class_names, normalize='true')
    # print(cm[0,0], cm[1,1], cm[2,2], cm[3,3], cm[4,4], cm[5,5], cm[6,6], cm[7,7])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_names)
    disp.plot(xticks_rotation='vertical')
    # plt.tight_layout()
    # plt.figure(figsize=(100, 100))
    plt.savefig("cm.png")
    plt.show()
    return



def analyze_confusion_matrix(test_labels:np.ndarray, predictions:np.ndarray, class_names:list[str]):
    '''writes the accuracy computed on test_labels and predictions to a txt file.'''
    with open("accuracy_results_vgg19_step3_fft_lbp.txt", "w") as f:

        cm = confusion_matrix(test_labels, predictions, labels = class_names, normalize='true')
        for i in range(len(class_names)):
            if cm[i, i] < 0.9:
                print(class_names[i], cm[i, i])
                f.write(f"{class_names[i]}: {cm[i, i]}\n")
                dist = cm[i, :]
                name_to_err = {class_names[j]: dist[j] for j in range(len(class_names)) if j != i}
                name_to_err = sorted(name_to_err.items(), key=lambda x: x[1], reverse=True)
                for name, err in name_to_err:
                    if err > 0.0:
                        print(f"  {name}: {err}")
                        f.write(f"  {name}: {err}\n")
                print()
                f.write("\n")
    return

def display_confusion_matrix_with_one_dec(test_labels:list[str], predictions:list[str]):
    '''Saves and displayes a confusion matrix where every class starting with dec_ is put in one class dec.'''
    class_names_agg_dec = ['2D', 'Arch_plans', 'Architecture', 'Exhibition', 'NOT_IMG', 'sculpture', 'WITHOUT_LABEL_PHOTO', 'dec']

    test_labels = ['dec' if l.startswith('dec_') else l for l in test_labels]
    predictions = ['dec' if p.startswith('dec_') else p for p in predictions]

    display_confusion_matrix(test_labels, predictions, class_names=class_names_agg_dec)

def measure_accuracy_score(test_labels:np.ndarray, predictions:np.ndarray):
    '''Prints the accuracy computed on test_labels and predictions.'''
    score = accuracy_score(test_labels, predictions)
    print(score)
    return

def convert_nums_to_labels(arr:np.ndarray, class_names:list[str])->list[str]:
    '''Converts numbers to labels according to the index of the class in the class list.'''
    arr_labels = [class_names[int(x)] for x in arr]
    return arr_labels

