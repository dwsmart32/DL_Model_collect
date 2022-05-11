from sklearn.metrics import classification_report, accuracy_score

def print_reporting(y_test, prediction):
    acc = accuracy_score(y_test, prediction)
    report = classification_report(y_test, prediction)
    print(f'accuracy: {str(round(acc * 100, 2))}%')
    print(report)