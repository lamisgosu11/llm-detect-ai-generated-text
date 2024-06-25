from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def evaluate(output, target):
    # get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()

    # Move tensors to CPU for numpy compatibility
    pred = pred.cpu()
    target = target.cpu()

    acc = accuracy_score(target, pred)
    prec = precision_score(target, pred, average='macro')
    rec = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')

    return acc, prec, rec, f1


def print_confusion_matrix(output, target):
    output = output.cpu()
    target = target.cpu()

    pred = output.argmax(dim=1)
    print(confusion_matrix(target, pred))
    print(classification_report(target, pred))