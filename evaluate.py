
def eval_matrics(y_predict, y_real, filename, power=8, single=False):
    thresholds = [0.9] * 12
    
    def _matrics(positive, over_threshold, max_equal):
        negative = ~ positive
        below_threshold = ~ over_threshold
        
        true_positive = (positive & over_threshold & max_equal).astype(float).sum()
        true_negative = (negative & below_threshold).astype(float).sum()
        false_positive = positive.astype(float).sum() - true_positive
        false_negative = negative.astype(float).sum() - true_negative
    
        accuracy = (true_positive + true_negative) / len(y_real)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        
        return accuracy, precision, recall
    
    def _total_matrics():
        positive = y_real.sum(axis=1) > 0
        over_threshold = positive & False
        for c in range(12):
            over_threshold |= y_predict[:, c] > thresholds[c]
        max_equal = y_real.argmax(axis=1) == y_predict.argmax(axis=1)
        
        return _matrics(positive, over_threshold, max_equal)
    
    def _class_matrics(c):
        positive = y_real[:, c] > 0
        over_threshold = y_predict[:, c] > thresholds[c]
        max_equal = (y_real.argmax(axis=1) == y_predict.argmax(axis=1)) & (y_real.argmax(axis=1) == c)
        
        return _matrics(positive, over_threshold, max_equal)
    
    result_text = ''
    r_power = pow(10, power)
    
    if single == True:
        _, precision, _ = _total_matrics()
        if precision < 1.0:
            pass
        else:
            e = 0.1
            e_power = 1
            while True:
                thresholds = [t + e for t in thresholds]
                _, precision, _ = _total_matrics()
                if precision < 1.0:
                    thresholds = [t - e for t in thresholds]
                    if e_power == power:
                        for c in range(12):
                            thresholds[c] = round(thresholds[c] * r_power) / r_power
                            accuracy, precision, recall = _class_matrics(c)
                            result_text += "Label:=%02d" % c +\
                                " A=%.4f" % accuracy +\
                                " P=%.4f" % precision +\
                                " R=%.4f" % recall +\
                                " T=%.8f\n" % thresholds[c]
                        break
                    e *= 0.1
                    e_power += 1                    
    else:
        for c in range(12):
            _, precision, _ = _class_matrics(c)
            if precision < 1.0:
                continue
            e = 0.1
            e_power = 1
            while True:
                thresholds[c] += e
                _, precision, _ = _class_matrics(c)
                if precision < 1.0:
                    thresholds[c] -= e
                    if e_power == power:
                        thresholds[c] = round(thresholds[c] * r_power) / r_power
                        accuracy, precision, recall = _class_matrics(c)
                        result_text += "Label:=%02d" % c +\
                            " A=%.4f" % accuracy +\
                            " P=%.4f" % precision +\
                            " R=%.4f" % recall +\
                            " T=%.8f\n" % thresholds[c]
                        break
                    e *= 0.1
                    e_power += 1
    
    accuracy, precision, recall = _total_matrics()
    result_text += "Total:   " +\
        " A=%.4f" % accuracy +\
        " P=%.4f" % precision +\
        " R=%.4f\n" % recall    
    print("Result:  ",
          " A=", "{:.4f}".format(accuracy),
          " P=", "{:.4f}".format(precision),
          " R=", "{:.4f}".format(recall))
    
    with open(filename, 'w') as f:
        f.write(result_text)
