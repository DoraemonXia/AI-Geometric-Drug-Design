import numpy as np

def get_true_positive_rate_under_thresholds(probs, label):
    """
    return a list which included the true posite rate under different threshold, 
    ranges from 0 to 1, step by 0.01.
    """
    probs = np.array(probs)
    label = np.array(label)

    thresholds = np.linspace(0, 1, 101)  # 在0到1之间平均取100个点作为阈值
    positive_probs = []  # 存储大于阈值的正样本概率

    for threshold in thresholds:
        # 预测为正的样本
        predicted_positive_indices = np.where(probs > threshold)[0]

        true_positive_count = np.sum(label[predicted_positive_indices] == 1)

        # 预测为正的样本数量
        predicted_positive_count = len(predicted_positive_indices)

        # 计算真正为正样本的比率
        true_positive_rate = true_positive_count / predicted_positive_count if predicted_positive_count > 0 else 0
        
        positive_probs.append(true_positive_rate)

    return  positive_probs

"""
how to use
just get_true_positive_rate_under_thresholds( probs, label )
"""