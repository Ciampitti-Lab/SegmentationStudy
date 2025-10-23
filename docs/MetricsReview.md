# Literature Review on Segmentation Metrics

P.S.: Copied from: [Geeks For Geeks](https://www.geeksforgeeks.org/computer-vision/what-are-different-evaluation-metrics-used-to-evaluate-image-segmentation-models/)


- [Literature Review on Segmentation Metrics](#literature-review-on-segmentation-metrics)
  - [Intersection over Union (IoU)](#intersection-over-union-iou)
  - [Dice Coefficient](#dice-coefficient)
  - [Precision and Recall](#precision-and-recall)
  - [F1 Score](#f1-score)
  - [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
  - [Hausdorff Distance](#hausdorff-distance)
  - [Pixel Accuracy](#pixel-accuracy)


## Intersection over Union (IoU)

IoU, also known as the Jaccard Index, is a common metric for evaluating image segmentation models. It measures the overlap between the predicted segmentation and the ground truth. The IoU value is calculated by dividing the area of overlap between the predicted and ground truth segments by the area of their union. This metric provides a clear indication of how well the predicted segmentation matches the actual segmentation.

Formula:

IoU = Area of Overlap / Area of Union

IoU ranges from 0 to 1, where 1 indicates a perfect match and 0 indicates no overlap. This metric is particularly useful in applications where precise boundary delineation is critical, such as medical imaging and autonomous driving. IoU is favored because it balances both false positives and false negatives, giving a comprehensive view of model performance.

## Dice Coefficient

The Dice Coefficient, also known as the Sørensen–Dice index, measures the similarity between two sets of data. It is particularly effective in assessing the accuracy of image segmentation models. The Dice Coefficient is calculated by doubling the area of overlap between the predicted segmentation and the ground truth, then dividing by the total number of pixels in both segmentations.

Formula:

Dice Coefficient = (2× Area of Overlap) / Total Number of Pixels in Both Segmentations

A Dice Coefficient of 1 signifies perfect overlap, while 0 indicates no overlap. This metric is especially useful in scenarios where the focus is on correctly identifying the segmented areas without much concern for the non-segmented areas. It is widely used in medical imaging for tasks such as tumor detection and organ segmentation, where the precise identification of regions is crucial.

## Precision and Recall

Precision and Recall are metrics borrowed from classification tasks but are also valuable in segmentation. Precision measures the proportion of true positive pixels among all pixels classified as positive by the model. Recall, on the other hand, measures the proportion of true positive pixels that were correctly identified out of all actual positive pixels.

Formulas:

Precision = True Positives / (True Positives + False Positives)

Recall = True Positives / (True Positives + False Negatives)
 
High precision indicates that the model has a low false positive rate, while high recall indicates a low false negative rate. These metrics are often used together with the F1 Score, which provides a harmonic mean of precision and recall. In image segmentation, precision and recall help balance the trade-off between over-segmentation and under-segmentation, ensuring that the model accurately identifies relevant regions without including too much irrelevant information.

## F1 Score
The F1 Score is a metric that combines precision and recall into a single number. It is the harmonic mean of precision and recall and provides a balanced measure of the model’s performance, especially when dealing with imbalanced classes.

Formula:

F1 Score = 2 × ( (Precision × Recall ) / (Precision + Recall))

The F1 Score ranges from 0 to 1, where 1 indicates perfect precision and recall, and 0 indicates the worst performance. It is particularly useful when you want to balance the trade-off between false positives and false negatives.

## Mean Absolute Error (MAE)

Mean Absolute Error (MAE) measures the average magnitude of errors between predicted and actual values. Unlike IoU and Dice, which focus on the overlap, MAE provides an absolute measure of how far off the predictions are from the ground truth.

Formula:

MAE = (1/n) x Σ ∣Predicted − Actual∣

MAE is useful in applications where you need to know the exact difference between the predicted and actual segmentation, providing a straightforward interpretation of error magnitude.

## Hausdorff Distance

Hausdorff Distance measures the greatest distance from a point in one set to the closest point in another set. In the context of image segmentation, it evaluates the worst-case discrepancy between the predicted and ground truth boundaries.

Formula:

dH(A,B) = max(h(A,B),h(B,A))

Hausdorff Distance is particularly useful for applications requiring strict boundary adherence, as it highlights the most significant errors in segmentation.

## Pixel Accuracy
Pixel Accuracy calculates the ratio of correctly predicted pixels to the total number of pixels. It is a straightforward metric that provides a general sense of the model’s overall performance.

Formula:

Pixel Accuracy = Number of Correct Pixels / Total Number of Pixels

While simple, pixel accuracy can be misleading in cases of imbalanced datasets, where the background class might dominate the metric