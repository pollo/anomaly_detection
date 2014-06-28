# Anomaly detection in Mahout

This repository contains a new class for anomaly detection in Mahout and a corresponding example based on Ted Dunning's previous work on EKG data.

You can find the new class under  ```src/main/java/org/apache/mahout/anomalydetection/AnomalyDetection.java```.

The AnomalyDetection class embeds the t-digest algorithm in order to spot anomalies and guides the user through the process of anomaly detection.

The EKAnomalyDetection class implements an anomaly detection scenario by applying the newly introduced AnomalyDetection class.

The example is provided under ```src/main/java/org/apache/mahout/anomalydetection/EKGAnomalyDetection.java```.

## References
For further information:

### Anomaly detection

* [Practical Machine Learning: A New Look At Anomaly Detection by Ted Dunning and Ellen Friedman](http://info.mapr.com/resources_ebook_anewlook_anomalydetection.html?cid=blog)
* [Ted Dunning's talk about anomaly detection](http://berlinbuzzwords.de/session/deep-learning-high-performance-time-series-databases)
* [Ted Dunning's example on anomaly detection on EKG data](https://github.com/tdunning/anomaly-detection)

### t-digest algorithm

* [Ted Dunning's implementation of t-digest](https://github.com/tdunning/t-digest)


