# Time series anomaly detection in Mahout

## Introduction

This repository contains a new class for time series anomaly detection in Mahout and a corresponding example based on Ted Dunning's previous work on EKG data.

You can find the new class under  ```src/main/java/org/apache/mahout/anomalydetection/TimeSeriesAnomalyDetection.java```.

The TimeSeriesAnomalyDetection class embeds the t-digest algorithm in order to spot anomalies and guides the user through the process of anomaly detection.

The EKAnomalyDetection class implements a time series anomaly detection scenario by applying the newly introduced TimeSeriesAnomalyDetection class.

The example is provided under ```src/main/java/org/apache/mahout/anomalydetection/EKGAnomalyDetection.java```.


## How to run the example

In order to run the example:

1. Assure maven is installed in your system ([https://maven.apache.org/](https://maven.apache.org/))
2. Execute: ```mvn clean install```
3. Execute the following command:

	```mvn -q exec:java -Dexec.mainClass=org.apache.mahout.anomalydetection.EKGAnomalyDetection```

In order to test it run:  ```mvn test```

## References
For further information:

### Anomaly detection

* [Practical Machine Learning: A New Look At Anomaly Detection by Ted Dunning and Ellen Friedman](http://info.mapr.com/resources_ebook_anewlook_anomalydetection.html?cid=blog)
* [A talk about anomaly detection](http://berlinbuzzwords.de/session/deep-learning-high-performance-time-series-databases)
* [Related to this example on anomaly detection on EKG data](https://github.com/tdunning/anomaly-detection)

### t-digest algorithm

* [The original implementation of and documentation for t-digest](https://github.com/tdunning/t-digest)


