/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the License); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.anomalydetection;

import com.google.common.collect.Lists;
import org.apache.mahout.math.*;
import com.tdunning.math.stats.TDigest;
import com.tdunning.math.stats.TreeDigest;

import java.util.List;

/**
 * Abstract base class for anomaly detection. The process is modeled in three steps: - building a
 * model for the data - using the model to build a representation of the data - apply t-digest to
 * detect when the representation differs from the data
 * <p/>
 * The class should be extended implementing the methods buildModel and reconstructSignal. The
 * method detectAnomalies can then be used to retrieve the anomalies found with the application of
 * the t-digest algorithm.
 * <p/>
 * The concept is taken from the book Practical Machine Learning:A New Look At Anomaly Detection by
 * Ted Dunning and Ellen Friedman
 */
public abstract class AnomalyDetection {
  /**
   * Build a model for the data. The user dependent implementation should store the model into the
   * state of the class. The model will then be used into the reconstructSignal method to build a
   * representation of the data.
   *
   * @param data Data used to build (or fit) the model
   */
  public abstract void buildModel(org.apache.mahout.math.Vector data);

  /**
   * Builds and returns the closest representation (reconstructed signal) of the data which the
   * model constructed in buildModel can provide.
   *
   * @param data Data for reconstruction
   * @return The reconstructed signal
   */
  public abstract org.apache.mahout.math.Vector reconstructSignal(
    org.apache.mahout.math.Vector data);

  /**
   * Detects and returns the anomalies.
   * <p/>
   * First a reconstruction of the data is obtained using the reconstructSignal method. Then the
   * t-digest algorithm is used to detect when the reconstructed signal differs too much from the
   * actual data (depending on the quantile parameter).
   *
   * @param data            Data used for anomaly detection
   * @param anomalyFraction Fraction of data point reported as anomalies
   * @param compression     Parameter used from the t-digest algorithm to set the data compression
   * @return List of Anomaly, each Anomaly identifies an anomalous data point reporting the data
   * value, the difference from the reconstructed signal (error) and the position in the sequence.
   * @throws IllegalArgumentException Thrown when size of the reconstructed signal differs from the
   *                                  size of the data
   */
  public List<Anomaly> detectAnomalies(
    Vector data,
    double anomalyFraction,
    double compression) throws IllegalArgumentException {
    // reconstruct signal starting from data
    Vector reconstructedSignal = this.reconstructSignal(data);

    // calculate error between reconstructed signal and actual data
    Vector delta;
    try {
      delta = data.minus(reconstructedSignal);
    } catch (CardinalityException e) {
      throw new IllegalArgumentException("The size of reconstructedSignal differs from the data size");
    }

    // run t-digest to compute threshold corresponding to the quantile
    TDigest digest = new TreeDigest(compression);
    for (org.apache.mahout.math.Vector.Element element : delta.all()) {
      digest.add(Math.abs(element.get()));
    }
    double threshold = digest.quantile(1 - anomalyFraction);

    // output anomalies (error above threshold)
    List<Anomaly> anomalies = Lists.newArrayList();
    for (int i = 0; i < data.size(); i++) {
      double element = delta.getQuick(i);
      if (Math.abs(element) > threshold) {
        // insert data, error and index into return Map
        anomalies.add(new Anomaly(data.getQuick(i),
          element,
          i));
      }
    }

    return anomalies;
  }
}

class Anomaly {
  private double data;
  private double error;
  private int index;

  public Anomaly(double data,
                 double error,
                 int index) {
    this.data = data;
    this.error = error;
    this.index = index;
  }

  public double getData() {
    return data;
  }

  public double getError() {
    return error;
  }

  public int getIndex() {
    return index;
  }
}
