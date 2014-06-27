/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.anomalydetection;


import com.google.common.collect.Lists;
import com.google.common.io.Resources;
import org.apache.mahout.clustering.streaming.cluster.BallKMeans;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.*;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.neighborhood.BruteSearch;
import org.apache.mahout.math.neighborhood.UpdatableSearcher;
import org.apache.mahout.math.random.WeightedThing;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Formatter;
import java.util.List;

import java.io.*;

/**
 * Read a bunch of EKG data, chop out windows and cluster the windows. Then reconstruct the signal and
 * figure out the error.
 */
public class EKGAnomalyDetection extends AnomalyDetection {
  private final int WINDOW = 32;
  private int STEP = 2;
  private int SAMPLES = 200000;

  private Vector window;
  private double t0;
  private double t1;


  UpdatableSearcher clustering;

  public DenseVector read16b(File in, double scale) throws IOException {
    DataInputStream input = new DataInputStream(new FileInputStream(in));

    int rows = (int) (in.length() / 2);

    DenseVector data = new DenseVector(rows);
    for (int i = 0; i < rows; i++) {
      data.setQuick(i, input.readShort() * scale);
    }
    return data;
  }

  @Override
  public void buildModel(Vector data) {
    Vector trace = data;

    // window and normalize the data
    this.t0 = System.nanoTime() / 1e9;
    List<WeightedVector> r = Lists.newArrayList();
    for (int i = 0; i < SAMPLES; i++) {
      int offset = i * STEP;
      WeightedVector row = new WeightedVector(new DenseVector(WINDOW), 1, i);
      row.assign(trace.viewPart(offset, WINDOW));
      row.assign(this.window, Functions.MULT);
      row.assign(Functions.mult(1 / row.norm(2)));
      r.add(row);
    }
    this.t1 = System.nanoTime() / 1e9;
    System.out.printf("Windowed data in %.2f s\n", this.t1 - this.t0);

    // now cluster the data
    this.t0 = System.nanoTime() / 1e9;
    BallKMeans km = new BallKMeans(new BruteSearch(new EuclideanDistanceMeasure()), 400, 10);
    this.clustering = km.cluster(r);
    this.t1 = System.nanoTime() / 1e9;
    System.out.printf("Clustered in %.2f s\n", this.t1 - this.t0);


    // and now dump the clustering results. This prints one line per cluster centroids, each with WINDOW values
    this.t0 = System.nanoTime() / 1e9;
    try (Formatter out = new Formatter("dict.tsv")) {
      for (Vector v : this.clustering) {
        String separator = "";
        for (Vector.Element element : v.all()) {
          out.format("%s%.3f", separator, element.get());
          separator = "\t";
        }
        out.format("\n");
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  @Override
  public Vector reconstructSignal(Vector data) {
    Vector trace = data;
    // and the final results. This output includes the original signal, the reconstructed signal and the error
    // in this algorithm, each window is windowed and we simply look for the nearest cluster signal for that
    // window. The trick is that we can look at each window independently because of the windowing. This works
    // because the window before and after the current one will independently approximate the portion of the signal
    // left over after subtracting this window.
    Vector reconstructedSignal = new DenseVector(trace.size());

    try (Formatter out = new Formatter("trace.tsv")) {
      Matrix rx = new DenseMatrix(WINDOW / 2, 2);
      Vector previous = new DenseVector(WINDOW);
      Vector current = new DenseVector(WINDOW);
      for (int i = 0; i + WINDOW < trace.size(); i += WINDOW / 2) {
        // copy chunk of data to temporary window storage and multiply by window
        WeightedVector row = new WeightedVector(new DenseVector(WINDOW), 1, i);
        row.assign(trace.viewPart(i, WINDOW));
        row.assign(this.window, Functions.MULT);

        // scale and find nearest dictionary entry
        double scale = row.norm(2);
        row.assign(Functions.mult(1 / scale));

        WeightedThing<Vector> cluster = this.clustering.search(row, 1).get(0);
        current.assign(cluster.getValue());
        current.assign(Functions.mult(scale));

        // we produce results half a window at a time. First column is original signal, second is reconstruction
        rx.viewColumn(0).assign(trace.viewPart(i, WINDOW / 2));
        rx.viewColumn(1).assign(previous.viewPart(WINDOW / 2, WINDOW / 2));
        rx.viewColumn(1).assign(current.viewPart(0, WINDOW / 2), Functions.PLUS);
        previous.assign(current);

        for (int j = 0; j < WINDOW / 2; j++) {
          out.format("%.3f\t%.3f\t%d\n", rx.get(j, 0), rx.get(j, 1), ((WeightedVector) cluster.getValue()).getIndex());
          reconstructedSignal.setQuick(i + j, rx.get(j, 1));
        }
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
    this.t1 = System.nanoTime() / 1e9;
    System.out.printf("Output in %.2f s\n", this.t1 - this.t0);

    return reconstructedSignal;
  }

  public void run()  throws IOException  {
    // read the data
    URL x = Resources.getResource("a02.dat");
    this.t0 = System.nanoTime() / 1e9;
    Vector trace = this.read16b(new File(x.getPath()), 1.0 / 200);
    this.t1 = System.nanoTime() / 1e9;
    System.out.printf("Read test data from %s in %.2f s\n", x, this.t1 - t0);

    // set up the window vector
    this.window = new DenseVector(WINDOW);
    for (int i = 0; i < WINDOW; i++) {
      double w = Math.sin(Math.PI * i / (WINDOW - 1.0));
      this.window.set(i, w * w);
    }

    this.buildModel(trace);

    Vector reconstructedSignal = this.reconstructSignal(trace);

    double quantile = 90.0/100;
    Matrix anomalies = this.detectAnomalies(trace, reconstructedSignal, quantile);

    //output anomalies
    try (Formatter out = new Formatter("anomalies.tsv")) {
      for (int i = 0; i < anomalies.numRows(); i++) {
        out.format("%.3f\t%.3f\t%d\n", anomalies.get(i, 0), anomalies.get(i, 1), anomalies.get(i, 2));
      }
    }
  }

  public static void main(String[] args) throws IOException {
    new EKGAnomalyDetection().run();
  }
}

