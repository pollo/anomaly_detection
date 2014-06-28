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
import java.util.ArrayList;
import java.util.Formatter;
import java.util.List;

import java.io.*;

/**
 * Example of anomaly detection using AnomalyDetection. The code is taken from
 * Ted Dunning's EKG anomaly detection example
 * (https://github.com/tdunning/anomaly-detection) and adapted to the
 * AnomalyDetection pattern.
 * 
 * According EKG Data can be found at
 * physionet.org/physiobank/database/#ecg-databases.
 * 
 * Read EKG data, extract windows and apply k-means clustering. Afterwards,
 * build a reconstructed signal to identify out the error.
 */
public class EKGAnomalyDetection extends AnomalyDetection {
	// Window Size for EKG Data Example
	private final int WINDOW = 32;
	// distance between starting points of two adjacent windows
	private int STEP = 2;
	// number of constructed windows used for clustering
	private int SAMPLES = 200000;
	// the fraction of returned anomalies
	private final double ANOMALY_FRACTION = 10.0 / 100;
	// according to Ted Dunning's description, 100 roughly represents a
	// compression ratio for small to mid-size data sets
	private final double COMPRESSION = 100;

	private Vector window;
	private double t0;
	private double t1;

	UpdatableSearcher clustering;

	/**
	 * Read EKG trace and extract scaled data points.
	 * 
	 * @param in
	 *            Input Data File
	 * @param scale
	 *            Scaling Factor for Data Points
	 * @return Vector of Data Points
	 * @throws IOException
	 */
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
	/**
	 * Build a model based on extracted EKG data points by using k-means clustering.
	 * 
	 * @param trace EKG Data Points.
	 */
	public void buildModel(Vector trace) {
		// initialize variable for timing output
		this.t0 = System.nanoTime() / 1e9;

		// list of windowed data
		List<WeightedVector> r = Lists.newArrayList();

		// create windows according to SAMPLES and STEP
		for (int i = 0; i < SAMPLES; i++) {
			int offset = i * STEP;
			WeightedVector row = new WeightedVector(new DenseVector(WINDOW), 1,
					i);
			row.assign(trace.viewPart(offset, WINDOW));
			row.assign(this.window, Functions.MULT);
			// normalizing the data
			row.assign(Functions.mult(1 / row.norm(2)));
			r.add(row);
		}
		// time for windowing data
		this.t1 = System.nanoTime() / 1e9;
		System.out.printf("Windowed data in %.2f s\n", this.t1 - this.t0);

		// clustering the data by applying k-means
		this.t0 = System.nanoTime() / 1e9;
		BallKMeans km = new BallKMeans(new BruteSearch(
				new EuclideanDistanceMeasure()), 400, 10);
		this.clustering = km.cluster(r);
		this.t1 = System.nanoTime() / 1e9;
		System.out.printf("Clustered in %.2f s\n", this.t1 - this.t0);

		// Output clustering results. One line per cluster
		// centroid, each with WINDOW values
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
	/**
	 * 
	 * Reconstruct the Signal by matching original windows against the nearest cluster centroid as provided by k-means output.
	 * 
	 * @param trace Data EKG Data Points.
	 * @return The reconstructed Signal.
	 */
	public Vector reconstructSignal(Vector trace) {
		// Reconstruct the Signal. Each window can be looked at independently
		// due to the windowing.
		// This works because the window before and after the current one will
		// independently approximate the portion of the signal
		// left over after subtracting this window.

		Vector reconstructedSignal = new DenseVector(trace.size());

		try (Formatter out = new Formatter("trace.tsv")) {
			// First Column = original, Second Column = reconstructed
			Matrix rx = new DenseMatrix(WINDOW / 2, 2);
			// Memorize window order
			Vector previous = new DenseVector(WINDOW);
			Vector current = new DenseVector(WINDOW);

			for (int i = 0; i + WINDOW < trace.size(); i += WINDOW / 2) {
				// copy chunk of data to temporary window storage and multiply
				// by window
				WeightedVector row = new WeightedVector(
						new DenseVector(WINDOW), 1, i);
				row.assign(trace.viewPart(i, WINDOW));
				// applying the window to the original data
				row.assign(this.window, Functions.MULT);

				// scale data
				double scale = row.norm(2);
				row.assign(Functions.mult(1 / scale));

				// find the closest centroid according to scaled data
				WeightedThing<Vector> cluster = this.clustering.search(row, 1)
						.get(0);
				current.assign(cluster.getValue());
				// scale data back to original
				current.assign(Functions.mult(scale));

				// Produce results of half a window at a time. The reconstructed
				// Signal is the sum of the 2nd half of the previous window and
				// the 1st half of the current window
				rx.viewColumn(0).assign(trace.viewPart(i, WINDOW / 2));
				rx.viewColumn(1).assign(
						previous.viewPart(WINDOW / 2, WINDOW / 2));
				rx.viewColumn(1).assign(current.viewPart(0, WINDOW / 2),
						Functions.PLUS);
				previous.assign(current);

				for (int j = 0; j < WINDOW / 2; j++) {
					out.format("%.3f\t%.3f\t%d\n", rx.get(j, 0), rx.get(j, 1),
							((WeightedVector) cluster.getValue()).getIndex());
					reconstructedSignal.setQuick(i + j, rx.get(j, 1));
				}
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		// Time for Signal reconstruction
		this.t1 = System.nanoTime() / 1e9;
		System.out.printf("Output in %.2f s\n", this.t1 - this.t0);

		return reconstructedSignal;
	}

	public void run() throws IOException {

		// read the EKG data
		URL x = Resources.getResource("a02.dat");
		this.t0 = System.nanoTime() / 1e9;
		Vector trace = this.read16b(new File(x.getPath()), 1.0 / 200);
		this.t1 = System.nanoTime() / 1e9;
		System.out
				.printf("Read test data from %s in %.2f s\n", x, this.t1 - t0);

		// set up the window vector
		this.window = new DenseVector(WINDOW);
		for (int i = 0; i < WINDOW; i++) {
			double w = Math.sin(Math.PI * i / (WINDOW - 1.0));
			this.window.set(i, w * w);
		}

		this.buildModel(trace);

		ArrayList<Anomaly> anomalies = null;
		try {
			anomalies = this.detectAnomalies(trace, ANOMALY_FRACTION,
					COMPRESSION);
		} catch (IllegalArgumentException e) {
			System.err.println("Error occured while detecting anomalies");
			System.err.println(e);
			System.exit(1);
		}

		// output anomalies
		try (Formatter out = new Formatter("anomalies.tsv")) {
			for (Anomaly a : anomalies) {
				out.format("%.3f\t%.3f\t%d\n", a.getData(), a.getError(),
						a.getIndex());
			}
		}
	}

	public static void main(String[] args) throws IOException {
		new EKGAnomalyDetection().run();
	}
}
