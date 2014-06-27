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

import org.apache.mahout.math.Centroid;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.WeightedVector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.stats.TDigest;

public abstract class AnomalyDetection {
	public abstract void buildModel(org.apache.mahout.math.Vector data);

	public abstract org.apache.mahout.math.Vector reconstructSignal(
			org.apache.mahout.math.Vector data);

	public org.apache.mahout.math.Matrix detectAnomalies(
			org.apache.mahout.math.Vector data,
			org.apache.mahout.math.Vector reconstructedSignal, double quantile) {

		// TODO: check length of data equal length of constructor
		org.apache.mahout.math.Matrix anomalies = new DenseMatrix(data.size(),
				2);
		
		// TODO: clarify whether this should be dynamic or not
		double compression = 100;
		// TODO: refactor code
		// double quantile = 90 / 100;

		// subtracting vectors
		data.assign(reconstructedSignal, Functions.MINUS);

		// t-digest application
		org.apache.mahout.math.stats.TDigest digest = new TDigest(compression);
		for (org.apache.mahout.math.Vector.Element element : data.all()) {
			digest.add(element.get());
		}

		double upperThreshold = digest.quantile(quantile);
		double lowerThreshold = digest.quantile(1 - quantile);

		// TODO: performance optimization, refactor code
		for (int i = 0, j = 0; i < data.size(); i++) {
			double element = data.getQuick(i);
			if (element > upperThreshold || element < lowerThreshold) {
				// insert value and index of value to return matrix
				anomalies.viewColumn(0).setQuick(j, element);
				anomalies.viewColumn(1).setQuick(j, i);
				// increment matrix row
				j++;
			}
		}
		return anomalies;
	}
}
