package org.apache.mahout.anomalydetection;

import org.junit.Test;
import java.io.*;
import java.util.Scanner;

import static org.junit.Assert.*;

public class EKGAnomalyDetectionTest {

  @Test
  public void testNumberFoundAnomalies() throws Exception {
    new EKGAnomalyDetection().run();
    Scanner traceFile = new Scanner(new File("trace.tsv"));
    int points_number = 0;
    while (traceFile.hasNextLine()) {
      points_number += 1;
      traceFile.nextLine();
    }
    Scanner anomaliesFile = new Scanner(new File("anomalies.tsv"));
    int anomalies_number = 0;
    while (anomaliesFile.hasNextLine()) {
      anomalies_number += 1;
      anomaliesFile.nextLine();
    }

    double threshold = 0.001;
    assertTrue("The number of found anomalies is not the fraction expected",
              Math.abs((double)anomalies_number/points_number - EKGAnomalyDetection.ANOMALY_FRACTION) < threshold);
  }
}
