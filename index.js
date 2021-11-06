const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  // standardizing the features by doing:
  // (Value - Average) / StandardDeviation
  // StandardDeviation is the square root of variance
  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      // standardization step:
      .sub(mean)
      .div(variance.pow(0.5))
      // distance calculation:
      .sub(scaledPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      // running the KNN algorithm:
      .expandDims(1)
      .concat(labels, 1)
      .unstack()
      .sort((a, b) => (a.arraySync()[0] > b.arraySync()[0] ? 1 : -1))
      .slice(0, k)
      .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k
  );
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    //shuffling our dataset, so when we take out the test and train data we won't be biased:
    shuffle: true,
    //setting up how many records in our Test set:
    splitTest: 10,
    //which columns we'll analyse first:
    dataColumns: [
      "lat",
      "long",
      "sqft_lot",
      "sqft_living",
      "bedrooms",
      "bathrooms",
    ],
    // what we want to predict:
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testpoint, i) => {
  // k value of 10:
  const result = knn(features, labels, tf.tensor(testpoint), 10);
  const err = (100 * (testLabels[i][0] - result)) / testLabels[i][0];

  console.log("Guess ", result, testLabels[i][0]);
  console.log("error", err, "%");
});
