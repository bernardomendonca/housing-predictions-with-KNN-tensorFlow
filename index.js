const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
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
    dataColumns: ["lat", "long"],
    // what we want to predict:
    labelColumns: ["price"],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels);

// k valu of 10:
const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);

console.log("Guess ", result, testLabels[0][0]);
