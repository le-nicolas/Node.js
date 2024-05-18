// train the network to understand the XOR (exclusive OR) operation.

// Import Brain.js
const brain = require('brain.js');

// Create a new neural network Multi layer perceptron
const net = new brain.NeuralNetwork({
  hiddenLayers: [3] // Optional: Define hidden layers configuration
});

// Prepare the training data
const trainingData = [
  { input: [2.0, 3.0, -1.0], output: [1.0] },
  { input: [3.0, -1.0, 0.5], output: [-1.0] },
  { input: [0.5, 1.0, 1.0], output: [-1.0] },
  { input: [1.0, 1.0, -1.0], output: [1.0] }
];
//kind of nice that normalization is not needed.
// first of, XOR is a binary operation that takes two bits and returns 1 if exactly one of the bits is 1, otherwise it returns 0.
// So, for inputs 0 and 0, the output is 0. For inputs 0 and 1 or 1 and 0, the output is 1. For inputs 1 and 1, the output is 0.

// Train the network
net.train(trainingData, {
  iterations: 20000,   // The maximum number of iterations to train the network
  log: true,           // Print the error rate to the console
  logPeriod: 1000,     // The number of iterations between logging
  learningRate: 0.01   // Learning rate
});

// Test the network
const testCases = [
  { input: [2.0, 3.0, -1.0], expected: 1.0 },
  { input: [3.0, -1.0, 0.5], expected: -1.0 },
  { input: [0.5, 1.0, 1.0], expected: -1.0 },
  { input: [1.0, 1.0, -1.0], expected: 1.0 }
];

testCases.forEach((testCase) => {
  const output = net.run(testCase.input);
  console.log(`Input: ${testCase.input}, Predicted: ${output}, Expected: ${testCase.expected}`);
});

// output should be near the expected value
// error is quite frustrating but oh well.
