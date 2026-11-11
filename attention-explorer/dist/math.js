export function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}
export function column(matrix, index) {
  return matrix.map(row => row[index]);
}
export function softmax(scores) {
  const max = Math.max(...scores);
  const exponentials = scores.map(score => Math.exp(score - max));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return { max, exponentials, total, weights: exponentials.map(value => value / total) };
}
export function compute(data, queryIndex = 0, { causal = false } = {}) {
  const project = weights => weights.map(row => data.X[0].map((_, index) => dot(row, column(data.X, index))));
  const Q = project(data.WQ);
  const K = project(data.WK);
  const V = project(data.WV);
  const query = column(Q, queryIndex);
  const raw = K[0].map((_, index) => dot(query, column(K, index)));
  const scaled = raw.map(value => value / Math.sqrt(query.length));
  const masked = scaled.map((value, keyIndex) => causal && keyIndex > queryIndex ? -Infinity : value);
  const normalized = softmax(masked);
  const output = V.map(row => dot(row, normalized.weights));
  return { Q, K, V, raw, scaled, ...normalized, output };
}
