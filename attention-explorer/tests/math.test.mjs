import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { compute, column } from '../dist/math.js';
const data = JSON.parse(readFileSync(new URL('../dist/data.json', import.meta.url)));
const result = compute(data);
function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < 2e-6));
}
test('Projection, scores, weights and b1 match the actual PyTorch model', () => {
  for (const key of ['Q', 'K', 'V']) close(result[key].flat(), data[key].flat());
  close(result.raw, column(data.raw, 0));
  close(result.scaled, column(data.scaled, 0));
  close(result.weights, column(data.weights, 0));
  close(result.output, column(data.B, 0));
});
test('Normalization and value contribution are consistent', () => {
  assert.ok(Math.abs(result.weights.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.ok(result.weights.every(value => value > 0));
  result.output.forEach((value, row) => close([value], [result.V[row].reduce((sum, v, j) => sum + v * result.weights[j], 0)]));
});
test('Causal attention matches PyTorch for all query positions', () => {
  for (let query = 0; query < 3; query++) {
    const causal = compute(data, query, { causal: true });
    close(causal.weights, column(data.causal_weights, query));
    close(causal.output, column(data.causal_B, query));
    for (let key = query + 1; key < 3; key++) assert.equal(causal.weights[key], 0);
  }
});
