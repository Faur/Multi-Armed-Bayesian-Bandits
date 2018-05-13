data {
  int K;                // num arms
  int N;                // num trials
  int z[N];  // arm on trial n
  int y[N];  // reward on trial n
}
transformed data {
  int successes[K] = rep_array(0, K);
  int trials[K] = rep_array(0, K);
  for (n in 1:N) {
    trials[z[n]] += 1;
    successes[z[n]] += y[n];
  }
}
generated quantities {
  simplex[K] is_best;
  vector[K] theta;
  for (k in 1:K)
    theta[k] = beta_rng(1 + successes[k], 1 + trials[k] - successes[k]);
  {
    real best_prob = max(theta);
    for (k in 1:K)
      is_best[k] = (theta[k] >= best_prob);
    is_best /= sum(is_best);  // uniform for ties
  }
}
