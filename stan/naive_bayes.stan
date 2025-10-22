data {
  int<lower=1> J;             // number of observations
  int<lower=1> C;
  int<lower=1> num_feat;      // number of numeric (PCA) features
  int<lower=0> bi_feat;       // number of binary features
  int<lower=1, upper=num_feat> order_feat; // which numeric feature to enforce ordering on
  matrix[J, num_feat] x_num;  // numeric features (PCA scores)
  matrix[J, bi_feat] x_bi;    // binary features (0/1)
}
parameters {
  simplex[C] nu;                          // class probabilities (sum to 1)
  matrix<lower=0, upper=1>[bi_feat, C] theta_bi; // P(binary f = 1 | class)

  // Means: ordered for one chosen feature, free for the others
  ordered[C] mu_ordered;                  // mu for order_feat: mu_ordered[1] < mu_ordered[2]
  matrix[num_feat - 1, C] mu_rest;        // remaining numeric-feature means

  matrix<lower=0>[num_feat, C] sigma;     // std devs for numeric features (per feature x class)
}
transformed parameters {
  matrix[num_feat, C] mu; // assembled means
  {
    int idx = 1;
    for (f in 1:num_feat) {
      if (f == order_feat) {
        for (c in 1:C)
          mu[f, c] = mu_ordered[c];
      } else {
        for (c in 1:C)
          mu[f, c] = mu_rest[idx, c];
        idx += 1;
      }
    }
  }
  vector[C] log_nu;

  for (c in 1:C){
    log_nu[c] = log(nu[c]);
  }
}
model {
  // ---- Prior for class prevalence: ~2% in class 1 ----
  {
    vector[C] alpha;
    alpha[1] = 2;
    alpha[2] = 98;
    nu ~ dirichlet(alpha); // mean nu[1] = 2/(2+98) = 0.02
  }

  // Priors for binary-feature probabilities
  for (f in 1:bi_feat)
    for (c in 1:C)
      theta_bi[f, c] ~ beta(2, 2); // avoids extremes; change if you have prior knowledge

  // Priors for numeric-feature means and sds
  mu_ordered ~ normal(0, 1);       // ordering constraint applies here
  to_vector(mu_rest) ~ normal(0, 1);
  for (f in 1:num_feat)
    for (c in 1:C)
      sigma[f, c] ~ normal(0, 1) T[0, ]; // half-normal
  // ---- Likelihood (mixture marginalization) ----
  for (j in 1:J) {
    vector[2] ps;
    for (c in 1:C) {
      real logp_bi = 0;
      for (f in 1:bi_feat)
        logp_bi += x_bi[j, f] * log(theta_bi[f, c]) + (1 - x_bi[j, f]) * log1m(theta_bi[f, c]);

      real logp_num = 0;
      for (f in 1:num_feat)
        logp_num += normal_lpdf(x_num[j, f] | mu[f, c], sigma[f, c]);

      ps[c] = log_nu[c] + logp_bi + logp_num;
    }
    target += log_sum_exp(ps); // marginalize over classes
  }
}
generated quantities {
  matrix[J, bi_feat] x_bi_rep;
  matrix[J, num_feat] x_num_rep;
  matrix[J, C] class_prob; // posterior predictive class probabilities (unnormalized -> normalized below)

  for (j in 1:J) {
    vector[C] ps;
    for (c in 1:C) {
      real logp_bi = 0;
      for (f in 1:bi_feat)
        logp_bi += x_bi[j, f] * log(theta_bi[f, c]) + (1 - x_bi[j, f]) * log1m(theta_bi[f, c]);

      real logp_num = 0;
      for (f in 1:num_feat)
        logp_num += normal_lpdf(x_num[j, f] | mu[f, c], sigma[f, c]);

      ps[c] = log_nu[c] + logp_bi + logp_num;
    }
    // normalized class probabilities
    real denom = log_sum_exp(ps);

    for (c in 1:C){
      class_prob[j, c] = exp(ps[c] - denom);
    }

    // posterior predictive draw (sample class, then features)
    int z = bernoulli_rng(nu[1]) ? 1 : 2;
    for (f in 1:bi_feat)
      x_bi_rep[j, f] = bernoulli_rng(theta_bi[f, z]);
    for (f in 1:num_feat)
      x_num_rep[j, f] = normal_rng(mu[f, z], sigma[f, z]);
  }
}



// data {
//     int<lower=1> J;
//     int<lower=1> C;
//     int<lower=0> num_feat;
//     int<lower=0> bi_feat;
//     matrix[J, num_feat] x_num;
//     matrix[J, bi_feat] x_bi;
// }
// parameters {
//   // ordered[C] raw_nu_ordered;
    
//   // Binary feature probabilities
//   matrix[bi_feat, C] theta_bi_raw;

//   // Continuous feature parameters
//   matrix[num_feat, C] mu;
//   matrix<lower=0>[num_feat, C] sigma;
// }
// transformed parameters{
//   simplex[C] nu;
//   matrix<lower=0, upper=1>[bi_feat, C] theta_bi = inv_logit(theta_bi_raw);

//   // nu = softmax(raw_nu_ordered);
// }
// model{
//   vector[bi_feat] log_bi;
//   vector[num_feat] log_num;
//   array[C] real ps;
  
//   //priors
//   nu ~ beta(.2, 9.8);

//   //if deciding on using simplex
//   // nu ~ dirichlet([2, 98]);
//   // nu ~ dirichlet([0.2, 9.8]); // for weaker prior, but I think we want a strong prior for this RQ
//   // nu ~ dirichlet(rep_vector(1.0, C));

//   for(f in 1:bi_feat){
//     for(c in 1:C){
//       theta_bi_raw[f, c] ~ beta(1, 1);
//     }
//   }

//   for(f in 1:num_feat){
//     for(c in 1:C){
//       mu[f, c] ~ normal(0, 1);
//       // sigma[f, c] ~ normal(0, 1);
//       sigma[f, c] ~ normal(0, 1) T[0, ];
//     }
//   }

//   log_nu[1] = log(nu);
//   log_nu[2] = log1m(nu);

//   for(j in 1:J){
//     for(c in 1:C){
//       real loglike = 0;
//       for(f in 1:bi_feat){
//         log_bi[f] = x_bi[j,f] * log(theta_bi[f,c]) + (1 - x_bi[j,f]) * log1m(theta_bi[f,c]);
//       }

//       for(f in 1:num_feat){
//         log_num[f] = normal_lpdf(x_num[j,f] | mu[f,c], sigma[f,c]);
//       }
//       ps[c] = log_nu[c] + sum(log_bi) + sum(log_num);
//     }
//     target += log_sum_exp(ps); // marginalize latent class
//   }
// }
// generated quantities {
//     matrix[J, bi_feat] x_bi_rep;
//     matrix[J, num_feat] x_num_rep;

//     for (j in 1:J) {
//         // sample latent class for posterior predictive
//         int z;
//         real p_class1 = exp(log_nu[1]); // probability of class 1
//         if (bernoulli_rng(p_class1) == 1)
//             z = 1;
//         else
//             z = 2;

//         // sample replicated binary features
//         for (f in 1:bi_feat)
//             x_bi_rep[j,f] = bernoulli_rng(theta_bi[f,z]);

//         // sample replicated numeric features
//         for (f in 1:num_feat)
//             x_num_rep[j,f] = normal_rng(mu[f,z], sigma[f,z]);
//     }
// }