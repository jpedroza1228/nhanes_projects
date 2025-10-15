data {
    int<lower=1> J;
    int<lower=1> C;
    int<lower=0> num_feat;
    int<lower=0> bi_feat;
    matrix[J, num_feat] x_num;
    matrix[J, bi_feat] x_bi;
}
parameters {
  real<lower=1e-6, upper=1 - 1e-6> nu;
  // If I wanted more than 2 classes
  // simplex[C] nu;
    
   // Binary feature probabilities
   matrix<lower=1e-6, upper=1 - 1e-6>[bi_feat, C] theta_bi;

   // Continuous feature parameters
   matrix[num_feat, C] mu;
   matrix<lower=0>[num_feat, C] sigma;
}
transformed parameters{
  vector[2] log_nu;
  matrix[bi_feat, C] theta_bi_inv;

  log_nu[1] = log(nu);
  log_nu[2] = log1m(nu);

  for(f in 1:bi_feat){
    for(c in 1:C){
      theta_bi_inv[f, c] = inv_logit(theta_bi[f, c]); 
    }
  }
}
model{
  vector[bi_feat] log_bi;
  vector[num_feat] log_num;
  array[C] real ps;
  

  //priors
  nu ~ beta(.2, 9.8);

  //if deciding on using simplex
  // nu ~ dirichlet([98, 2]);
  // nu ~ dirichlet([9.8, 0.2]) // for weaker prior, but I think we want a strong prior for this RQ
  // nu ~ dirichlet(rep_vector(1.0, C));

  for(f in 1:bi_feat){
    for(c in 1:C){
      theta_bi[f, c] ~ beta(1, 1);
    }
  }

  for(f in 1:num_feat){
    for(c in 1:C){
      mu[f, c] ~ normal(0, 10);
      sigma[f, c] ~ normal(0, 1);
    }
  }

  for(j in 1:J){
    for(c in 1:C){
      real loglike = 0;
      for(f in 1:bi_feat){
        log_bi[f] = x_bi[j,f] * log(theta_bi_inv[f,c]) + (1 - x_bi[j,f]) * log1m(theta_bi_inv[f,c]);
      }

      for(f in 1:num_feat){
        log_num[f] = normal_lpdf(x_num[j,f] | mu[f,c], sigma[f,c]);
      }
      ps[c] = log_nu[c] + sum(log_bi) + sum(log_num);
    }
    target += log_sum_exp(ps); // marginalize latent class
  }
}
// generated quantities {
//     matrix[J, bi_feat] x_bi_rep;
//     matrix[J, num_feat] x_num_rep;

//     for (j in 1:J) {
//         // sample latent class for posterior predictive
//         int z;
//         real p_class1 = exp(log(nu)); // probability of class 1
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