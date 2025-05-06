#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace RcppArmadillo;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// Sample N x P observations from a Standard
// Multivariate Normal given N observations, a
// vector of P means, and a P x P cov matrix
// [[Rcpp::export]]


arma::vec multi1(int m, const arma::vec& a, const arma::mat& b){
  //     a is row vector
  //     b is (m*m) matrix
  //     a is replaced by the matrix product of a*b
  arma::vec c(m, fill::zeros);
  
  for(int j=0; j<m; j++)
  {
    for(int k=0; k<m; k++)
    {
      c(j) += a(k)*b(k, j);
    }
  }
  return c;
}


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List loop1(int m, int T, arma::vec& phi, const arma::mat& pRS, 
      const arma::mat& gamma, arma::mat& logalp){
  arma::rowvec tmp(m,fill::zeros);
  double sumphi; 
  double lscale;

  lscale = 0;
  for(int i=0; i<T; i++)
  {
	if (i > 0) phi=multi1(m, phi, gamma);
    sumphi=0;
    for(int j=0; j<m; j++)
	  {
	    phi(j) *= pRS(i, j);
        sumphi += phi(j);
  	}
    for(int j=0; j<m; j++)
  	{
		  phi(j) /= sumphi;
	  }
    lscale += log(sumphi);
    for(int j=0; j<m; j++)
  	{
	  	logalp(i,j) = log(phi(j)) + lscale;
	  }
  }
  List ret = List::create(Named("lscale") = lscale , _["logalp"] = logalp);
  return ret;
//  return lscale;
}
	  
	  
	  